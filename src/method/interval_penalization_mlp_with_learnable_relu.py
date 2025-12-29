import logging
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn.functional as F

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class MLPWithLearnableReLUIntervalPenalization(MethodPluginABC):
    """
    Continual learning regularizer for MLPs that protects learned representations 
    inside activation hypercubes across tasks.

    This plugin adds additional penalties to the task loss to reduce representation 
    drift while allowing adaptation outside protected regions:

    Penalties:
        1. **Variance loss (`var_scale`)**  
           Encourages compact and stable activations inside interval regions 
           by minimizing variance across the batch.

        2. **Internal representation drift loss (`lambda_int_drift`)**  
           Limits changes in activations inside previously learned intervals by 
           constraining parameters above `IntervalActivation` layers to preserve 
           outputs for prior tasks. Works for hidden representations.

        3. **Input layer representation drift loss (`lambda_int_input`)**
            Limits changes in activations inside previously learned interval
            in the activation space immediately after the first layer.
            

    Attributes:
        var_scale (float): Weight of the variance regularization term.
        lambda_int_drift (float): Weight of the interval drift preservation term.
        lambda_int_input (float): Weight of the interval input drift preservation term.
        reduced_dim (int): Dimension of the projected subspace used for input hypercubes.
        dil_mode (bool): If True, classifier head is also regularized for class-incremental learning.
        regularize_classifier (bool): If True, applies regularization to classifier head.
        task_id (int | None): Current task index.
        projection_matrix (torch.Tensor | None): Learned subspace projection for interval computation.
        low_dim_inputs_min (torch.Tensor | None): Lower bounds of projected activations per dimension.
        low_dim_inputs_max (torch.Tensor | None): Upper bounds of projected activations per dimension.
        input_mean (torch.Tensor | None): Running global mean of inputs used for centering.
        num_samples (int): Total number of samples seen across tasks for mean computation.
        params_buffer (dict): Snapshot of frozen model parameters from previous tasks.
        data_buffer (list): Stored input batches used for subspace estimation and statistics.

    Methods:
        setup_task(task_id: int) -> None
            Prepare the model for a new task:
                - Freeze old parameters
                - Collect preactivations for LearnableReLU and activations for IntervalActivation layers
                - Compute or merge projection subspace for low-dimensional interval representation
                - Anchor new ReLU hinges beyond old-task support
                - Reset activation hypercubes
                - Activate one additional basis function per LearnableReLU
        forward(x, y, loss, preds) -> Tuple[torch.Tensor, torch.Tensor]
            Adds interval regularization penalties to the provided task loss.
            - Variance loss discourages large variations within intervals
            - Drift loss penalizes changes in old-task activations inside the hypercube
            Returns updated loss along with original predictions.
    """

    def __init__(self,
            var_scale: float = 0.01,
            lambda_int_drift: float = 1.0,
            lambda_int_input: float = 1.0,
            reduced_dim: int = 50,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the interval penalization plugin for continual learning.

        Args:
            var_scale (float, optional): Weight of variance regularization. Default: 0.01.
            lambda_int_drift (float, optional): Weight of interval drift preservation for hidden layers. Default: 1.0.
            lambda_int_input (float, optional): Weight of interval drift preservation for input layer. Default: 1.0.
            reduced_dim (int, optional): Dimension of the random projection space for input hypercubes. Default: 50.
            dil_mode (bool, optional): If True, classifier head is regularized (used in DIL / CIL scenarios). Default: False.
            regularize_classifier (bool, optional): If True, applies penalties to classifier head. Default: False.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with var_scale={var_scale}, "
                 f"lambda_int_drift={lambda_int_drift}")

        self.var_scale = var_scale
        self.lambda_int_drift = lambda_int_drift
        self.lambda_int_input = lambda_int_input
        self.reduced_dim = reduced_dim

        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier

        self.projection_matrix = None
        self.low_dim_inputs_min = None
        self.low_dim_inputs_max = None

        self.input_mean = None  # Global mean (D,)
        self.num_samples = 0
        self.residual_max = None

        self.params_buffer = {}
        self.data_buffer = set()


    def setup_task(self, task_id: int) -> None:
        """
        Prepare the model for a new task.

        Performs:
            1. Freezing and snapshotting previously learned parameters
            2. Collecting preactivations for LearnableReLU layers
            and activations for IntervalActivation layers
            3. Computing low-dimensional projection subspace for input hypercubes
            4. Merging subspaces with previous tasks if task_id > 0
            5. Resetting activation hypercubes for IntervalActivation layers
            6. Anchoring new LearnableReLU hinges and activating additional basis functions

        Args:
            task_id (int): Index of the current task. Zero indicates the first task.
        """

        self.task_id = task_id
        device = next(self.module.parameters()).device

        if task_id == 0:
            self.data_buffer.clear()
            return
        
        self.module.eval()
        device = next(self.module.parameters()).device

        # ------------------------------------------------------------
        # Phase 0: Calculate projection matrix to lower-dimensional space
        # to get hypercubes around inputs to the first layer.
        # ------------------------------------------------------------
        current_data = torch.cat([x.flatten(start_dim=1) for x in self.data_buffer], dim=0).to(device)

        with torch.no_grad():
            old_mean = self.input_mean.clone() if self.input_mean is not None else torch.zeros(current_data.size(1), device=device)
            if self.input_mean is None:
                self.input_mean = current_data.mean(0)
                self.num_samples = current_data.size(0)
            else:
                n_old, n_new = self.num_samples, current_data.size(0)
                total_samples = n_old + n_new
                new_mean = current_data.mean(0)
                updated_mean = (self.input_mean * n_old + new_mean * n_new) / total_samples
                self.input_mean = updated_mean
                self.num_samples = total_samples

            # Center with global mean
            X_centered = current_data - self.input_mean

            # 2. Extract Task Basis directly from data using SVD
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            actual_dim = min(self.reduced_dim, S.size(0))
            M_task = Vh[:actual_dim, :]  # [k, D]

            if M_task.shape[0] < self.reduced_dim:
                # Pad with random orthogonal vectors if needed
                extra = self.reduced_dim - M_task.shape[0]
                random_extra = torch.randn(extra, current_data.size(1), device=device)
                Q, _ = torch.linalg.qr(random_extra.t())
                extra_basis = Q.t()[:extra]
                M_task = torch.cat([M_task, extra_basis], dim=0)

            if self.projection_matrix is None:
                self.projection_matrix = M_task.detach()
            else:
                # 3. MERGE BASES: Union of Subspaces via SVD for top directions
                B = torch.cat([self.projection_matrix.t(), M_task.t()], dim=1)  # [D, 2k]
                U, S, Vh = torch.linalg.svd(B, full_matrices=False)
                actual_dim = min(self.reduced_dim, S.size(0))
                new_projection = U[:, :actual_dim].t().detach()  # [k, D]
                if new_projection.shape[0] < self.reduced_dim:
                    extra = self.reduced_dim - new_projection.shape[0]
                    random_extra = torch.randn(extra, current_data.size(1), device=device)
                    Q, _ = torch.linalg.qr(random_extra.t())
                    extra_basis = Q.t()[:extra]
                    new_projection = torch.cat([new_projection, extra_basis], dim=0)

                similarity = self.projection_matrix @ new_projection.t()  # [k, k]
                for j in range(self.reduced_dim):
                    i = torch.argmax(torch.abs(similarity[:, j]))
                    if similarity[i, j] < 0:
                        new_projection[j] *= -1

                # Reproject old bounds to new basis (before assigning new projection)
                if self.low_dim_inputs_min is not None:
                    R = self.projection_matrix @ new_projection.t()  # (k, k)
                
                    R_t = R.t() 
                    R_pos = torch.relu(R_t)
                    R_neg = torch.relu(-R_t)

                    # Vectorized Interval Transformation
                    # new_old_min = (R_pos @ old_min - R_neg @ old_max)
                    new_old_min = torch.mv(R_pos, self.low_dim_inputs_min) - torch.mv(R_neg, self.low_dim_inputs_max)
                    new_old_max = torch.mv(R_pos, self.low_dim_inputs_max) - torch.mv(R_neg, self.low_dim_inputs_min)

                    # Adjust for global mean shift
                    shift = (old_mean - self.input_mean) @ new_projection.t()
                    new_old_min += shift
                    new_old_max += shift

                self.projection_matrix = new_projection

            # Project current data to the (possibly updated) basis (using global centering)
            z = X_centered @ self.projection_matrix.t()

            # Reconstruct subspace component: M^T z
            X_proj = z @ self.projection_matrix          # [N, D]

            # Residuals: r = x - mu - M^T z
            R = X_centered - X_proj                      # [N, D]

            # Robust per-dimension bound on |r|
            residual_max_task = torch.quantile(R.abs(), 0.95, dim=0)  # [D]

            if self.residual_max is None:
                self.residual_max = residual_max_task
            else:
                # Union across tasks (worst-case)
                self.residual_max = torch.maximum(self.residual_max, residual_max_task)

            # Calculate robust bounds for the current task (e.g., 5th/95th percentiles)
            task_min = torch.quantile(z, 0.05, dim=0)
            task_max = torch.quantile(z, 0.95, dim=0)

            if self.low_dim_inputs_min is None:
                self.low_dim_inputs_min = task_min.to(device)
                self.low_dim_inputs_max = task_max.to(device)
            else:
                self.low_dim_inputs_min = torch.minimum(new_old_min, task_min).to(device)
                self.low_dim_inputs_max = torch.maximum(new_old_max, task_max).to(device)

        # ------------------------------------------------------------
        # Phase 1: Freeze parameters
        # ------------------------------------------------------------
        self.params_buffer = {}
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                self.params_buffer[name] = p.detach().clone()


        # ------------------------------------------------------------
        # Phase 2: Register hooks & collect statistics
        # ------------------------------------------------------------
        preacts = {}
        acts = {}
        hooks = []

        for idx, layer in enumerate(self.module.layers + [self.module.head]):

            if type(layer).__name__ == "LearnableReLU":
                layer.freeze_basis_function(task_id - 1)
                preacts[idx] = []

                def preact_hook(module, inputs, outputs, idx=idx):
                    x = inputs[0]
                    z = F.linear(x, module.weight, module.bias)
                    preacts[idx].append(z.detach())

                hooks.append(layer.register_forward_hook(preact_hook))

            elif type(layer).__name__ == "IntervalActivation":
                acts[idx] = []

                def act_hook(module, inputs, outputs, idx=idx):
                    acts[idx].append(outputs.detach())

                hooks.append(layer.register_forward_hook(act_hook))

        # ------------------------------------------------------------
        # Phase 3: Forward pass over stored data
        # ------------------------------------------------------------
        self.module.eval()
        with torch.no_grad():
            for x in self.data_buffer:
                x = x.to(next(self.module.parameters()).device)
                _ = self.module(x)

        # ------------------------------------------------------------
        # Phase 4: Update activation hypercubes
        # ------------------------------------------------------------
        for idx, layer in enumerate(self.module.layers + [self.module.head]):
            if type(layer).__name__ == "IntervalActivation":
                layer.reset_range(acts[idx])

        # ------------------------------------------------------------
        # Phase 5: Anchor LearnableReLU hinges & activate new basis
        # ------------------------------------------------------------
        for idx, layer in enumerate(self.module.layers + [self.module.head]):
            if type(layer).__name__ == "LearnableReLU":
                z_all = torch.cat(preacts[idx], dim=0)
                layer.anchor_next_shift(
                    z=z_all,
                    task_id=task_id,
                    percentile=0.95,
                )
                layer.set_no_used_basis_functions(task_id + 1)

        # ------------------------------------------------------------
        # Phase 6: Cleanup
        # ------------------------------------------------------------
        for h in hooks:
            h.remove()

        self.module.train()
        self.data_buffer.clear()

                    
    def forward(self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, 
                preds: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Augment task loss with interval-based regularization penalties.

        Penalties applied:
            - **Variance loss:** Penalizes large variance in interval activations
            - **Drift loss:** Penalizes changes in activations for previous tasks
            - **Output regularization:** Optional regularization on classifier head parameters

        Args:
            x (torch.Tensor): Input batch of shape [B, D].
            y (torch.Tensor): Target labels (passed through, not used here).
            loss (torch.Tensor): Current task loss.
            preds (torch.Tensor): Model predictions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated loss including penalties, unchanged predictions.
        """

        self.data_buffer.add(x.detach())

        layers = self.module.layers + [self.module.head]
        interval_act_layers = [layer for layer in layers if type(layer).__name__ == "IntervalActivation"]

        var_loss = torch.tensor(0.0, device=x.device)
        int_drift_loss = torch.tensor(0.0, device=x.device)

        for idx, layer in enumerate(interval_act_layers):

            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lower_bound_reg = torch.tensor(0.0, device=x.device)
                upper_bound_reg = torch.tensor(0.0, device=x.device)
                
                lb = layer.min.to(x.device)
                ub = layer.max.to(x.device)

                if idx == 0:
                    curr_W = self.module.layers[0].weight        # [out, D]
                    curr_b = self.module.layers[0].bias          # [out]

                    prev_W, prev_b = None, None
                    for name, p in self.module.named_parameters():
                        if p is curr_W and name in self.params_buffer:
                            prev_W = self.params_buffer[name]
                        elif p is curr_b and name in self.params_buffer:
                            prev_b = self.params_buffer[name]

                    if prev_W is None or prev_b is None:
                        raise ValueError("Previous parameters for first layer not found")

                    delta_W = curr_W - prev_W                     # [out, D]
                    delta_b = curr_b - prev_b                     # [out]

                    delta_A = delta_W @ self.projection_matrix.t()   # [out, k]

                    delta_W_proj = delta_A @ self.projection_matrix  # [out, D]
                    delta_W_res = delta_W - delta_W_proj             # [out, D]

                    r_max = self.residual_max.to(x.device)           # [D]
                    res_drift_radius = delta_W_res.abs() @ r_max     # [out]

                    delta_mean = delta_W @ self.input_mean.to(x.device)

                    z_lb = self.low_dim_inputs_min.to(x.device)      # [k]
                    z_ub = self.low_dim_inputs_max.to(x.device)      # [k]

                    delta_A_pos = torch.relu(delta_A)
                    delta_A_neg = torch.relu(-delta_A)

                    lower = (
                        delta_A_pos @ z_lb
                        - delta_A_neg @ z_ub
                        + delta_b
                        + delta_mean
                        - res_drift_radius
                    )

                    upper = (
                        delta_A_pos @ z_ub
                        - delta_A_neg @ z_lb
                        + delta_b
                        + delta_mean
                        + res_drift_radius
                    )

                    int_drift_loss += self.lambda_int_input * (lower.sum().pow(2) + upper.sum().pow(2))
                                  
                # Regularize all layers above
                next_layer = layers[2*idx+2]

                if (self.regularize_classifier or self.dil_mode) and hasattr(next_layer, "classifier"):
                    target_module = next_layer.classifier
                else:
                    target_module = next_layer

                if target_module is not None:
                    for name, p in target_module.named_parameters():
                        for mod_name, mod_param in self.module.named_parameters():
                            if mod_param is p and mod_name in self.params_buffer:
                                prev_param = self.params_buffer[mod_name]
                                if "weight" in name:
                                    weight_diff = p - prev_param

                                    weight_diff_pos = torch.relu(weight_diff)
                                    weight_diff_neg = torch.relu(-weight_diff)

                                    lower_bound_reg += (weight_diff_pos @ lb - weight_diff_neg @ ub).sum()
                                    upper_bound_reg += (weight_diff_pos @ ub - weight_diff_neg @ lb).sum()

                                elif "bias" in name:
                                    bias_diff = p - prev_param

                                    lower_bound_reg += bias_diff.sum()
                                    upper_bound_reg += bias_diff.sum()

                    int_drift_loss += self.lambda_int_drift * (lower_bound_reg.pow(2) + upper_bound_reg.pow(2))


        loss = (
            loss
            + self.var_scale * var_loss
            + int_drift_loss
        )
        return loss, preds