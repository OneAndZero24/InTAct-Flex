import logging
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class MLPWithLearnableReLUIntervalPenalization(MethodPluginABC):
    """
    Continual learning regularizer that protects representations learned inside 
    activation hypercubes across tasks for an MLP architecture.

    This plugin adds multiple penalties to the task loss:
    
    - **Variance loss (`var_scale`)**  
      Minimizes activation variance inside each interval, encouraging stable 
      and compact representations.
    
    - **Internal representation drift loss (`lambda_int_drift`)**  
      Constrains parameters above an `IntervalActivation` to keep producing 
      similar outputs for previously learned intervals.

    Together, these terms reduce representation drift inside protected regions, 
    while still allowing free adaptation outside.

    Attributes:
        var_scale (float): Weight of the variance regularizer.
        lambda_int_drift (float): Weight of the output preservation term.
        task_id (int): Identifier of the current task.
        params_buffer (dict): Snapshot of frozen parameters from the previous task.
        old_state (dict): Full parameter/buffer snapshot used for drift comparison.
        data_buffer (set): A buffer to store data samples.
        regularize_classifier (bool): If True, the classifier head is regularized. Default: False.

    Methods:
        setup_task(task_id):
            Prepares state before starting a new task (snapshots old params/buffers).
        forward_with_snapshot(x, stop_at="IntervalActivation"):
            Runs a forward pass with frozen params up to the first IntervalActivation.
        snapshot_state():
            Creates a snapshot of all parameters and buffers.
        forward(x, y, loss, preds):
            Adds interval regularization terms to the given loss.
    """

    def __init__(self,
            var_scale: float = 0.01,
            lambda_int_drift: float = 1.0,
            reduced_dim: int = 50,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Default: 0.01.
            lambda_int_drift (float, optional): Weight of the output preservation penalty. Default: 1.0.
            reduced_dim (int, optional): Dimension of the random projection space for input hypercubes. Default: 50.
            dil_mode (bool, optional): If True, the classifier head is also regularized. If False (TIL/CIL scenarios)
                                        past class neurons should be simply masked without the regularization.
            regularize_classifier (bool, optional): If True, the classifier head is regularized. Default: False.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with var_scale={var_scale}, "
                 f"lambda_int_drift={lambda_int_drift}")

        self.var_scale = var_scale
        self.lambda_int_drift = lambda_int_drift
        self.reduced_dim = reduced_dim

        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier

        self.projection_matrix = None
        self.low_dim_inputs_min = None
        self.low_dim_inputs_max = None

        self.params_buffer = {}
        self.data_buffer = set()

        self.input_mean = None          # (d,)
        self.cov = None                 # (d, d)
        self.num_samples = 0


    def setup_task(self, task_id: int) -> None:
        """
        Prepare the model for a new task.

        For task_id > 0, this method:
        1) Freezes previously learned parameters and snapshots old state
        2) Collects:
        - preactivations (z = Wx + b) for LearnableReLU layers
        - activations for IntervalActivation layers
        3) Anchors new ReLU hinges beyond old-task support
        4) Resets activation hypercubes
        5) Activates one additional basis function per LearnableReLU

        Args:
            task_id (int): Index of the current task.
        """

        self.task_id = task_id

        if task_id == 0:
            self.data_buffer.clear()
            return
        
        self.module.eval()

        # ------------------------------------------------------------
        # Phase 0: Update Basis and Merge Empirical Hypercubes
        # ------------------------------------------------------------
        self.module.eval()
        device = next(self.module.parameters()).device

        # 1. Update Global Mean and Covariance (Cumulative)
        current_data = torch.cat([x.flatten(start_dim=1) for x in self.data_buffer], dim=0).to(device)
        if self.input_mean is None:
            self.input_mean = current_data.mean(0)
            X_centered = current_data - self.input_mean
            self.cov = (X_centered.t() @ X_centered / (X_centered.size(0) - 1))
            self.num_samples = current_data.size(0)
        else:
            # Incremental Update (Correcting for mean shift)
            n_old, n_new = self.num_samples, current_data.size(0)
            total_samples = n_old + n_new
            new_mean = current_data.mean(0)
            updated_mean = (self.input_mean * n_old + new_mean * n_new) / total_samples
            X_new_centered = current_data - updated_mean
            mean_diff = (self.input_mean - updated_mean).unsqueeze(0)
            self.cov = ( (self.cov * (n_old - 1)) +
                        (X_new_centered.t() @ X_new_centered) +
                        (n_old * mean_diff.t() @ mean_diff) ) / (total_samples - 1)
            self.input_mean = updated_mean
            self.num_samples = total_samples

        # 2. Update the basis
        eigvals, eigvecs = torch.linalg.eigh(self.cov)
        new_basis = eigvecs[:, -self.reduced_dim:].t()  # (k, d)

        # Sign flip correction
        signs = torch.sign(new_basis[torch.arange(self.reduced_dim), new_basis.abs().argmax(dim=1)])
        new_basis *= signs.unsqueeze(1)
        Q, _ = torch.linalg.qr(new_basis.t())
        self.projection_matrix = Q.t()

        # 3. Compute overall hypercube using 95% intervals from PC variances (full distribution)
        top_eigvals = eigvals[-self.reduced_dim:]
        sigma = torch.sqrt(torch.clamp(top_eigvals, min=1e-6))
        self.low_dim_inputs_min = -1.96 * sigma
        self.low_dim_inputs_max = 1.96 * sigma

        # ------------------------------------------------------------
        # Phase 1: Freeze parameters & snapshot old state (InTAct)
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
        Add interval regularization penalties to the current loss.  

        Penalties:
            - Variance loss: discourages variance within interval activations.  
            - Drift loss: penalizes change of activations inside the old-task hypercube.  
            - Output reg: discourages parameter updates that break interval consistency.  

        Args:
            x (torch.Tensor): Input tensor.  
            y (torch.Tensor): Target labels (unused here, passed through).  
            loss (torch.Tensor): Current task loss.  
            preds (torch.Tensor): Model predictions.  

        Returns:
            (loss, preds): Updated loss with added penalties, predictions unchanged.
        """

        self.data_buffer.add(x.detach().cpu())

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
                    curr_W = self.module.layers[0].weight
                    curr_b = self.module.layers[0].bias
                    prev_W = None
                    prev_b = None
                    for name, p in self.module.named_parameters():
                        if p is curr_W and name in self.params_buffer:
                            prev_W = self.params_buffer[name]
                        elif p is curr_b and name in self.params_buffer:
                            prev_b = self.params_buffer[name]
                    if prev_W is None or prev_b is None:
                        raise ValueError("Previous parameters for first layer not found in params_buffer")
                    
                    # Affine drift in projected space
                    # Delta A = (W - W_old) P^T
                    delta_A = (curr_W - prev_W) @ self.projection_matrix.t()
                    delta_b = curr_b - prev_b
                    delta_b_effective = delta_b + (curr_W - prev_W) @ self.input_mean

                    # Interval arithmetic in z-space
                    z_lb = self.low_dim_inputs_min.to(x.device)
                    z_ub = self.low_dim_inputs_max.to(x.device)

                    delta_A_pos = torch.relu(delta_A)
                    delta_A_neg = torch.relu(-delta_A)

                    lower = (
                        delta_A_pos @ z_lb
                        - delta_A_neg @ z_ub
                        + delta_b_effective
                    )
                    upper = (
                        delta_A_pos @ z_ub
                        - delta_A_neg @ z_lb
                        + delta_b_effective
                    )
                    int_drift_loss += lower.mean().pow(2) + upper.mean().pow(2)
                    
              
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

                                    lower_bound_reg += (weight_diff_pos @ lb - weight_diff_neg @ ub).mean()
                                    upper_bound_reg += (weight_diff_pos @ ub - weight_diff_neg @ lb).mean()

                                elif "bias" in name:
                                    bias_diff = p - prev_param

                                    lower_bound_reg += bias_diff.mean()
                                    upper_bound_reg += bias_diff.mean()

                    int_drift_loss += lower_bound_reg.pow(2) + upper_bound_reg.pow(2)


        loss = (
            loss
            + self.var_scale * var_loss
            + self.lambda_int_drift * int_drift_loss
        )
        return loss, preds