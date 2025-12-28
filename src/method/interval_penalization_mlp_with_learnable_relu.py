import logging
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.method.method_plugin_abc import MethodPluginABC
from src.method.utils import initialize_projection

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

    def forward_with_snapshot(self, x: torch.Tensor, stop_at: str="IntervalActivation") -> torch.Tensor:
        """
        Runs the model forward using parameters and buffers from the previous task snapshot.  
        Used to compare new activations with old-task activations.

        Args:
            x (torch.Tensor): Input tensor.
            stop_at (str, optional): Layer type name at which to stop the forward pass.
                                     Default is "IntervalActivation".

        Returns:
            torch.Tensor: Activations at the stopping point with old parameters/buffers.
        """
        saved_param_datas = {name: param.data for name, param in self.module.named_parameters()}
        saved_buffers = {name: buf for name, buf in self.module.named_buffers()}

        for name, param in self.module.named_parameters():
            param.data = self.old_state["params"][name].clone()
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = self.old_state["buffers"][name].clone()

        out = x
        for layer in self.module.layers:
            if type(layer).__name__ == "LearnableReLU":
                out = layer(out, regularization_mode=True)
            if type(layer).__name__ == stop_at:
                break

        for name, param in self.module.named_parameters():
            param.data = saved_param_datas[name]
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = saved_buffers[name]

        return out.detach()

    @torch.no_grad()
    def snapshot_state(self) -> dict:
        """
        Take a full snapshot of the current model state.  
        Stores both parameters and buffers (detached & cloned).  

        Returns:
            dict: {"params": OrderedDict, "buffers": OrderedDict}
        """
        return {
            "params": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_parameters()),
            "buffers": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_buffers()),
        }


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

        # ------------------------------------------------------------
        # Phase 0: Calculate projection matrix to lower-dimensional space
        # to get hypercubes around inputs to the first layer.
        # ------------------------------------------------------------
        if self.projection_matrix is None:
            first_layer = self.module.layers[0]

            if hasattr(first_layer, "in_features"):
                d = first_layer.in_features
            else:
                d = first_layer.weight.size(1)

            device = next(self.module.parameters()).device
            self.projection_matrix = initialize_projection(k=self.reduced_dim, d=d, device=device)
            log.info(f"Initialized random projection matrix of shape {self.projection_matrix.shape}.")
            
        first_layer_activations = []
        with torch.no_grad():
            for x in self.data_buffer:
                first_layer_activations.append(x.flatten(start_dim=1) @ self.projection_matrix.t())

        z = torch.cat(first_layer_activations, dim=0)
        mu_z, sigma_z = z.mean(0), z.std(0)
        z_lb, z_ub = mu_z - 1.96 * sigma_z, mu_z + 1.96 * sigma_z

        if self.low_dim_inputs_min is None or self.low_dim_inputs_max is None:
            self.low_dim_inputs_min = z_lb
            self.low_dim_inputs_max = z_ub
        else:
            self.low_dim_inputs_min = torch.minimum(self.low_dim_inputs_min, z_lb)
            self.low_dim_inputs_max = torch.maximum(self.low_dim_inputs_max, z_ub)

        # ------------------------------------------------------------
        # Phase 1: Freeze parameters & snapshot old state (InTAct)
        # ------------------------------------------------------------
        self.params_buffer = {}
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                self.params_buffer[name] = p.detach().clone()


        self.old_state = self.snapshot_state()

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

                    for name, p in self.module.named_parameters():
                        if p is curr_W:
                            prev_W = self.params_buffer[name]
                        elif p is curr_b:
                            prev_b = self.params_buffer[name]

                    # Affine drift in projected space
                    # ΔA = (W - W_old) P^T
                    delta_A = (curr_W - prev_W) @ self.projection_matrix.t()
                    delta_b = curr_b - prev_b

                    # Interval arithmetic in z-space
                    z_lb = self.low_dim_inputs_min.to(x.device)
                    z_ub = self.low_dim_inputs_max.to(x.device)

                    delta_A_pos = torch.relu(delta_A)
                    delta_A_neg = torch.relu(-delta_A)

                    lower = (
                        delta_A_pos @ z_lb
                        - delta_A_neg @ z_ub
                        + delta_b
                    )

                    upper = (
                        delta_A_pos @ z_ub
                        - delta_A_neg @ z_lb
                        + delta_b
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