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
    
    - **Feature loss (`lambda_feat`)**  
      Penalizes deviations of new activations from old-task activations 
      inside the same hypercube, with a stronger penalty near the cube center.

    Together, these terms reduce representation drift inside protected regions, 
    while still allowing free adaptation outside.

    Attributes:
        var_scale (float): Weight of the variance regularizer.
        lambda_int_drift (float): Weight of the output preservation term.
        lambda_feat (float): Weight of the drift regularizer.
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
            lambda_feat: float = 1.0,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Default: 0.01.
            lambda_int_drift (float, optional): Weight of the output preservation penalty. Default: 1.0.
            lambda_feat (float, optional): Weight of the interval drift penalty. Default: 1.0.
            dil_mode (bool, optional): If True, the classifier head is also regularized. If False (TIL/CIL scenarios)
                                        past class neurons should be simply masked without the regularization.
            regularize_classifier (bool, optional): If True, the classifier head is regularized. Default: False.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with var_scale={var_scale}, "
                 f"lambda_int_drift={lambda_int_drift}, "
                 f"lambda_feat={lambda_feat}")

        self.var_scale = var_scale
        self.lambda_int_drift = lambda_int_drift
        self.lambda_feat = lambda_feat

        self.input_shape = None
        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier
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
            out = layer(out)
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
        # Phase 1: Freeze parameters & snapshot old state (InTAct)
        # ------------------------------------------------------------
        self.params_buffer = {}
        for name, p in deepcopy(list(self.module.named_parameters())):
            if p.requires_grad:
                p.requires_grad = False
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

        self.data_buffer.add(x)

        layers = self.module.layers + [self.module.head]
        interval_act_layers = [layer for layer in layers if type(layer).__name__ == "IntervalActivation"]

        var_loss = torch.tensor(0.0, device=x.device)
        int_drift_loss = torch.tensor(0.0, device=x.device)
        feat_loss = torch.tensor(0.0, device=x.device)

        for idx, layer in enumerate(interval_act_layers):

            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layer.min.to(x.device)
                ub = layer.max.to(x.device)

                # Drift only at the FIRST IntervalActivation
                if idx == 0:
                    x = x.flatten(start_dim=1)
                    y_old = self.forward_with_snapshot(x)
                    mask = ((acts >= lb) & (acts <= ub)).float()
                    feat_loss += (
                        (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                    )
              
                # Regularize all layers above
                next_layer = layers[2*idx+2]

                if (self.regularize_classifier or self.dil_mode) and hasattr(next_layer, "classifier"):
                    target_module = next_layer.classifier
                else:
                    target_module = next_layer

                if target_module is not None:
                    lower_bound_reg = torch.tensor(0.0, device=x.device)
                    upper_bound_reg = torch.tensor(0.0, device=x.device)
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

                    int_drift_loss += lower_bound_reg.sum().pow(2) + upper_bound_reg.sum().pow(2)


        loss = (
            loss
            + self.var_scale * var_loss
            + self.lambda_int_drift * int_drift_loss
            + self.lambda_feat * feat_loss
        )
        return loss, preds