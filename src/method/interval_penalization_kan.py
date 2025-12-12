import logging
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class KANIntervalPenalization(MethodPluginABC):
    """
    Continual learning regularizer for ReLU-KAN networks that protects learned 
    function outputs within input domains across tasks.

    This plugin adds multiple penalties to the task loss:
    
    - **Output variance loss (`var_scale`)**  
      Minimizes variance of layer outputs, encouraging stable and compact 
      representations in the learned output space.
    
    - **Knot displacement penalty (`lambda_knot_disp`)**  
      Penalizes movement of knots that lie within the old task's input range,
      while allowing knots outside this range to move freely for adaptation.
    
    - **Boundary consistency loss (`lambda_boundary`)**  
      Ensures the learned function produces similar outputs at the boundaries 
      (min/max) of the old task's input domain.
    
    - **Output interval alignment loss (`lambda_output_align`)**
      Penalizes when new task's output range drifts too far from old task's 
      output range, encouraging output stability.

    Together, these terms protect learned function outputs in critical regions 
    while allowing adaptation to new input distributions.

    Attributes:
        var_scale (float): Weight of output variance regularizer.
        lambda_knot_disp (float): Weight of knot displacement penalty.
        lambda_boundary (float): Weight of boundary consistency loss.
        lambda_output_align (float): Weight of output interval alignment loss.
        task_id (int): Identifier of the current task.
        params_buffer (dict): Snapshot of parameters from previous task.
        input_ranges (dict): Min/max input values per layer from previous task.
        output_ranges (dict): Min/max output values per layer from previous task.
        x_buffer (list): Temporary buffer to accumulate inputs during task.

    Methods:
        setup_task(task_id):
            Prepares state before starting a new task (snapshots params/ranges).
        compute_output_with_params(layer, x, params):
            Evaluates layer output for given inputs using specified parameters.
        snapshot_state():
            Creates a snapshot of all parameters.
        forward(x, y, loss, preds):
            Adds KAN-specific interval regularization terms to the given loss.
    """

    def __init__(self,
            var_scale: float = 0.01,
            lambda_knot_disp: float = 1.0,
            lambda_boundary: float = 1.0,
            lambda_output_align: float = 1.0,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the KAN interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of output variance penalty. Default: 0.01.
            lambda_knot_disp (float, optional): Weight of knot displacement penalty. Default: 1.0.
            lambda_boundary (float, optional): Weight of boundary consistency penalty. Default: 1.0.
            lambda_output_align (float, optional): Weight of output alignment penalty. Default: 1.0.
            dil_mode (bool, optional): If True, classifier head is regularized.
            regularize_classifier (bool, optional): If True, classifier head is regularized. Default: False.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"KANIntervalPenalization initialized with var_scale={var_scale}, "
                 f"lambda_knot_disp={lambda_knot_disp}, "
                 f"lambda_boundary={lambda_boundary}, "
                 f"lambda_output_align={lambda_output_align}")

        self.var_scale = var_scale
        self.lambda_knot_disp = lambda_knot_disp
        self.lambda_boundary = lambda_boundary
        self.lambda_output_align = lambda_output_align

        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier
        self.params_buffer = {}
        self.input_ranges = {}
        self.output_ranges = {}
        self.x_buffer = []

    def compute_output_with_params(self, layer, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Compute final output for ReLU-KAN layer using specified parameters.
        
        This evaluates the complete layer output (phi @ W^T) using the specified 
        parameters without modifying the layer's actual parameters.

        Args:
            layer: ReLU-KAN layer module.
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].
            params (dict): Dictionary containing 'a_pos', 't_pos', 'a_neg', 't_neg', 'w'.

        Returns:
            torch.Tensor: Layer output of shape [batch_size, out_features].
        """
        a_pos = params['a_pos']
        t_pos = params['t_pos']
        a_neg = params['a_neg']
        t_neg = params['t_neg']
        w = params['w']

        diff_pos = x[:, :, None] - t_pos[None, :, :]
        diff_neg = x[:, :, None] - t_neg[None, :, :]
        phi_pos = F.relu(a_pos[None, :, :] * diff_pos)
        phi_neg = F.relu(-a_neg[None, :, :] * diff_neg)
        
        # Use softmin_two from relu_kan module
        from src.model.layer.relu_kan import softmin_two
        phi_sum = softmin_two(phi_pos.sum(dim=2), phi_neg.sum(dim=2))
        
        # Apply weight matrix to get final output
        output = phi_sum @ w.T
        
        return output

    @torch.no_grad()
    def snapshot_state(self) -> dict:
        """
        Take a full snapshot of the current model parameters.

        Returns:
            dict: OrderedDict of parameters (detached & cloned).
        """
        return OrderedDict((k, v.detach().clone()) for k, v in self.module.named_parameters())

    def setup_task(self, task_id: int) -> None:
        """
        Prepare the plugin for a new task.

        - Task 0: only sets `task_id` and initializes buffers.
        - Task >0: saves parameter snapshot, computes input ranges and output ranges 
          from accumulated inputs in x_buffer.

        Args:
            task_id (int): Identifier for the current task.
        """
        self.task_id = task_id
        
        if task_id > 0:
            # Snapshot parameters
            self.params_buffer = self.snapshot_state()
            
            # Compute input and output ranges from accumulated data
            self.input_ranges = {}
            self.output_ranges = {}
            
            if len(self.x_buffer) > 0:
                # Stack all accumulated inputs
                all_x = torch.cat(self.x_buffer, dim=0)
                
                # Collect ReLU-KAN layers
                kan_layers = []
                for layer in self.module.layers:
                    if type(layer).__name__ == "ReLUKAN":
                        kan_layers.append(layer)
                if self.regularize_classifier or self.dil_mode:
                    if hasattr(self.module.head, 'classifier') and type(self.module.head.classifier).__name__ == "ReLUKAN":
                        kan_layers.append(self.module.head.classifier)
                
                # Forward through network to collect ranges
                self.module.eval()
                with torch.no_grad():
                    x_current = all_x.flatten(start_dim=1)
                    
                    for idx, layer in enumerate(self.module.layers):
                        if type(layer).__name__ == "ReLUKAN":
                            # Store input range for this layer
                            x_min = x_current.min(dim=0)[0]
                            x_max = x_current.max(dim=0)[0]
                            self.input_ranges[idx] = {'min': x_min, 'max': x_max}
                            
                            # Forward through layer to get output
                            output = layer(x_current)
                            
                            # Store output range
                            output_min = output.min(dim=0)[0]
                            output_max = output.max(dim=0)[0]
                            self.output_ranges[idx] = {'min': output_min, 'max': output_max}
                            
                            x_current = output
                        else:
                            x_current = layer(x_current)
                    
                    # Handle classifier if needed
                    if (self.regularize_classifier or self.dil_mode) and hasattr(self.module.head, 'classifier'):
                        if type(self.module.head.classifier).__name__ == "ReLUKAN":
                            head_idx = len(self.module.layers)
                            x_min = x_current.min(dim=0)[0]
                            x_max = x_current.max(dim=0)[0]
                            self.input_ranges[head_idx] = {'min': x_min, 'max': x_max}
                            
                            output = self.module.head.classifier(x_current)
                            output_min = output.min(dim=0)[0]
                            output_max = output.max(dim=0)[0]
                            self.output_ranges[head_idx] = {'min': output_min, 'max': output_max}

            self.module.train()
        
        # Reset buffer for new task
        self.x_buffer = []

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, 
                preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add KAN-specific interval regularization penalties to the current loss.

        Penalties:
            - Variance loss: discourages variance in layer outputs.
            - Knot displacement: penalizes knot movement within old input ranges.
            - Boundary consistency: ensures output consistency at range boundaries.
            - Output alignment: keeps new output ranges close to old output ranges.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target labels (unused here, passed through).
            loss (torch.Tensor): Current task loss.
            preds (torch.Tensor): Model predictions.

        Returns:
            (loss, preds): Updated loss with added penalties, predictions unchanged.
        """
        # Accumulate inputs for range computation
        self.x_buffer.append(x.detach().cpu())

        var_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        knot_disp_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        boundary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        output_align_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Collect ReLU-KAN layers
        kan_layers = []
        kan_indices = []
        for idx, layer in enumerate(self.module.layers):
            if type(layer).__name__ == "ReLUKAN":
                kan_layers.append(layer)
                kan_indices.append(idx)
        
        if self.regularize_classifier or self.dil_mode:
            if hasattr(self.module.head, 'classifier') and type(self.module.head.classifier).__name__ == "ReLUKAN":
                kan_layers.append(self.module.head.classifier)
                kan_indices.append(len(self.module.layers))

        # Forward through network to get current outputs
        x_current = x.flatten(start_dim=1)
        layer_idx = 0
        
        for idx, layer in enumerate(self.module.layers):
            if type(layer).__name__ == "ReLUKAN":
                # Current params
                current_params = {
                    'a_pos': layer.a_pos,
                    't_pos': layer.t_pos,
                    'a_neg': layer.a_neg,
                    't_neg': layer.t_neg,
                    'w': layer.w
                }
                
                # Compute output with current params
                output_current = self.compute_output_with_params(layer, x_current, current_params)
                
                # 1. Variance loss on output
                output_flat = output_current.view(output_current.size(0), -1)
                batch_var = output_flat.var(dim=0, unbiased=False).mean()
                var_loss += batch_var

                if self.task_id > 0 and kan_indices[layer_idx] in self.input_ranges:
                    old_idx = kan_indices[layer_idx]
                    x_min = self.input_ranges[old_idx]['min'].to(x.device)
                    x_max = self.input_ranges[old_idx]['max'].to(x.device)
                    
                    # Get old params for this layer
                    old_params = {}
                    for name, param in self.module.named_parameters():
                        if param is layer.a_pos:
                            old_params['a_pos'] = self.params_buffer[name]
                        elif param is layer.t_pos:
                            old_params['t_pos'] = self.params_buffer[name]
                        elif param is layer.a_neg:
                            old_params['a_neg'] = self.params_buffer[name]
                        elif param is layer.t_neg:
                            old_params['t_neg'] = self.params_buffer[name]
                        elif param is layer.w:
                            old_params['w'] = self.params_buffer[name]
                    
                    if len(old_params) == 5:
                        # 2. Knot displacement penalty (weighted by being inside range)
                        t_pos_inside = ((layer.t_pos >= x_min.unsqueeze(1)) & 
                                       (layer.t_pos <= x_max.unsqueeze(1))).float()
                        t_neg_inside = ((layer.t_neg >= x_min.unsqueeze(1)) & 
                                       (layer.t_neg <= x_max.unsqueeze(1))).float()
                        
                        knot_disp_loss += (t_pos_inside * (layer.t_pos - old_params['t_pos']).pow(2)).sum()
                        knot_disp_loss += (t_neg_inside * (layer.t_neg - old_params['t_neg']).pow(2)).sum()
                        
                        # 3. Boundary consistency loss
                        # Evaluate outputs at boundaries: x_min, x_max
                        x_min_batch = x_min.unsqueeze(0)  # [1, in_features]
                        x_max_batch = x_max.unsqueeze(0)  # [1, in_features]
                        
                        output_old_min = self.compute_output_with_params(layer, x_min_batch, old_params)
                        output_new_min = self.compute_output_with_params(layer, x_min_batch, current_params)
                        output_old_max = self.compute_output_with_params(layer, x_max_batch, old_params)
                        output_new_max = self.compute_output_with_params(layer, x_max_batch, current_params)
                        
                        boundary_loss += (output_old_min - output_new_min).pow(2).mean()
                        boundary_loss += (output_old_max - output_new_max).pow(2).mean()
                        
                        # 4. Output interval alignment loss
                        if old_idx in self.output_ranges:
                            output_min_old = self.output_ranges[old_idx]['min'].to(x.device)
                            output_max_old = self.output_ranges[old_idx]['max'].to(x.device)
                            
                            output_min_new = output_current.min(dim=0)[0]
                            output_max_new = output_current.max(dim=0)[0]
                            
                            # Penalize distance between interval centers
                            center_old = (output_min_old + output_max_old) / 2.0
                            center_new = (output_min_new + output_max_new) / 2.0
                            
                            # Normalize by old interval width
                            width_old = (output_max_old - output_min_old) + 1e-8
                            
                            output_align_loss += ((center_new - center_old).pow(2) / width_old).mean()
                
                layer_idx += 1
                x_current = layer(x_current)
            else:
                x_current = layer(x_current)
        
        # Handle classifier if needed
        if (self.regularize_classifier or self.dil_mode) and hasattr(self.module.head, 'classifier'):
            if type(self.module.head.classifier).__name__ == "ReLUKAN":
                layer = self.module.head.classifier
                
                current_params = {
                    'a_pos': layer.a_pos,
                    't_pos': layer.t_pos,
                    'a_neg': layer.a_neg,
                    't_neg': layer.t_neg,
                    'w': layer.w
                }
                
                output_current = self.compute_output_with_params(layer, x_current, current_params)
                output_flat = output_current.view(output_current.size(0), -1)
                batch_var = output_flat.var(dim=0, unbiased=False).mean()
                var_loss += batch_var
                
                if self.task_id > 0 and kan_indices[layer_idx] in self.input_ranges:
                    old_idx = kan_indices[layer_idx]
                    x_min = self.input_ranges[old_idx]['min'].to(x.device)
                    x_max = self.input_ranges[old_idx]['max'].to(x.device)
                    
                    old_params = {}
                    for name, param in self.module.named_parameters():
                        if param is layer.a_pos:
                            old_params['a_pos'] = self.params_buffer[name]
                        elif param is layer.t_pos:
                            old_params['t_pos'] = self.params_buffer[name]
                        elif param is layer.a_neg:
                            old_params['a_neg'] = self.params_buffer[name]
                        elif param is layer.t_neg:
                            old_params['t_neg'] = self.params_buffer[name]
                        elif param is layer.w:
                            old_params['w'] = self.params_buffer[name]
                    
                    if len(old_params) == 5:
                        t_pos_inside = ((layer.t_pos >= x_min.unsqueeze(1)) & 
                                       (layer.t_pos <= x_max.unsqueeze(1))).float()
                        t_neg_inside = ((layer.t_neg >= x_min.unsqueeze(1)) & 
                                       (layer.t_neg <= x_max.unsqueeze(1))).float()
                        
                        knot_disp_loss += (t_pos_inside * (layer.t_pos - old_params['t_pos']).pow(2)).sum()
                        knot_disp_loss += (t_neg_inside * (layer.t_neg - old_params['t_neg']).pow(2)).sum()
                        
                        x_min_batch = x_min.unsqueeze(0)
                        x_max_batch = x_max.unsqueeze(0)
                        
                        output_old_min = self.compute_output_with_params(layer, x_min_batch, old_params)
                        output_new_min = self.compute_output_with_params(layer, x_min_batch, current_params)
                        output_old_max = self.compute_output_with_params(layer, x_max_batch, old_params)
                        output_new_max = self.compute_output_with_params(layer, x_max_batch, current_params)
                        
                        boundary_loss += (output_old_min - output_new_min).pow(2).mean()
                        boundary_loss += (output_old_max - output_new_max).pow(2).mean()
                        
                        if old_idx in self.output_ranges:
                            output_min_old = self.output_ranges[old_idx]['min'].to(x.device)
                            output_max_old = self.output_ranges[old_idx]['max'].to(x.device)
                            
                            output_min_new = output_current.min(dim=0)[0]
                            output_max_new = output_current.max(dim=0)[0]
                            
                            center_old = (output_min_old + output_max_old) / 2.0
                            center_new = (output_min_new + output_max_new) / 2.0
                            width_old = (output_max_old - output_min_old) + 1e-8
                            
                            output_align_loss += ((center_new - center_old).pow(2) / width_old).mean()

        # Debug logging
        if self.task_id > 0:
            log.debug(f"Task {self.task_id} - Regularization losses:")
            log.debug(f"  Variance: {var_loss.item():.6f} (weighted: {(self.var_scale * var_loss).item():.6f})")
            log.debug(f"  Knot displacement: {knot_disp_loss.item():.6f} (weighted: {(self.lambda_knot_disp * knot_disp_loss).item():.6f})")
            log.debug(f"  Boundary: {boundary_loss.item():.6f} (weighted: {(self.lambda_boundary * boundary_loss).item():.6f})")
            log.debug(f"  Output alignment: {output_align_loss.item():.6f} (weighted: {(self.lambda_output_align * output_align_loss).item():.6f})")
            log.debug(f"  Base task loss: {loss.item():.6f}")
        
        loss = (
            loss
            + self.var_scale * var_loss
            + self.lambda_knot_disp * knot_disp_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_output_align * output_align_loss
        )
        
        if self.task_id > 0:
            log.debug(f"  Total loss: {loss.item():.6f}")
        
        return loss, preds
