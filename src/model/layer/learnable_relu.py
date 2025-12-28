import torch
from torch import nn
from torch.nn import functional as F

import math

class LearnableReLU(nn.Module):

    def __init__(self,
        in_features: int,
        out_features: int,
        k: int) -> None:

        """
        Linear layer augmented with task-wise learnable ReLU basis functions
        for continual learning.

        This module applies a linear transformation followed by a sum of
        learnable scaled and shifted ReLU basis functions. Each basis
        function is introduced when a new task is added in a continual
        learning (CL) setting, allowing the model to incrementally expand
        its representational capacity without modifying previously learned
        parameters.

        Each basis function is parameterized independently per output
        feature.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            k (int): Maximum number of learnable ReLU basis functions,
                typically corresponding to the maximum number of tasks.
        """

        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.no_curr_used_basis_functions = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True).to(device)
        self.bias = nn.Parameter(torch.empty(1, out_features), requires_grad=True).to(device)

        # Unconstrained parameters
        self.raw_scales = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k)
        )

        # Non-trainable shifts
        self.register_buffer(
            "cum_shifts",
            torch.zeros(k, 1, out_features)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layer parameters.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def cumulative_scales(self) -> torch.Tensor:
        """
        Returns S_i = softplus(b_i), shape (k, 1, out_features)
        """
        raw = torch.stack(list(self.raw_scales), dim=0)
        return F.softplus(raw)
    
    def basis_scales(self) -> torch.Tensor:
        """
        Returns a_i = S_i - S_{i-1}
        """
        S = self.cumulative_scales()
        a = S.clone()
        a[1:] = S[1:] - S[:-1]
        return a

    
    def set_no_used_basis_functions(self, value: int) -> None:
        """
        Set the number of currently active basis functions.

        This method is typically called when a new task is introduced
        in a continual learning setting, enabling an additional
        ReLU basis function while keeping previously learned basis
        functions unchanged.

        Args:
            value (int): Number of basis functions to be used.
        """
        self.no_curr_used_basis_functions = value
    
    def freeze_basis_function(self, idx: int) -> None:
        """
        Freeze a learnable ReLU basis function.

        This method disables gradient updates for the scale
        parameters associated with a specific basis function. It is
        typically used in a continual learning setting to prevent
        modification of basis functions learned for previous tasks
        while allowing new basis functions to be trained.

        Args:
            idx (int): Index of the basis function to freeze.
        """
        self.raw_scales[idx].requires_grad_(False)
    
    @torch.no_grad()
    def anchor_next_shift(self, z, task_id, percentile=0.95):
        P = torch.quantile(z, percentile, dim=0, keepdim=True)
        if task_id == 0:
            self.cum_shifts[0] = P
        else:
            self.cum_shifts[task_id] = torch.maximum(
                P, self.cum_shifts[task_id - 1]
            )

    def min_derivative_interval(self, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
        """
        Worst-case derivative over hypercube.
        """
        x_c = 0.5 * (x_min + x_max)
        x_r = 0.5 * (x_max - x_min)

        mu = F.linear(x_c, self.weight, self.bias)
        rad = F.linear(x_r, self.weight.abs())

        z_wc = mu - rad  # worst-case

        a = self.basis_scales()[:self.no_curr_used_basis_functions]
        c = self.cum_shifts[:self.no_curr_used_basis_functions]

        deriv = torch.zeros_like(z_wc)
        for ai, ci in zip(a, c):
            deriv += ai * (z_wc > ci).float()

        return deriv

    def forward(self, x):
        z = F.linear(x, self.weight, self.bias)
        a = self.basis_scales()[:self.no_curr_used_basis_functions]
        c = self.cum_shifts[:self.no_curr_used_basis_functions]

        out = torch.zeros_like(z)
        for ai, ci in zip(a, c):
            out += ai * torch.relu(z - ci)

        return out

        