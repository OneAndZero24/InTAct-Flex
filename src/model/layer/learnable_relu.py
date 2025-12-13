import torch
from torch import nn
from torch.nn import functional as F

import math

class LearnableReLU(nn.Module):

    def __init__(self,
        in_features: int,
        out_features: int,
        k: int,
        beta: float = 1.0) -> None:

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
            beta (float): Beta value for Softplus formulation. Default
                value is 1.0.
        """

        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.beta = beta

        self.no_curr_used_basis_functions = 1

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_features, 1), requires_grad=True)

        self.scales = nn.ParameterList(
            nn.Parameter(torch.empty(out_features, 1), requires_grad=True) for _ in range(k)
        )
        self.shift_increments = nn.ParameterList(
            nn.Parameter(torch.empty(out_features, 1))
            for _ in range(k)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layer parameters.
        """

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        for scale_to_init, shift_inc_to_init in zip(self.scales, self.shift_increments):
            nn.init.ones_(scale_to_init)
            nn.init.zeros_(shift_inc_to_init)

    
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

        This method disables gradient updates for the scale and shift
        parameters associated with a specific basis function. It is
        typically used in a continual learning setting to prevent
        modification of basis functions learned for previous tasks
        while allowing new basis functions to be trained.

        Args:
            idx (int): Index of the basis function to freeze.
        """
        self.scales[idx].requires_grad_(False)
        self.shift_increments[idx].requires_grad_(False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with monotone-by-construction ReLU basis functions.
        """

        num_active = self.no_curr_used_basis_functions

        z = F.linear(x, self.weight, self.bias)
        z_fixed = z.clone()

        cumulative_shift = torch.zeros(
            self.out_features, 1, device=z.device, dtype=z.dtype
        )

        for scale, inc in zip(
            self.scales[:num_active],
            self.shift_increments[:num_active],
        ):
            # Positive increment
            delta = F.softplus(inc, beta=self.beta)

            cumulative_shift = cumulative_shift + delta

            z = z + scale * torch.relu(z_fixed + cumulative_shift)

        return z

        