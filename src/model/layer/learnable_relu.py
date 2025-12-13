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

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_features, 1), requires_grad=True)

        self.scales = nn.ParameterList(
            nn.Parameter(torch.empty(out_features, 1), requires_grad=True) for _ in range(k)
        )
        self.shifts = nn.ParameterList(
            nn.Parameter(torch.empty(out_features, 1), requires_grad=True) for _ in range(k)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters.
        """

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        for scale_to_init, shift_to_init in zip(self.scales, self.shifts):
            nn.init.ones_(scale_to_init)
            nn.init.zeros_(shift_to_init)

    
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LearnableReLU layer.

        Applies a linear transformation followed by the addition of
        task-specific learnable ReLU basis functions. Each active basis
        function corresponds to a task introduced during continual
        learning.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """

        no_curr_used_basis_functions = self.no_curr_used_basis_functions

        x = F.linear(x, self.weight, self.bias)
        x_cloned = x.clone()
        for scale, shift in zip(self.scales[:no_curr_used_basis_functions], self.shifts[:no_curr_used_basis_functions]):
            x += scale * torch.relu(x_cloned + shift)

        return x
        