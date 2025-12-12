import torch
from torch import nn
from torch.nn import functional as F


def softmin_two(x1, x2, beta=100.0):
    """
    Computes a differentiable approximation of min(x1, x2) in a numerically stable way.
    Args:
        x1, x2: tensors of the same shape
        beta: sharpness parameter
    Returns:
        approx_min: tensor of same shape
    """
    stacked = torch.stack([x1, x2], dim=-1)          # shape [..., 2]
    return - torch.logsumexp(-beta * stacked, dim=-1) / beta

class ReLUKAN(nn.Module):
    """ReLU-based Kolmogorov-Arnold Network (KAN) layer.
    
    This layer implements a learnable activation function using bidirectional ReLU 
    basis functions with learnable knot positions and coefficients. It uses separate
    positive and negative ReLU components to capture both increasing and decreasing
    patterns, generalizing linear layers through non-linear transformations.
    
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        k: Number of ReLU basis functions per input feature (for each direction).
    """

    def __init__(self,
        in_features: int,
        out_features: int,
        k: int):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.w = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)

        self.a_pos = nn.Parameter(torch.empty(in_features, k), requires_grad=True)
        self.t_pos = nn.Parameter(torch.empty(in_features, k), requires_grad=True)

        self.a_neg = nn.Parameter(torch.empty(in_features, k), requires_grad=True)
        self.t_neg = nn.Parameter(torch.empty(in_features, k), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters.
        
        Initializes:
        - w: Kaiming uniform initialization for weight matrix
        - a_pos: Uniform initialization in [-1, 1] for positive ReLU coefficients
        - t_pos: Uniform initialization in [-1, 1] for positive knot positions
        - a_neg: Uniform initialization in [-1, 1] for negative ReLU coefficients
        - t_neg: Uniform initialization in [-1, 1] for negative knot positions
        """

        nn.init.kaiming_uniform_(self.w)
        nn.init.uniform_(self.a_pos, 0, 1.0)
        nn.init.uniform_(self.t_pos, -1.0, 1.0)
        nn.init.uniform_(self.a_neg, 0, 1.0)
        nn.init.uniform_(self.t_neg, -1.0, 1.0)

    def forward(self, x):
        """Forward pass through the ReLU-KAN layer.
        
        Computes separate positive and negative ReLU basis function responses,
        sums them to create the feature transformation, then applies a linear
        transformation to produce the output.
        
        Args:
            x: Input tensor of shape [batch_size, in_features].
        
        Returns:
            Output tensor of shape [batch_size, out_features].
        """

        diff_pos = x[:, :, None] - self.t_pos[None, :, :]
        diff_neg = x[:, :, None] - self.t_neg[None, :, :]
        phi_pos = F.relu(self.a_pos[None, :, :] * diff_pos)             # [batch_size, in_features, k]
        phi_neg = F.relu(-self.a_neg[None, :, :] * diff_neg)            # [batch_size, in_features, k]
        phi_sum = softmin_two(phi_pos.sum(dim=2), phi_neg.sum(dim=2))   # [batch_size, in_features]
        y = phi_sum @ self.w.T                                          # [batch_size, out_features]
        return y
        