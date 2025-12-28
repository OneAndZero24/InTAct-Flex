import torch

def initialize_projection(k: int, d: int, device: torch.device) -> torch.Tensor:
    """
    Initialize a random orthogonal projection matrix.
    Args:
        k (int): Target projection dimension.
        d (int): Original data dimension.
        device (torch.device): Device to create the tensor on.
    Returns:
        torch.Tensor: Orthogonal projection matrix of shape (k, d).
    """
    Q, _ = torch.linalg.qr(torch.randn(d, k, device=device))
    return Q.t()