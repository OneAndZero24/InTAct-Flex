import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np

# Direct import to avoid triggering src/__init__.py
from model.layer.relu_kan import ReLUKAN, softmin_two


def test_relu_kan_1d_visualization():
    """Test ReLUKAN with 1D input and k=1, visualizing all intermediate steps."""
    
    # Setup
    torch.manual_seed(42)
    in_features = 1
    out_features = 1
    k = 1
    
    # Create layer
    layer = ReLUKAN(in_features=in_features, out_features=out_features, k=k)
    
    # Set specific parameter values for clear visualization
    with torch.no_grad():
        layer.w.fill_(1.0)
        layer.a_pos.fill_(0.8)
        layer.t_pos.fill_(-0.5)
        layer.a_neg.fill_(0.6)
        layer.t_neg.fill_(0.5)
    
    # Create input range
    x = torch.linspace(-2, 2, 100).unsqueeze(1)  # [100, 1]
    
    # Forward pass with intermediate steps
    print("=" * 60)
    print("ReLU-KAN 1D Visualization Test (k=1)")
    print("=" * 60)
    
    print("\nLayer Parameters:")
    print(f"  w (weight matrix): {layer.w.item():.3f}")
    print(f"  a_pos (positive coefficient): {layer.a_pos.item():.3f}")
    print(f"  t_pos (positive knot): {layer.t_pos.item():.3f}")
    print(f"  a_neg (negative coefficient): {layer.a_neg.item():.3f}")
    print(f"  t_neg (negative knot): {layer.t_neg.item():.3f}")
    
    # Compute intermediate steps
    with torch.no_grad():
        # Step 1: Compute differences
        diff_pos = x[:, :, None] - layer.t_pos[None, :, :]  # [100, 1, 1]
        diff_neg = x[:, :, None] - layer.t_neg[None, :, :]  # [100, 1, 1]
        
        print("\n" + "=" * 60)
        print("Step 1: Compute differences from knot positions")
        print("=" * 60)
        print(f"  diff_pos = x - t_pos = x - ({layer.t_pos.item():.3f})")
        print(f"  diff_neg = x - t_neg = x - ({layer.t_neg.item():.3f})")
        print(f"  Shape: {diff_pos.shape}")
        
        # Step 2: Apply ReLU with coefficients
        phi_pos = torch.nn.functional.relu(layer.a_pos[None, :, :] * diff_pos)
        phi_neg = torch.nn.functional.relu(-layer.a_neg[None, :, :] * diff_neg)
        
        print("\n" + "=" * 60)
        print("Step 2: Apply ReLU with coefficients")
        print("=" * 60)
        print(f"  phi_pos = ReLU(a_pos * diff_pos) = ReLU({layer.a_pos.item():.3f} * diff_pos)")
        print(f"  phi_neg = ReLU(-a_neg * diff_neg) = ReLU({-layer.a_neg.item():.3f} * diff_neg)")
        print(f"  Shape: {phi_pos.shape}")
        
        # Step 3: Sum across k dimension and apply softmin
        phi_pos_sum = phi_pos.sum(dim=2)  # [100, 1]
        phi_neg_sum = phi_neg.sum(dim=2)  # [100, 1]

        phi_sum = softmin_two(phi_pos_sum, phi_neg_sum)
        
        print("\n" + "=" * 60)
        print("Step 3: Sum basis functions and apply softmin")
        print("=" * 60)
        print(f"  phi_pos_sum = sum(phi_pos, dim=2)")
        print(f"  phi_neg_sum = sum(phi_neg, dim=2)")
        print(f"  phi_sum = softmin_two(phi_pos_sum, phi_neg_sum)")
        print(f"  Shape: {phi_sum.shape}")
        
        # Step 4: Linear transformation
        y = phi_sum @ layer.w
        
        print("\n" + "=" * 60)
        print("Step 4: Linear transformation")
        print("=" * 60)
        print(f"  y = phi_sum @ w = phi_sum * {layer.w.item():.3f}")
        print(f"  Shape: {y.shape}")
        
        # Verify forward pass matches
        y_forward = layer(x)
        assert torch.allclose(y, y_forward, atol=1e-6), "Forward pass mismatch!"
        print("\n✓ Forward pass verification successful!")
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('ReLU-KAN 1D Forward Pass: All Intermediate Steps (k=1)', 
                 fontsize=14, fontweight='bold')
    
    x_np = x.squeeze().numpy()
    
    # Plot 1: Differences
    axes[0, 0].plot(x_np, diff_pos.squeeze().numpy(), 'b-', linewidth=2, label='diff_pos')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=layer.t_pos.item(), color='b', linestyle=':', alpha=0.5, 
                       label=f't_pos={layer.t_pos.item():.2f}')
    axes[0, 0].set_xlabel('Input x')
    axes[0, 0].set_ylabel('diff_pos')
    axes[0, 0].set_title('Step 1a: Positive Difference (x - t_pos)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x_np, diff_neg.squeeze().numpy(), 'r-', linewidth=2, label='diff_neg')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=layer.t_neg.item(), color='r', linestyle=':', alpha=0.5,
                       label=f't_neg={layer.t_neg.item():.2f}')
    axes[0, 1].set_xlabel('Input x')
    axes[0, 1].set_ylabel('diff_neg')
    axes[0, 1].set_title('Step 1b: Negative Difference (x - t_neg)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 2: ReLU activations
    axes[1, 0].plot(x_np, phi_pos.squeeze().numpy(), 'b-', linewidth=2, label='phi_pos')
    axes[1, 0].axvline(x=layer.t_pos.item(), color='b', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Input x')
    axes[1, 0].set_ylabel('phi_pos')
    axes[1, 0].set_title(f'Step 2a: Positive ReLU (a_pos={layer.a_pos.item():.2f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x_np, phi_neg.squeeze().numpy(), 'r-', linewidth=2, label='phi_neg')
    axes[1, 1].axvline(x=layer.t_neg.item(), color='r', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Input x')
    axes[1, 1].set_ylabel('phi_neg')
    axes[1, 1].set_title(f'Step 2b: Negative ReLU (a_neg={layer.a_neg.item():.2f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 3: Combined and final output
    axes[2, 0].plot(x_np, phi_pos_sum.squeeze().numpy(), 'b--', linewidth=1.5, 
                    alpha=0.6, label='phi_pos_sum')
    axes[2, 0].plot(x_np, phi_neg_sum.squeeze().numpy(), 'r--', linewidth=1.5, 
                    alpha=0.6, label='phi_neg_sum')
    axes[2, 0].plot(x_np, phi_sum.squeeze().numpy(), 'purple', linewidth=2, 
                    label='phi_sum (combined)')
    axes[2, 0].set_xlabel('Input x')
    axes[2, 0].set_ylabel('Activation')
    axes[2, 0].set_title('Step 3: Combined via softmin_two')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(x_np, y.squeeze().numpy(), 'g-', linewidth=2, label='output')
    axes[2, 1].set_xlabel('Input x')
    axes[2, 1].set_ylabel('Output y')
    axes[2, 1].set_title(f'Step 4: Final Output (w={layer.w.item():.2f})')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mikser/InTAct-Flex/tests/relu_kan_1d_visualization.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: tests/relu_kan_1d_visualization.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def test_relu_kan_basic():
    """Basic functionality test for ReLUKAN."""
    
    print("\n" + "=" * 60)
    print("Basic Functionality Test")
    print("=" * 60)
    
    # Test various configurations
    configs = [
        (2, 3, 1),
        (5, 5, 3),
        (10, 1, 5),
    ]
    
    for in_feat, out_feat, k in configs:
        layer = ReLUKAN(in_features=in_feat, out_features=out_feat, k=k)
        x = torch.randn(4, in_feat)  # batch_size=4
        y = layer(x)
        
        assert y.shape == (4, out_feat), f"Output shape mismatch: expected (4, {out_feat}), got {y.shape}"
        assert not torch.isnan(y).any(), "NaN detected in output"
        assert not torch.isinf(y).any(), "Inf detected in output"
        
        print(f"✓ Config (in={in_feat}, out={out_feat}, k={k}): "
              f"Input {x.shape} -> Output {y.shape}")
    
    print("\n✓ All basic tests passed!")


def test_relu_kan_gradient():
    """Test that gradients flow properly through ReLUKAN."""
    
    print("\n" + "=" * 60)
    print("Gradient Flow Test")
    print("=" * 60)
    
    layer = ReLUKAN(in_features=3, out_features=2, k=2)
    x = torch.randn(5, 3, requires_grad=True)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        print(f"✓ {name:15s}: grad_mean={param.grad.mean().item():8.4f}, "
              f"grad_std={param.grad.std().item():.4f}")
    
    # Check input gradient
    assert x.grad is not None, "No gradient for input"
    print(f"✓ {'input':15s}: grad_mean={x.grad.mean().item():8.4f}, "
          f"grad_std={x.grad.std().item():.4f}")
    
    print("\n✓ Gradient flow test passed!")


if __name__ == "__main__":
    test_relu_kan_1d_visualization()
    test_relu_kan_basic()
    test_relu_kan_gradient()
