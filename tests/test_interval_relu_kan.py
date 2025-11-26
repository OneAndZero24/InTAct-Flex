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
from model.layer.relu_kan import ReLUKAN
from model.layer.interval_activation import IntervalActivation


def test_interval_relu_kan_visualization():
    """Test ReLUKAN with IntervalActivation layer, visualizing learned intervals."""
    
    # Setup
    torch.manual_seed(42)
    in_features = 3
    out_features = 3
    k = 2
    
    # Create ReLUKAN layer
    relu_kan = ReLUKAN(in_features=in_features, out_features=out_features, k=k)
    
    # Create IntervalActivation layer on top
    interval_layer = IntervalActivation(
        layer_name="relu_kan_interval",
        lower_percentile=0.05,
        upper_percentile=0.95,
        use_nonlinear_transform=False  # ReLU-KAN output doesn't need additional activation
    )
    
    print("=" * 70)
    print("ReLU-KAN + Interval Activation Test")
    print("=" * 70)
    print(f"\nArchitecture:")
    print(f"  Input: {in_features} features")
    print(f"  ReLU-KAN: {in_features} → {out_features} (k={k})")
    print(f"  IntervalActivation: percentiles [{interval_layer.lower_percentile:.2f}, "
          f"{interval_layer.upper_percentile:.2f}]")
    
    # Generate training data for "Task 1"
    n_samples = 500
    x_task1 = torch.randn(n_samples, in_features) * 2.0  # Mean 0, std 2
    
    print(f"\nTask 1 Training Data:")
    print(f"  Samples: {n_samples}")
    print(f"  Input range: [{x_task1.min().item():.2f}, {x_task1.max().item():.2f}]")
    
    # Collect activations for interval estimation
    interval_layer.train()
    activations = []
    
    batch_size = 50
    for i in range(0, n_samples, batch_size):
        batch = x_task1[i:i+batch_size]
        with torch.no_grad():
            kan_output = relu_kan(batch)
            interval_output = interval_layer(kan_output)
            activations.append(interval_output)
    
    # Reset range to compute intervals
    interval_layer.reset_range(activations)
    
    print(f"\nLearned Intervals (per neuron):")
    for i in range(out_features):
        print(f"  Neuron {i}: [{interval_layer.min[i].item():.4f}, "
              f"{interval_layer.max[i].item():.4f}] "
              f"(width: {(interval_layer.max[i] - interval_layer.min[i]).item():.4f})")
    
    # Test with new data
    interval_layer.eval()
    n_test = 200
    x_test = torch.linspace(-4, 4, n_test).unsqueeze(1).repeat(1, in_features)
    
    with torch.no_grad():
        kan_output_test = relu_kan(x_test)
        interval_output_test = interval_layer(kan_output_test)
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('ReLU-KAN with Interval Activation: Learned Bounds per Neuron', 
                 fontsize=14, fontweight='bold')
    
    # Plot each output neuron
    for neuron_idx in range(out_features):
        row = neuron_idx // 3
        col = neuron_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Get data for this neuron
        kan_out = kan_output_test[:, neuron_idx].numpy()
        interval_out = interval_output_test[:, neuron_idx].numpy()
        x_range = x_test[:, 0].numpy()
        
        min_val = interval_layer.min[neuron_idx].item()
        max_val = interval_layer.max[neuron_idx].item()
        
        # Plot ReLU-KAN output
        ax.plot(x_range, kan_out, 'b-', linewidth=2, label='ReLU-KAN output', alpha=0.7)
        
        # Plot interval bounds
        ax.axhline(y=min_val, color='r', linestyle='--', linewidth=2, 
                   label=f'Lower bound ({min_val:.3f})', alpha=0.8)
        ax.axhline(y=max_val, color='g', linestyle='--', linewidth=2, 
                   label=f'Upper bound ({max_val:.3f})', alpha=0.8)
        
        # Shade the interval region
        ax.axhspan(min_val, max_val, alpha=0.2, color='yellow', 
                   label='Learned interval')
        
        # Mark points outside interval
        outside_lower = kan_out < min_val
        outside_upper = kan_out > max_val
        if np.any(outside_lower):
            ax.scatter(x_range[outside_lower], kan_out[outside_lower], 
                      color='red', s=20, alpha=0.5, zorder=5)
        if np.any(outside_upper):
            ax.scatter(x_range[outside_upper], kan_out[outside_upper], 
                      color='darkgreen', s=20, alpha=0.5, zorder=5)
        
        ax.set_xlabel('Input value (x[0])', fontsize=10)
        ax.set_ylabel(f'Output neuron {neuron_idx}', fontsize=10)
        ax.set_title(f'Neuron {neuron_idx} - Interval: '
                    f'[{min_val:.3f}, {max_val:.3f}]', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Statistics
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Compute statistics
    with torch.no_grad():
        task1_output = relu_kan(x_task1)
        within_interval = torch.zeros(out_features)
        
        for i in range(out_features):
            in_range = (task1_output[:, i] >= interval_layer.min[i]) & \
                      (task1_output[:, i] <= interval_layer.max[i])
            within_interval[i] = in_range.float().mean() * 100
    
    stats_text = "Task 1 Coverage Statistics:\n"
    for i in range(out_features):
        stats_text += f"  Neuron {i}: {within_interval[i].item():.1f}% of training samples within interval\n"
    
    interval_widths = (interval_layer.max - interval_layer.min).numpy()
    stats_text += f"\nInterval Widths: mean={interval_widths.mean():.4f}, "
    stats_text += f"std={interval_widths.std():.4f}, "
    stats_text += f"min={interval_widths.min():.4f}, max={interval_widths.max():.4f}"
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round', 
                 facecolor='wheat', alpha=0.3))
    
    plt.savefig('/Users/mikser/InTAct-Flex/tests/interval_relu_kan_visualization.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: tests/interval_relu_kan_visualization.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


def test_interval_preservation_across_tasks():
    """Test that intervals preserve Task 1 activations while adapting to Task 2."""
    
    print("\n" + "=" * 70)
    print("Multi-Task Interval Preservation Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    in_features = 2
    out_features = 2
    k = 3
    
    # Create layers
    relu_kan = ReLUKAN(in_features=in_features, out_features=out_features, k=k)
    interval_layer = IntervalActivation(
        layer_name="task_interval",
        lower_percentile=0.1,
        upper_percentile=0.9,
        use_nonlinear_transform=False
    )
    
    # Task 1: Learn from data distribution 1
    print("\n--- Task 1: Training ---")
    n_task1 = 300
    x_task1 = torch.randn(n_task1, in_features) * 1.5
    
    interval_layer.train()
    activations = []
    with torch.no_grad():
        for i in range(0, n_task1, 30):
            batch = x_task1[i:i+30]
            output = interval_layer(relu_kan(batch))
            activations.append(output)
    
    interval_layer.reset_range(activations)
    
    print(f"Task 1 samples: {n_task1}")
    print("Learned intervals:")
    for i in range(out_features):
        print(f"  Neuron {i}: [{interval_layer.min[i].item():.4f}, "
              f"{interval_layer.max[i].item():.4f}]")
    
    # Task 2: Different data distribution
    print("\n--- Task 2: Testing on new distribution ---")
    n_task2 = 300
    x_task2 = torch.randn(n_task2, in_features) * 3.0 + 2.0  # Shifted and wider
    
    interval_layer.eval()
    with torch.no_grad():
        task1_output = interval_layer(relu_kan(x_task1))
        task2_output = interval_layer(relu_kan(x_task2))
    
    # Check preservation for each neuron
    print("\nInterval preservation analysis:")
    for i in range(out_features):
        min_val = interval_layer.min[i].item()
        max_val = interval_layer.max[i].item()
        
        # Task 1 coverage
        task1_in = ((task1_output[:, i] >= interval_layer.min[i]) & 
                   (task1_output[:, i] <= interval_layer.max[i])).float().mean() * 100
        
        # Task 2 coverage
        task2_in = ((task2_output[:, i] >= interval_layer.min[i]) & 
                   (task2_output[:, i] <= interval_layer.max[i])).float().mean() * 100
        
        # Task 2 violations
        task2_below = (task2_output[:, i] < interval_layer.min[i]).float().mean() * 100
        task2_above = (task2_output[:, i] > interval_layer.max[i]).float().mean() * 100
        
        print(f"\n  Neuron {i}:")
        print(f"    Task 1: {task1_in:.1f}% within interval")
        print(f"    Task 2: {task2_in:.1f}% within interval")
        print(f"            {task2_below:.1f}% below, {task2_above:.1f}% above")
    
    # Visualization
    fig, axes = plt.subplots(1, out_features, figsize=(14, 5))
    fig.suptitle('Interval Preservation: Task 1 vs Task 2', 
                 fontsize=14, fontweight='bold')
    
    for i in range(out_features):
        ax = axes[i] if out_features > 1 else axes
        
        # Histograms
        ax.hist(task1_output[:, i].numpy(), bins=30, alpha=0.5, 
               label='Task 1', color='blue', density=True)
        ax.hist(task2_output[:, i].numpy(), bins=30, alpha=0.5, 
               label='Task 2', color='orange', density=True)
        
        # Interval bounds
        ax.axvline(x=interval_layer.min[i].item(), color='red', 
                  linestyle='--', linewidth=2, label='Lower bound')
        ax.axvline(x=interval_layer.max[i].item(), color='green', 
                  linestyle='--', linewidth=2, label='Upper bound')
        
        # Shade interval
        ax.axvspan(interval_layer.min[i].item(), interval_layer.max[i].item(),
                  alpha=0.2, color='yellow')
        
        ax.set_xlabel(f'Activation value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Neuron {i}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mikser/InTAct-Flex/tests/interval_multi_task_comparison.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: tests/interval_multi_task_comparison.png")
    plt.show()
    
    print("\n✓ Multi-task test completed!")


if __name__ == "__main__":
    test_interval_relu_kan_visualization()
    test_interval_preservation_across_tasks()
