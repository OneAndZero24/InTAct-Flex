"""
Comprehensive Visualization Test for KAN Interval Penalization

This test demonstrates how the interval penalization method works for ReLU-KAN networks
in continual learning scenarios. It visualizes all components:
1. ReLU-KAN basis functions (positive and negative ReLU knots)
2. Input/output interval tracking across tasks
3. Four penalty components:
   - Output variance regularization
   - Knot displacement penalty
   - Boundary consistency loss
   - Output interval alignment loss
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Direct imports
from model.layer.relu_kan import ReLUKAN
from method.interval_penalization_kan import KANIntervalPenalization

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def create_simple_kan_model(in_features=3, hidden_features=5, out_features=2, k=3):
    """Create a simple two-layer KAN model for testing."""
    class SimpleKANModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                ReLUKAN(in_features, hidden_features, k),
                ReLUKAN(hidden_features, out_features, k)
            ])
            # Dummy head for compatibility with interval penalization
            self.head = torch.nn.Module()
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    return SimpleKANModel()


def visualize_relu_kan_basis(layer, input_range=(-3, 3), num_points=300):
    """Visualize the ReLU-KAN basis functions for each input dimension."""
    x = torch.linspace(input_range[0], input_range[1], num_points)
    
    fig, axes = plt.subplots(2, layer.in_features, figsize=(5*layer.in_features, 8))
    if layer.in_features == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('ReLU-KAN Basis Functions per Input Feature', fontsize=16, fontweight='bold')
    
    for i in range(layer.in_features):
        # Positive ReLU components
        ax_pos = axes[0, i]
        for j in range(layer.k):
            t_pos = layer.t_pos[i, j].item()
            a_pos = layer.a_pos[i, j].item()
            
            diff = x - t_pos
            phi = torch.relu(a_pos * diff)
            
            ax_pos.plot(x.numpy(), phi.numpy(), label=f'k={j+1}: t={t_pos:.2f}, a={a_pos:.2f}', alpha=0.7)
            ax_pos.axvline(t_pos, color='gray', linestyle='--', alpha=0.3)
        
        ax_pos.set_title(f'Input {i+1}: Positive ReLU Components', fontweight='bold')
        ax_pos.set_xlabel('Input Value')
        ax_pos.set_ylabel('Activation')
        ax_pos.legend(fontsize=8)
        ax_pos.grid(True, alpha=0.3)
        
        # Negative ReLU components
        ax_neg = axes[1, i]
        for j in range(layer.k):
            t_neg = layer.t_neg[i, j].item()
            a_neg = layer.a_neg[i, j].item()
            
            diff = x - t_neg
            phi = torch.relu(-a_neg * diff)
            
            ax_neg.plot(x.numpy(), phi.numpy(), label=f'k={j+1}: t={t_neg:.2f}, a={a_neg:.2f}', alpha=0.7)
            ax_neg.axvline(t_neg, color='gray', linestyle='--', alpha=0.3)
        
        ax_neg.set_title(f'Input {i+1}: Negative ReLU Components', fontweight='bold')
        ax_neg.set_xlabel('Input Value')
        ax_neg.set_ylabel('Activation')
        ax_neg.legend(fontsize=8)
        ax_neg.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_learned_function(layer, input_range=(-3, 3), num_points=300):
    """Visualize the complete learned function output for each neuron."""
    x = torch.linspace(input_range[0], input_range[1], num_points)
    
    # Create inputs: for 1D, just x; for higher dims, repeat x
    if layer.in_features == 1:
        x_input = x.unsqueeze(1)
    else:
        x_input = x.unsqueeze(1).repeat(1, layer.in_features)
    
    with torch.no_grad():
        output = layer(x_input)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(layer.out_features):
        ax.plot(x.numpy(), output[:, i].numpy(), label=f'Output Neuron {i+1}', linewidth=2)
    
    ax.set_title('Complete ReLU-KAN Layer Output Functions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Input Value', fontsize=12)
    ax.set_ylabel('Output Value', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_knot_positions(layer, input_ranges=None, task_names=None):
    """Visualize knot positions with input ranges highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Positive knots
    ax_pos = axes[0]
    for i in range(layer.in_features):
        y_positions = [i] * layer.k
        x_positions = layer.t_pos[i, :].detach().numpy()
        ax_pos.scatter(x_positions, y_positions, s=100, alpha=0.7, label=f'Input {i+1}')
        
    if input_ranges:
        for idx, (task_name, ranges) in enumerate(zip(task_names, input_ranges)):
            x_min = ranges['min'].numpy() if hasattr(ranges['min'], 'numpy') else ranges['min']
            x_max = ranges['max'].numpy() if hasattr(ranges['max'], 'numpy') else ranges['max']
            
            for i in range(len(x_min)):
                rect = Rectangle((x_min[i], i - 0.3), x_max[i] - x_min[i], 0.6,
                               alpha=0.2, color=f'C{idx}', label=f'{task_name} range' if i == 0 else '')
                ax_pos.add_patch(rect)
    
    ax_pos.set_ylabel('Input Feature Index')
    ax_pos.set_xlabel('Knot Position')
    ax_pos.set_title('Positive ReLU Knot Positions', fontweight='bold')
    ax_pos.set_yticks(range(layer.in_features))
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3, axis='x')
    
    # Negative knots
    ax_neg = axes[1]
    for i in range(layer.in_features):
        y_positions = [i] * layer.k
        x_positions = layer.t_neg[i, :].detach().numpy()
        ax_neg.scatter(x_positions, y_positions, s=100, alpha=0.7, label=f'Input {i+1}')
        
    if input_ranges:
        for idx, (task_name, ranges) in enumerate(zip(task_names, input_ranges)):
            x_min = ranges['min'].numpy() if hasattr(ranges['min'], 'numpy') else ranges['min']
            x_max = ranges['max'].numpy() if hasattr(ranges['max'], 'numpy') else ranges['max']
            
            for i in range(len(x_min)):
                rect = Rectangle((x_min[i], i - 0.3), x_max[i] - x_min[i], 0.6,
                               alpha=0.2, color=f'C{idx}', label=f'{task_name} range' if i == 0 else '')
                ax_neg.add_patch(rect)
    
    ax_neg.set_ylabel('Input Feature Index')
    ax_neg.set_xlabel('Knot Position')
    ax_neg.set_title('Negative ReLU Knot Positions', fontweight='bold')
    ax_neg.set_yticks(range(layer.in_features))
    ax_neg.legend()
    ax_neg.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def visualize_output_ranges(output_ranges, layer_idx, task_names):
    """Visualize output ranges across tasks for a specific layer."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    num_neurons = len(output_ranges[0]['min'])
    x_positions = np.arange(num_neurons)
    width = 0.35 / len(output_ranges)
    
    for task_idx, (task_name, ranges) in enumerate(zip(task_names, output_ranges)):
        min_vals = ranges['min'].numpy() if hasattr(ranges['min'], 'numpy') else ranges['min']
        max_vals = ranges['max'].numpy() if hasattr(ranges['max'], 'numpy') else ranges['max']
        heights = max_vals - min_vals
        
        offset = (task_idx - len(output_ranges)/2 + 0.5) * width
        bars = ax.bar(x_positions + offset, heights, width, bottom=min_vals,
                     label=task_name, alpha=0.7)
        
        # Add center markers
        centers = (min_vals + max_vals) / 2
        ax.scatter(x_positions + offset, centers, color='black', s=30, zorder=5)
    
    ax.set_xlabel('Output Neuron', fontsize=12)
    ax.set_ylabel('Activation Range', fontsize=12)
    ax.set_title(f'Layer {layer_idx} Output Ranges Across Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'N{i+1}' for i in range(num_neurons)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def visualize_penalty_components(penalties_over_time):
    """Visualize how different penalty components evolve during training."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Penalty Components During Task 2 Training', fontsize=16, fontweight='bold')
    
    penalties = {
        'Variance Loss': [],
        'Knot Displacement': [],
        'Boundary Consistency': [],
        'Output Alignment': []
    }
    
    for p in penalties_over_time:
        penalties['Variance Loss'].append(p['var_loss'])
        penalties['Knot Displacement'].append(p['knot_disp_loss'])
        penalties['Boundary Consistency'].append(p['boundary_loss'])
        penalties['Output Alignment'].append(p['output_align_loss'])
    
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (name, values) in enumerate(penalties.items()):
        ax = ax_list[idx]
        ax.plot(values, color=colors[idx], linewidth=2)
        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Value')
        ax.grid(True, alpha=0.3)
        
        if len(values) > 0:
            ax.axhline(values[0], color='red', linestyle='--', alpha=0.5, label='Initial')
            ax.axhline(values[-1], color='green', linestyle='--', alpha=0.5, label='Final')
            ax.legend()
    
    plt.tight_layout()
    return fig


def test_kan_interval_penalization_complete_visualization():
    """
    Complete test showing all aspects of KAN interval penalization:
    1. ReLU-KAN basis functions
    2. Learned functions before and after Task 1
    3. Input/output range tracking
    4. Knot positions with protected regions
    5. Training dynamics on Task 2 with all penalty components
    """
    print("=" * 80)
    print("COMPREHENSIVE KAN INTERVAL PENALIZATION VISUALIZATION TEST")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # Configuration
    in_features = 2
    hidden_features = 4
    out_features = 3
    k = 3  # Number of ReLU basis functions
    
    print(f"\nModel Configuration:")
    print(f"  Input: {in_features} features")
    print(f"  Hidden: {hidden_features} neurons (ReLU-KAN, k={k})")
    print(f"  Output: {out_features} neurons (ReLU-KAN, k={k})")
    
    # Create model
    model = create_simple_kan_model(in_features, hidden_features, out_features, k)
    
    # Initialize interval penalization plugin
    plugin = KANIntervalPenalization(
        var_scale=0.01,
        lambda_knot_disp=1.0,
        lambda_boundary=1.0,
        lambda_output_align=1.0,
        dil_mode=False,
        regularize_classifier=False
    )
    plugin.module = model
    
    print("\n" + "=" * 80)
    print("TASK 1: Initial Training")
    print("=" * 80)
    
    # ============= Task 1 =============
    plugin.setup_task(0)
    
    # Visualize initial basis functions
    print("\n[1/9] Visualizing initial ReLU-KAN basis functions...")
    fig1 = visualize_relu_kan_basis(model.layers[0])
    
    # Generate Task 1 data (narrow distribution)
    n_samples_task1 = 200
    x_task1 = torch.randn(n_samples_task1, in_features) * 0.5 + 1.0  # Centered at 1.0, std 0.5
    y_task1 = torch.randint(0, out_features, (n_samples_task1,))
    
    print(f"\nTask 1 Data:")
    print(f"  Samples: {n_samples_task1}")
    print(f"  Input mean: {x_task1.mean(dim=0).numpy()}")
    print(f"  Input std: {x_task1.std(dim=0).numpy()}")
    print(f"  Input range: [{x_task1.min().item():.2f}, {x_task1.max().item():.2f}]")
    
    # Simple training loop for Task 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\nTraining on Task 1...")
    for epoch in range(50):
        optimizer.zero_grad()
        preds = model(x_task1)
        loss = criterion(preds, y_task1)
        
        # Apply plugin (only variance loss in Task 0)
        loss, _ = plugin.forward(x_task1, y_task1, loss, preds)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    # Visualize learned functions after Task 1
    print("\n[2/9] Visualizing learned functions after Task 1...")
    fig2 = visualize_learned_function(model.layers[0])
    
    # Setup Task 2 (this triggers interval capturing)
    print("\n" + "=" * 80)
    print("TASK 2: Learning with Interval Protection")
    print("=" * 80)
    
    plugin.setup_task(1)
    
    # Store Task 1 ranges for visualization
    task1_input_ranges = {}
    task1_output_ranges = {}
    for idx in plugin.input_ranges:
        task1_input_ranges[idx] = {
            'min': plugin.input_ranges[idx]['min'].cpu().clone(),
            'max': plugin.input_ranges[idx]['max'].cpu().clone()
        }
        task1_output_ranges[idx] = {
            'min': plugin.output_ranges[idx]['min'].cpu().clone(),
            'max': plugin.output_ranges[idx]['max'].cpu().clone()
        }
    
    print("\n[3/9] Task 1 captured input ranges:")
    for idx in task1_input_ranges:
        ranges = task1_input_ranges[idx]
        print(f"  Layer {idx}:")
        for i in range(len(ranges['min'])):
            print(f"    Feature {i}: [{ranges['min'][i].item():.3f}, {ranges['max'][i].item():.3f}]")
    
    print("\n[4/9] Task 1 captured output ranges:")
    for idx in task1_output_ranges:
        ranges = task1_output_ranges[idx]
        print(f"  Layer {idx}:")
        for i in range(len(ranges['min'])):
            print(f"    Neuron {i}: [{ranges['min'][i].item():.3f}, {ranges['max'][i].item():.3f}]")
    
    # Visualize knot positions with Task 1 ranges
    print("\n[5/9] Visualizing knot positions with protected regions...")
    fig3 = visualize_knot_positions(
        model.layers[0],
        input_ranges=[task1_input_ranges[0]],
        task_names=['Task 1']
    )
    
    # Generate Task 2 data (different distribution)
    n_samples_task2 = 200
    x_task2 = torch.randn(n_samples_task2, in_features) * 0.8 - 1.0  # Centered at -1.0, std 0.8
    y_task2 = torch.randint(0, out_features, (n_samples_task2,))
    
    print(f"\nTask 2 Data:")
    print(f"  Samples: {n_samples_task2}")
    print(f"  Input mean: {x_task2.mean(dim=0).numpy()}")
    print(f"  Input std: {x_task2.std(dim=0).numpy()}")
    print(f"  Input range: [{x_task2.min().item():.2f}, {x_task2.max().item():.2f}]")
    
    # Track penalties during Task 2 training
    penalties_over_time = []
    
    print("\nTraining on Task 2 with interval penalization...")
    for epoch in range(50):
        optimizer.zero_grad()
        preds = model(x_task2)
        loss_base = criterion(preds, y_task2)
        
        # Compute penalties
        loss_with_penalties, _ = plugin.forward(x_task2, y_task2, loss_base, preds)
        
        # Extract individual penalty components for visualization
        with torch.no_grad():
            # Recompute to extract components
            var_loss = torch.tensor(0.0)
            knot_disp_loss = torch.tensor(0.0)
            boundary_loss = torch.tensor(0.0)
            output_align_loss = torch.tensor(0.0)
            
            x_current = x_task2
            for idx, layer in enumerate(model.layers):
                if type(layer).__name__ == "ReLUKAN":
                    output = layer(x_current)
                    var_loss += output.var(dim=0).mean()
                    
                    if idx in plugin.input_ranges:
                        x_min = plugin.input_ranges[idx]['min']
                        x_max = plugin.input_ranges[idx]['max']
                        
                        # Knot displacement
                        old_params = {}
                        for name, param in model.named_parameters():
                            if param is layer.a_pos:
                                old_params['a_pos'] = plugin.params_buffer[name]
                            elif param is layer.t_pos:
                                old_params['t_pos'] = plugin.params_buffer[name]
                            elif param is layer.a_neg:
                                old_params['a_neg'] = plugin.params_buffer[name]
                            elif param is layer.t_neg:
                                old_params['t_neg'] = plugin.params_buffer[name]
                            elif param is layer.w:
                                old_params['w'] = plugin.params_buffer[name]
                        
                        if len(old_params) == 5:
                            t_pos_inside = ((layer.t_pos >= x_min.unsqueeze(1)) & 
                                          (layer.t_pos <= x_max.unsqueeze(1))).float()
                            t_neg_inside = ((layer.t_neg >= x_min.unsqueeze(1)) & 
                                          (layer.t_neg <= x_max.unsqueeze(1))).float()
                            
                            knot_disp_loss += (t_pos_inside * (layer.t_pos - old_params['t_pos']).pow(2)).sum()
                            knot_disp_loss += (t_neg_inside * (layer.t_neg - old_params['t_neg']).pow(2)).sum()
                        
                            # Boundary consistency - actual computation
                            current_params = {
                                'a_pos': layer.a_pos,
                                't_pos': layer.t_pos,
                                'a_neg': layer.a_neg,
                                't_neg': layer.t_neg,
                                'w': layer.w
                            }
                            
                            x_min_batch = x_min.unsqueeze(0)
                            x_max_batch = x_max.unsqueeze(0)
                            
                            output_old_min = plugin.compute_output_with_params(layer, x_min_batch, old_params)
                            output_new_min = plugin.compute_output_with_params(layer, x_min_batch, current_params)
                            output_old_max = plugin.compute_output_with_params(layer, x_max_batch, old_params)
                            output_new_max = plugin.compute_output_with_params(layer, x_max_batch, current_params)
                            
                            boundary_loss += (output_old_min - output_new_min).pow(2).mean()
                            boundary_loss += (output_old_max - output_new_max).pow(2).mean()
                        
                        # Output alignment
                        if idx in plugin.output_ranges:
                            output_min_old = plugin.output_ranges[idx]['min']
                            output_max_old = plugin.output_ranges[idx]['max']
                            output_min_new = output.min(dim=0)[0]
                            output_max_new = output.max(dim=0)[0]
                            
                            center_old = (output_min_old + output_max_old) / 2.0
                            center_new = (output_min_new + output_max_new) / 2.0
                            width_old = (output_max_old - output_min_old) + 1e-8
                            
                            output_align_loss += ((center_new - center_old).pow(2) / width_old).mean()
                    
                    x_current = output
            
            penalties_over_time.append({
                'var_loss': var_loss.item(),
                'knot_disp_loss': knot_disp_loss.item(),
                'boundary_loss': boundary_loss.item(),
                'output_align_loss': output_align_loss.item()
            })
        
        loss_with_penalties.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {loss_with_penalties.item():.4f}")
    
    # Visualize penalty components
    print("\n[6/9] Visualizing penalty components during training...")
    fig4 = visualize_penalty_components(penalties_over_time)
    
    # Capture Task 2 ranges
    plugin.setup_task(2)
    
    task2_input_ranges = {}
    task2_output_ranges = {}
    for idx in plugin.input_ranges:
        task2_input_ranges[idx] = {
            'min': plugin.input_ranges[idx]['min'].cpu().clone(),
            'max': plugin.input_ranges[idx]['max'].cpu().clone()
        }
        task2_output_ranges[idx] = {
            'min': plugin.output_ranges[idx]['min'].cpu().clone(),
            'max': plugin.output_ranges[idx]['max'].cpu().clone()
        }
    
    # Visualize updated knot positions
    print("\n[7/9] Visualizing knot positions after Task 2...")
    fig5 = visualize_knot_positions(
        model.layers[0],
        input_ranges=[task1_input_ranges[0], task2_input_ranges[0]],
        task_names=['Task 1', 'Task 2']
    )
    
    # Compare output ranges across tasks
    print("\n[8/9] Visualizing output range evolution...")
    fig6 = visualize_output_ranges(
        [task1_output_ranges[0], task2_output_ranges[0]],
        layer_idx=0,
        task_names=['Task 1', 'Task 2']
    )
    
    # Final learned functions
    print("\n[9/9] Visualizing learned functions after Task 2...")
    fig7 = visualize_learned_function(model.layers[0])
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\nKnot Movement Analysis:")
    for idx in range(model.layers[0].in_features):
        print(f"\n  Input Feature {idx}:")
        
        # Positive knots
        for name, param in model.named_parameters():
            if param is model.layers[0].t_pos:
                old_t_pos = plugin.params_buffer[name][idx, :].numpy()
                new_t_pos = model.layers[0].t_pos[idx, :].detach().numpy()
                
                x_min = task1_input_ranges[0]['min'][idx].item()
                x_max = task1_input_ranges[0]['max'][idx].item()
                
                print(f"    Positive knots:")
                for k in range(len(old_t_pos)):
                    was_inside = x_min <= old_t_pos[k] <= x_max
                    displacement = abs(new_t_pos[k] - old_t_pos[k])
                    status = "PROTECTED" if was_inside else "FREE"
                    print(f"      k{k}: {old_t_pos[k]:.3f} → {new_t_pos[k]:.3f} "
                          f"(Δ={displacement:.3f}, {status})")
        
        # Negative knots
        for name, param in model.named_parameters():
            if param is model.layers[0].t_neg:
                old_t_neg = plugin.params_buffer[name][idx, :].numpy()
                new_t_neg = model.layers[0].t_neg[idx, :].detach().numpy()
                
                print(f"    Negative knots:")
                for k in range(len(old_t_neg)):
                    was_inside = x_min <= old_t_neg[k] <= x_max
                    displacement = abs(new_t_neg[k] - old_t_neg[k])
                    status = "PROTECTED" if was_inside else "FREE"
                    print(f"      k{k}: {old_t_neg[k]:.3f} → {new_t_neg[k]:.3f} "
                          f"(Δ={displacement:.3f}, {status})")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  [1] Initial ReLU-KAN basis functions")
    print("  [2] Learned functions after Task 1")
    print("  [3] Knot positions with Task 1 protected regions")
    print("  [4] Penalty components during Task 2 training")
    print("  [5] Knot positions after Task 2 (both tasks)")
    print("  [6] Output range evolution across tasks")
    print("  [7] Final learned functions after Task 2")
    
    print("\nKey Observations:")
    print("  • Variance loss encourages compact representations")
    print("  • Knots inside Task 1 range are penalized for movement (protected)")
    print("  • Knots outside Task 1 range can move freely (adapt to Task 2)")
    print("  • Boundary consistency maintains function values at range boundaries")
    print("  • Output alignment keeps output distributions stable across tasks")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_kan_interval_penalization_complete_visualization()
