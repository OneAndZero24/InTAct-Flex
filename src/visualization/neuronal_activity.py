import logging
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import wandb
from tqdm import tqdm

# from model import IncrementalClassifier
# from src.method.composer import Composer
# from src.method.method_plugin_abc import MethodPluginABC
# from src.util.fabric import setup_fabric


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_layer_activations(model, x):
    """
    Get average of absolute values of activations for each layer.
    """
    model.eval()
    activations = []

    with torch.no_grad():
        current = torch.flatten(x, start_dim=1)
        actual_model = model._forward_module if hasattr(model, '_forward_module') else model

        if hasattr(actual_model, 'layers'):
            for layer in actual_model.layers:
                current = layer(current)
                if type(layer).__name__ == "IntervalActivation" or isinstance(layer, torch.nn.ReLU):
                    activations.append(current)

    return activations


def compute_differences(activation_history, task_end_snapshots, x_points, signed=False):
    N = len(task_end_snapshots)
    differences_lists = []
    x_values_lists = []

    for img_id in range(N):
        baseline_index = task_end_snapshots[img_id]
        baseline_activations = activation_history[baseline_index][img_id]
        differences = []
        x_values_this = []

        for s in range(baseline_index, len(activation_history)):
            current_activations = activation_history[s][img_id]

            if signed:
                total_diff = np.mean([
                    (torch.sum(curr - base) / torch.norm(base)).item()
                    for curr, base in zip(current_activations, baseline_activations)
                ])
            else:
                total_diff = np.mean([
                    (torch.sum(curr - base).abs() / torch.norm(base)).item()
                    for curr, base in zip(current_activations, baseline_activations)
                ])

            differences.append(total_diff)
            x_values_this.append(x_points[s])

        differences_lists.append(differences)
        x_values_lists.append(x_values_this)

    return differences_lists, x_values_lists


def plot_drift(
    fig, ax, differences, x_lists, colors,
    selected_images, selected_labels, N,
    label_prefix='', linestyle='-', add_images=True,
    y_label='Average Activation Difference'
):
    for img_id in range(N - 1):
        x_values = x_lists[img_id]
        y_values = differences[img_id]

        ax.plot(
            x_values, y_values,
            linewidth=2.5,
            color=colors[img_id],
            linestyle=linestyle,
            label=f'{label_prefix}Task {img_id+1}'
        )

        marker_x = [x for x in x_values if abs(x - round(x)) < 1e-6]
        marker_y = [y_values[i] for i, x in enumerate(x_values) if abs(x - round(x)) < 1e-6]

        if marker_x:
            ax.plot(
                marker_x, marker_y,
                marker='o',
                markersize=8,
                color=colors[img_id],
                linestyle='None'
            )

        if add_images:
            x = x_values[0]
            y = y_values[0] + 0.5  # Adjusted to be a little bit higher
            img_np = selected_images[img_id].cpu().squeeze().numpy()

            if img_np.ndim == 3:
                if img_np.shape[0] == 1:
                    img_np = img_np[0]
                elif img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))

            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            imagebox = OffsetImage(img_np, zoom=1.8, cmap='gray' if img_np.ndim == 2 else None)
            ab = AnnotationBbox(
                imagebox, (x, y),
                frameon=True,
                pad=0.3,
                bboxprops=dict(edgecolor=colors[img_id], linewidth=2, facecolor='white')
            )
            ax.add_artist(ab)

    ax.set_xlabel('Task ID', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax.set_xticks(range(1, N+1))
    ax.set_xticklabels([f'{i}' for i in range(1, N+1)])
    ax.tick_params(labelsize=14)
    ax.set_xlim(0.8, N)


def plot_per_layer(
    axes, activation_history, task_end_snapshots, x_lists, colors,
    selected_images, selected_labels, N, num_layers,
    label_prefix='', linestyle='-', add_images=True
):
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]

        for img_id in range(N - 1):
            baseline_index = task_end_snapshots[img_id]
            x_values = x_lists[img_id]

            y_values = [
                torch.mean(torch.abs(activation_history[baseline_index + i][img_id][layer_idx])).item()
                for i in range(len(x_values))
            ]

            ax.plot(
                x_values, y_values,
                linewidth=2.5,
                color=colors[img_id],
                linestyle=linestyle,
                label=f'{label_prefix}Sample from Task {img_id+1}'
            )

            marker_x = [x for x in x_values if abs(x - round(x)) < 1e-6]
            marker_y = [y_values[i] for i, x in enumerate(x_values) if abs(x - round(x)) < 1e-6]

            if marker_x:
                ax.plot(
                    marker_x, marker_y,
                    marker='o',
                    markersize=8,
                    color=colors[img_id],
                    linestyle='None'
                )

            if add_images:
                x = x_values[0]
                y = y_values[0]
                img_np = selected_images[img_id].cpu().squeeze().numpy()

                if img_np.ndim == 3:
                    if img_np.shape[0] == 1:
                        img_np = img_np[0]
                    elif img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))

                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                imagebox = OffsetImage(img_np, zoom=1.2, cmap='gray' if img_np.ndim == 2 else None)
                ab = AnnotationBbox(
                    imagebox, (x, y),
                    frameon=True,
                    pad=0.3,
                    bboxprops=dict(edgecolor=colors[img_id], linewidth=2, facecolor='white')
                )
                ax.add_artist(ab)

        ax.set_title(f'Idx of Activation Layer: {layer_idx+1}', fontsize=14, pad=10)
        ax.set_ylabel('Average Absolute Activation Value', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=14)
        ax.set_xticks(range(1, N+1))
        ax.set_xticklabels([f'{i}' for i in range(1, N+1)])
        ax.tick_params(labelsize=14)
        ax.set_xlim(0.8, N)

def generate_plots(activation_history, x_points, task_end_snapshots, selected_images, selected_labels, output_dir):
    N = len(selected_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    abs_diffs, x_lists = compute_differences(
        activation_history, task_end_snapshots, x_points, signed=False
    )

    log.info('Creating activation drift visualization...')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks', fontsize=16)

    plot_drift(
        fig, ax, abs_diffs, x_lists, colors,
        selected_images, selected_labels, N,
        label_prefix='', linestyle='-', add_images=True,
        y_label='L1 Norm Activation Difference'
    )

    plt.tight_layout()
    output_path = output_dir / 'activation_drift_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f'Saved visualization to {output_path}')
    plt.close()

    # Signed differences
    log.info('Creating signed activation drift visualization...')
    signed_diffs, _ = compute_differences(
        activation_history, task_end_snapshots, x_points, signed=True
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks', fontsize=16)

    plot_drift(
        fig, ax, signed_diffs, x_lists, colors,
        selected_images, selected_labels, N,
        label_prefix='', linestyle='-', add_images=True,
        y_label='Total Signed Activation Difference'
    )

    plt.tight_layout()
    output_path_signed = output_dir / 'activation_signed_drift_visualization.png'
    plt.savefig(output_path_signed, dpi=300, bbox_inches='tight')
    log.info(f'Saved signed visualization to {output_path_signed}')
    plt.close()

    # Detailed per-layer plot
    log.info('Creating detailed per-layer activation plot...')
    num_layers = (
        len(activation_history[0][0])
        if activation_history and activation_history[0] and activation_history[0][0]
        else 0
    )

    if num_layers == 0:
        log.warning('No layer activations recorded. Skipping detailed per-layer plot.')
        log.info(f'Main visualization saved to {output_dir}')
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), sharex=True)
    fig.suptitle('Per-Layer Activation Over Tasks', fontsize=16)

    if num_layers == 1:
        axes = [axes]

    plot_per_layer(
        axes, activation_history, task_end_snapshots, x_lists, colors,
        selected_images, selected_labels, N, num_layers,
        label_prefix='', linestyle='-', add_images=True
    )

    axes[-1].set_xlabel('Task ID', fontsize=14)
    plt.tight_layout()
    output_path_detailed = output_dir / 'activation_per_layer_detailed.png'
    plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
    log.info(f'Saved detailed visualization to {output_path_detailed}')
    plt.close()
    log.info(f'All visualizations saved to {output_dir}')


def load_data(visualizations_dir: Path):
    data = {}

    with open(visualizations_dir / 'activation_history.pkl', 'rb') as f:
        data['activation_history'] = pickle.load(f)

    with open(visualizations_dir / 'x_points.pkl', 'rb') as f:
        data['x_points'] = pickle.load(f)

    with open(visualizations_dir / 'task_end_snapshots.pkl', 'rb') as f:
        data['task_end_snapshots'] = pickle.load(f)

    data['selected_images'] = torch.load(visualizations_dir / 'selected_images.pt')

    with open(visualizations_dir / 'selected_labels.pkl', 'rb') as f:
        data['selected_labels'] = pickle.load(f)

    return data


def load_and_plot_visualizations(visualizations_dir: str):
    """
    Load saved data from the visualizations directory and regenerate the plots.
    """
    output_dir = Path(visualizations_dir)
    data = load_data(output_dir)

    generate_plots(
        data['activation_history'], data['x_points'],
        data['task_end_snapshots'], data['selected_images'],
        data['selected_labels'], output_dir
    )


def compare_and_plot_visualizations(
    folder1: str, folder2: str, output_folder: str,
    method_name: str = 'InTAct', baseline_name: str = 'LwF'
):
    """
    Load data from two folders and generate comparison plots.
    """
    data1 = load_data(Path(folder1))
    data2 = load_data(Path(folder2))

    N = len(data1['selected_labels'])
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    selected_images = data1['selected_images']
    selected_labels = data1['selected_labels']

    abs_diffs1, x_lists1 = compute_differences(
        data1['activation_history'], data1['task_end_snapshots'], data1['x_points'], signed=False
    )
    abs_diffs2, x_lists2 = compute_differences(
        data2['activation_history'], data2['task_end_snapshots'], data2['x_points'], signed=False
    )

    print('Creating activation drift comparison visualization...')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Activation Drift Over Tasks Comparison', fontsize=16)

    plot_drift(
        fig, ax, abs_diffs1, x_lists1, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{method_name} ', linestyle='-', add_images=True,
        y_label='L1 Norm Activation Difference'
    )

    plot_drift(
        fig, ax, abs_diffs2, x_lists2, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{baseline_name} ', linestyle='--', add_images=False,
        y_label='L1 Norm Activation Difference'
    )

    plt.tight_layout()
    plt.xlabel('Task ID', fontsize=14)
    plt.ylabel('L1 Norm Activation Difference', fontsize=14)

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'activation_drift_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved visualization to {output_path}')
    plt.close()

    # Signed comparison
    print('Creating signed activation drift comparison visualization...')
    signed_diffs1, _ = compute_differences(
        data1['activation_history'], data1['task_end_snapshots'], data1['x_points'], signed=True
    )
    signed_diffs2, _ = compute_differences(
        data2['activation_history'], data2['task_end_snapshots'], data2['x_points'], signed=True
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Signed Activation Drift Over Tasks Comparison', fontsize=16)

    plot_drift(
        fig, ax, signed_diffs1, x_lists1, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{method_name} ', linestyle='-', add_images=True,
        y_label='Total Signed Activation Difference'
    )

    plot_drift(
        fig, ax, signed_diffs2, x_lists2, colors,
        selected_images, selected_labels, N,
        label_prefix=f'{baseline_name} ', linestyle='--', add_images=False,
        y_label='Total Signed Activation Difference'
    )

    plt.tight_layout()
    output_path_signed = output_dir / 'activation_signed_drift_visualization.png'
    plt.savefig(output_path_signed, dpi=300, bbox_inches='tight')
    print(f'Saved signed visualization to {output_path_signed}')
    plt.close()

    # Per-layer comparison
    print('Creating detailed per-layer activation comparison plot...')
    num_layers = (
        len(data1['activation_history'][0][0])
        if data1['activation_history'] and data1['activation_history'][0] and data1['activation_history'][0][0]
        else 0
    )

    if num_layers == 0:
        print('No layer activations recorded. Skipping detailed per-layer plot.')
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), sharex=True)
    fig.suptitle('Per-Layer Activation Over Tasks Comparison', fontsize=16)

    if num_layers == 1:
        axes = [axes]

    plot_per_layer(
        axes, data1['activation_history'], data1['task_end_snapshots'],
        x_lists1, colors, selected_images, selected_labels, N, num_layers,
        label_prefix=f'{method_name} ', linestyle='-', add_images=True
    )

    plot_per_layer(
        axes, data2['activation_history'], data2['task_end_snapshots'],
        x_lists2, colors, selected_images, selected_labels, N, num_layers,
        label_prefix=f'{baseline_name} ', linestyle='--', add_images=False
    )

    axes[-1].set_xlabel('Task ID', fontsize=14)
    plt.tight_layout()
    output_path_detailed = output_dir / 'activation_per_layer_detailed.png'
    plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved detailed visualization to {output_path_detailed}')
    print(f'All visualizations regenerated in {output_dir}')

if __name__ == "__main__":
    compare_and_plot_visualizations(
        "/home/patrykkrukowski/Projects/local_cl/local-cl/ablation_study/visualizations/lcl",
        "/home/patrykkrukowski/Projects/local_cl/local-cl/ablation_study/visualizations/lwf",
        "/home/patrykkrukowski/Projects/local_cl/local-cl/ablation_study/visualizations"
    )