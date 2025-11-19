import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from omegaconf import DictConfig

from torch.utils.data import DataLoader
from torch import nn

from model import IncrementalClassifier
from src.method.composer import Composer
from src.util.fabric import setup_fabric
from src.experiment import train, test

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_scenarios(config: DictConfig):
    """
    Build training and testing scenarios from the Hydra config.

    Args:
        config (DictConfig): Experiment configuration.

    Returns:
        Tuple containing:
            - train_scenario: Scenario object for training.
            - test_scenario: Scenario object for testing.
    """
    dataset_partial = instantiate(config.dataset)
    train_dataset = dataset_partial(train=True)
    test_dataset = dataset_partial(train=False)
    scenario_partial = instantiate(config.scenario)
    train_scenario = scenario_partial(train_dataset)
    test_scenario = scenario_partial(test_dataset)
    return train_scenario, test_scenario


def activation_visualization(config: DictConfig) -> None:
    """
    Visualize activation changes across continual learning tasks.

    This function trains the model while recording activation values on
    representative samples, generates visualizations, and saves data
    for later regeneration of plots.

    Args:
        config (DictConfig): Experiment configuration.
    """
    if config.exp.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    calc_bwt = getattr(config.exp, "calc_bwt", False)
    calc_fwt = getattr(config.exp, "calc_fwt", False)
    acc_table = getattr(config.exp, "acc_table", False)
    stop_task = getattr(config.exp, "stop_after_task", None)
    save_model = hasattr(config.exp, "model_path")

    if save_model:
        model_path = config.exp.model_path

    log.info("Initializing scenarios")
    train_scenario, test_scenario = get_scenarios(config)

    log.info("Launching Fabric")
    fabric = setup_fabric(config)

    log.info("Building model")
    model = fabric.setup(instantiate(config.model))

    log.info("Setting up method")
    method = instantiate(config.method)(model)

    gen_cm = config.exp.gen_cm
    log_per_batch = config.exp.log_per_batch

    log.info("Setting up dataloaders")
    train_tasks, test_tasks = [], []

    for train_task, test_task in zip(train_scenario, test_scenario):
        train_tasks.append(
            fabric.setup_dataloaders(
                DataLoader(
                    train_task,
                    batch_size=config.exp.batch_size,
                    shuffle=True,
                    generator=torch.Generator(device=fabric.device),
                )
            )
        )
        test_tasks.append(
            fabric.setup_dataloaders(
                DataLoader(
                    test_task,
                    batch_size=1,
                    shuffle=False,
                    generator=torch.Generator(device=fabric.device),
                )
            )
        )

    N = len(train_scenario)
    R = np.zeros((N, N))
    if calc_fwt:
        b = np.zeros(N)

    log.info("Selecting representative images from each task with the lowest label")
    selected_images, selected_labels = [], []

    for task_id, test_task in enumerate(test_tasks):
        min_label = float("inf")
        selected_image = None

        for X, y, _ in test_task:
            label = y.item()
            if label < min_label:
                min_label = label
                selected_image = X

        if selected_image is not None:
            selected_images.append(selected_image)
            selected_labels.append(min_label)
        else:
            log.warning(f"No samples found for task {task_id + 1}")

    log.info(f"Selected {len(selected_images)} images (one per task)")

    activation_history, x_points, task_end_snapshots = [], [], []
    num_samples_per_task = 20

    for task_id, (train_task, test_task) in enumerate(zip(train_tasks, test_tasks)):
        log.info(f"Task {task_id + 1}/{N}")

        if (
            hasattr(method.module, "head")
            and isinstance(method.module.head, IncrementalClassifier)
            and not config.exp.dil
        ):
            log.info("Incrementing model head")
            method.module.head.increment(train_task.dataset.get_classes())

        log.info("Setting up task")
        method.setup_task(task_id)
        step = max(1, config.exp.epochs // num_samples_per_task)

        with fabric.init_tensor():
            for epoch in range(config.exp.epochs):
                lastepoch = epoch == config.exp.epochs - 1
                log.info(f"Epoch {epoch + 1}/{config.exp.epochs}")

                train(method, train_task, task_id, log_per_batch)
                acc = test(method, test_task, task_id, gen_cm, log_per_batch)

                if calc_fwt:
                    method_tmp = Composer(
                        deepcopy(method.module),
                        config.method.criterion,
                        method.lr,
                        method.criterion_scale,
                        method.reg_type,
                        method.gamma,
                        method.clipgrad,
                        method.retaingraph,
                        method.log_reg,
                    )
                    log.info("FWT training pass")
                    method_tmp.setup_task(task_id)
                    train(method_tmp, train_task, task_id, log_per_batch, quiet=True)
                    b[task_id] = test(
                        method_tmp, test_task, task_id, gen_cm, log_per_batch, quiet=True
                    )

                if lastepoch:
                    R[task_id, task_id] = acc

                if task_id > 0:
                    for j in range(task_id - 1, -1, -1):
                        acc = test(
                            method,
                            test_tasks[j],
                            j,
                            gen_cm,
                            log_per_batch,
                            cm_suffix=f" after {task_id}",
                        )
                        if lastepoch:
                            R[task_id, j] = acc

                wandb.log({f"avg_acc": R[task_id, : task_id + 1].mean()})

                if (epoch + 1) % step == 0 or lastepoch:
                    log.info(f"Recording activations after task {task_id} epoch {epoch}")
                    method.module.eval()
                    avg_activations_list = []

                    for img_id in range(N):
                        img = selected_images[img_id]
                        avg_activations = get_layer_activations(method.module, img)
                        avg_activations_list.append(avg_activations)

                    log.info(
                        f" Task {task_id}, Epoch {epoch}, Image {img_id}: "
                        f"{len(avg_activations)} layers recorded"
                    )

                    activation_history.append(avg_activations_list)
                    current_x = task_id + (epoch + 1) / config.exp.epochs
                    x_points.append(current_x)

            task_end_snapshots.append(len(activation_history) - 1)

        if stop_task is not None and task_id == stop_task:
            break

        if calc_bwt:
            wandb.log({"bwt": (R[task_id, :task_id] - R.diagonal()[:-1]).mean()})

        if calc_fwt:
            fwt = [R[i - 1, i] - b[i] for i in range(1, task_id + 1)]
            wandb.log({"fwt": np.array(fwt).mean()})

    if save_model:
        log.info("Saving model")
        torch.save(model.state_dict(), model_path)

    if acc_table:
        log.info("Logging accuracy table")
        wandb.log(
            {
                "acc_table": wandb.Table(
                    data=R.tolist(), columns=[f"task_{i}" for i in range(N)]
                )
            }
        )

    log.info("Saving essential data for plot regeneration...")
    output_dir = Path(config.exp.log_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "activation_history.pkl", "wb") as f:
        pickle.dump(activation_history, f)
    with open(output_dir / "x_points.pkl", "wb") as f:
        pickle.dump(x_points, f)
    with open(output_dir / "task_end_snapshots.pkl", "wb") as f:
        pickle.dump(task_end_snapshots, f)

    torch.save(selected_images, output_dir / "selected_images.pt")
    with open(output_dir / "selected_labels.pkl", "wb") as f:
        pickle.dump(selected_labels, f)

    generate_plots(
        activation_history,
        x_points,
        task_end_snapshots,
        selected_images,
        selected_labels,
        output_dir,
    )
    exit(0)


def get_layer_activations(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Compute average activations for each layer.

    Args:
        model (Any): Model instance.
        x (Tensor): Input tensor for which to extract activations.

    Returns:
        List[Tensor]: List of layer activations.
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


def compute_differences(
    activation_history: List[List[torch.Tensor]],
    task_end_snapshots: List[int],
    x_points: List[float],
    signed: bool = False
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute differences between current activations and baseline per task.

    Args:
        activation_history (List[List[Tensor]]): Recorded activations per task.
        task_end_snapshots (List[int]): Index of end-of-task activations.
        x_points (List[float]): Corresponding x-axis points for plotting.
        signed (bool, optional): Whether to keep signed differences. Defaults to False.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Differences and x-values per sample.
    """
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
    fig: plt.Figure,
    ax: plt.Axes,
    differences: List[List[float]],
    x_lists: List[List[float]],
    colors: np.ndarray,
    selected_images: List[torch.Tensor],
    selected_labels: List[int],
    N: int,
    label_prefix: str = '',
    linestyle: str = '-',
    add_images: bool = True,
    y_label: str = 'Average Activation Difference'
) -> None:
    """
    Plot activation drift over tasks for multiple samples.

    Args:
        fig (plt.Figure): Matplotlib figure object.
        ax (plt.Axes): Matplotlib axes object.
        differences (List[List[float]]): Differences per task per sample.
        x_lists (List[List[float]]): X-axis points for each sample.
        colors (np.ndarray): Colors for each sample/task.
        selected_images (List[Tensor]): Representative images for each task.
        selected_labels (List[int]): Labels corresponding to selected images.
        N (int): Number of tasks/samples.
        label_prefix (str, optional): Prefix for legend labels. Defaults to ''.
        linestyle (str, optional): Line style for the plot. Defaults to '-'.
        add_images (bool, optional): Whether to add images at task markers. Defaults to True.
        y_label (str, optional): Y-axis label. Defaults to 'Average Activation Difference'.
    """
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
    axes: List[plt.Axes],
    activation_history: List[List[List[torch.Tensor]]],
    task_end_snapshots: List[int],
    x_lists: List[List[float]],
    colors: np.ndarray,
    selected_images: List[torch.Tensor],
    selected_labels: List[int],
    N: int,
    num_layers: int,
    label_prefix: str = '',
    linestyle: str = '-',
    add_images: bool = True
) -> None:
    """
    Plot per-layer average absolute activation values over tasks.

    Args:
        axes (List[plt.Axes]): List of axes objects for each layer.
        activation_history (List[List[List[Tensor]]]): Activation values recorded per sample per layer.
        task_end_snapshots (List[int]): Indices marking end-of-task activations.
        x_lists (List[List[float]]): X-axis values per sample.
        colors (np.ndarray): Colors for each sample.
        selected_images (List[Tensor]): Representative images for each task.
        selected_labels (List[int]): Labels corresponding to selected images.
        N (int): Number of tasks/samples.
        num_layers (int): Number of layers in the model.
        label_prefix (str, optional): Legend prefix. Defaults to ''.
        linestyle (str, optional): Line style for the plot. Defaults to '-'.
        add_images (bool, optional): Whether to overlay sample images. Defaults to True.
    """
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


def load_data(visualizations_dir: Path) -> Dict:
    """
    Load activation visualization data from disk.

    Args:
        visualizations_dir (Path): Directory containing saved visualization files.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - activation_history (List[List[List[Tensor]]])
            - x_points (List[float])
            - task_end_snapshots (List[int])
            - selected_images (List[Tensor])
            - selected_labels (List[int])
    """
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


def load_and_plot_visualizations(visualizations_dir: str) -> None:
    """
    Load previously saved activation visualization data and regenerate plots.

    Args:
        visualizations_dir (str): Directory containing the saved visualizations.
    """
    output_dir = Path(visualizations_dir)
    data = load_data(output_dir)

    generate_plots(
        data['activation_history'], data['x_points'],
        data['task_end_snapshots'], data['selected_images'],
        data['selected_labels'], output_dir
    )


def compare_and_plot_visualizations(
    folder1: str,
    folder2: str,
    output_folder: str,
    method_name: str = 'InTAct',
    baseline_name: str = 'LwF'
) -> None:
    """
    Compare activation drift between two methods and generate plots.

    Args:
        folder1 (str): Directory of first method's visualization data.
        folder2 (str): Directory of second method's visualization data.
        output_folder (str): Directory to save comparison plots.
        method_name (str, optional): Name of first method. Defaults to 'InTAct'.
        baseline_name (str, optional): Name of baseline method. Defaults to 'LwF'.
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