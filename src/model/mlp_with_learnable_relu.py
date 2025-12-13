from typing import Union
from omegaconf import ListConfig

import torch
import torch.nn as nn

from model.cl_module_abc import CLModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer.learnable_relu import LearnableReLU
from model.layer.interval_activation import IntervalActivation


class MLPWithLearnableReLU(CLModuleABC):
    """
    Multi-Layer Perceptron (MLP) with LearnableReLU layers for continual learning.

    This MLP consists exclusively of LearnableReLU layers in the backbone.
    Each LearnableReLU layer incrementally expands its representational
    capacity by adding task-specific ReLU basis functions. The final
    classification head is implemented as an IncrementalClassifier.

    Args:
        initial_out_features (int): Number of output features for the initial task.
        sizes (list[int]): List of feature sizes for each layer, including input
            and final hidden layer.
        k (int): Maximum number of learnable ReLU basis functions per layer.
        head_type (str, optional): Type of classifier head. Defaults to "Normal".
        mask_past_classifier_neurons (bool, optional): Whether to mask neurons
            corresponding to previous tasks in the classifier head.
        config (Union[dict, list[dict], ListConfig], optional): Configuration
            dictionary passed to the classifier head.
    """

    def __init__(
        self,
        initial_out_features: int,
        sizes: list[int],
        k: int,
        head_type: str = "Normal",
        mask_past_classifier_neurons: bool = False,
        config: Union[dict, list[dict], ListConfig] = {},
    ) -> None:
        """
        Initialize the LearnableReLU-based MLP.

        Args:
            initial_out_features (int): Number of output features for the initial task.
            sizes (list[int]): Layer sizes, including input and final hidden layer.
            k (int): Maximum number of ReLU basis functions per layer.
            head_type (str, optional): Type of classifier head.
            mask_past_classifier_neurons (bool, optional): Whether to mask past
                classifier neurons.
            config (Union[dict, list[dict], ListConfig], optional): Configuration
                for the classifier head.
        """

        super().__init__(
            IncrementalClassifier(
                in_features=sizes[-1],
                out_features=initial_out_features,
                head_type=head_type,
                mask_past_classifier_neurons=mask_past_classifier_neurons,
                **config,
            )
        )

        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.extend(
                [
                    LearnableReLU(
                    in_features=in_size,
                    out_features=out_size,
                    k=k),
                    IntervalActivation(use_nonlinear_transform=False)
                ]
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the classifier head.
        """

        x = torch.flatten(x, start_dim=1)

        for layer in self.layers:
            x = layer(x)

        return self.head(x)
