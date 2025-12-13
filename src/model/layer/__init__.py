from enum import Enum
from functools import partial

from torch import nn

from .interval_activation import IntervalActivation
from .learnable_relu import LearnableReLU

class LayerType(Enum):
    """
    enum = (NORMAL, INTERVAL, RELU_KAN)
    """

    NORMAL = "Normal"
    INTERVAL = "Interval"
    LEARNABLE_RELU = "LearnableReLU"


def _instantiate(
    map: dict,
    layer_type: str=LayerType.NORMAL,
    *args, 
    **kwargs
):
    layer = map[layer_type]
    if layer_type  == LayerType.NORMAL:
        return layer(*args)
    return layer(*args, **kwargs)


instantiate = partial(_instantiate, {
    LayerType.NORMAL: nn.Linear,
    LayerType.INTERVAL: IntervalActivation,
    LayerType.LEARNABLE_RELU: LearnableReLU
})


instantiate2D = partial(_instantiate, {
    LayerType.NORMAL: nn.Conv2d
})