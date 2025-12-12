from abc import ABCMeta

from torch import nn

class CLModuleABC(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for modules that record activations of specified layers during forward passes.

    Attributes:
        activations (list): A list to store activations recorded from specified layers.
        head (nn.Module): The head module of the neural network.
    """

    def __init__(self, head: nn.Module, *args, **kwargs) -> None:
        """
        Initializes the activation recording module.

        Args:
            head (nn.Module): The neural network module whose activations are to be recorded.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self.activations = None
        self.head = head