import logging
from typing import Optional, Tuple
from copy import deepcopy

from torch import optim, Tensor
import torch.nn as nn

import wandb

from model.cl_module_abc import CLModuleABC
from method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Composer:
    """
    Composer class for managing the training process of a module with optional plugins and regularization.

    Attributes:
        module (CLModuleABC): The module to be trained.
        optimizer (Optional[optim.Optimizer]): The optimizer for training.
        lr (float): The learning rate for subsequent tasks.
        reg_type (Optional[str]): The type of regularization to apply (e.g., L1, L2).
        task_heads (bool): Whether to use task-specific heads for multi-task learning.
        clipgrad (Optional[float]): The gradient clipping value.
        retaingraph (Optional[bool]): Whether to retain the computation graph during backpropagation.
        log_reg (Optional[bool]): Whether to log the regularization loss during training.
        plugins (Optional[list[MethodPluginABC]]): List of plugins to be used during training.
        heads (list[nn.Module]): List of task-specific heads, initialized if `task_heads` is True.
    """

    def __init__(self, 
        module: CLModuleABC,
        lr: float,
        task_heads: bool=False,
        clipgrad: Optional[float]=None,
        retaingraph: Optional[bool]=False,
        log_reg: Optional[bool]=False,
        plugins: Optional[list[MethodPluginABC]]=[]
    ) -> None:
        """
        Initialize the Composer class.

        Args:
            module (CLModuleABC): The continual learning module to be trained.
            lr (float): The learning rate for subsequent tasks.
            task_heads (bool, optional): Whether to use task-specific heads for multi-task learning. Defaults to False.
            clipgrad (Optional[float], optional): Maximum gradient norm for gradient clipping. Defaults to None.
            retaingraph (Optional[bool], optional): Whether to retain the computation graph during backpropagation. Defaults to False.
            log_reg (Optional[bool], optional): Whether to log the regularization loss during training. Defaults to False.
            plugins (Optional[list[MethodPluginABC]], optional): List of method plugins to extend the training process. Defaults to an empty list.
        """

        self.module = module

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.lr = lr
        self.task_heads = task_heads
        self.clipgrad = clipgrad
        self.retaingraph = retaingraph
        self.plugins = plugins
        self.log_reg = log_reg
        
        if self.task_heads:
            self.heads = []

        for plugin in self.plugins:
            plugin.set_module(self.module)
            log.info(f'Plugin {plugin.__class__.__name__} added to composer')


    def _setup_optim(self) -> None:
        """
        Sets up the optimizer for the model.
        This method initializes the optimizer with the model parameters that require
        gradients. It uses the Adam optimizer.
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    def setup_task(self, task_id: int) -> None:
        """
        Set up the task with the given task ID.
        This method initializes the optimizer for the specified task and
        calls the setup_task method on each plugin associated with this instance.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """

        if self.task_heads:
            if task_id >= len(self.heads):
                tmp_head = self.module.head
                if task_id > 0:
                   tmp_head = deepcopy(self.module.head)
                self.heads.append(tmp_head)
            self.module.head = self.heads[task_id]

        self._setup_optim()
        for plugin in self.plugins:
            plugin.setup_task(task_id)


    def forward(self, x: Tensor, y: Tensor, task_id: int ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the model and apply plugins.

        Args:
            x (torch.Tensor): Input tensor to the model.
            y (torch.Tensor): Target tensor for computing the loss.
            task_id (int): The ID of the current task.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss after applying regularization and plugins.
                - preds (torch.Tensor): The model predictions after applying plugins.
        """

        preds = self.module(x)
        loss = self.criterion(preds, y)
            
        loss_ct = loss

        for plugin in self.plugins:
            loss, preds = plugin.forward(x, y, loss, preds)

        if self.log_reg:
            wandb.log({f'Loss/train/{task_id}/reg': loss-loss_ct.mean()})
        return loss, preds


    def backward(self, loss: Tensor) -> None:  
        """
        Performs a backward pass and updates the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
            
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Optionally clips the gradients to prevent exploding gradients.
        4. Updates the model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        loss.backward(retain_graph=self.retaingraph)
        if self.clipgrad is not None:
            nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()