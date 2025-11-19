import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict


def distillation_loss(outputs_new: torch.Tensor, outputs_old: torch.Tensor, T: float=2) -> torch.Tensor:
    """
    Computes the distillation loss between the new model outputs and the old model outputs.
    Distillation loss is used to transfer knowledge from a teacher model (old model) to a student model (new model).
    It measures the difference between the softened output probabilities of the two models.

    Args:
        outputs_new (torch.Tensor): The output logits from the new (student) model.
        outputs_old (torch.Tensor): The output logits from the old (teacher) model.

        T (float, optional): The temperature parameter to soften the probabilities. Default is 2.
    Returns:
        torch.Tensor: The computed distillation loss.
    """

    size = outputs_old.size(dim=1)
    prob_new = F.softmax(outputs_new[:,:size]/T,dim=1)
    prob_old = F.softmax(outputs_old/T,dim=1)
    return prob_old.mul(-1*torch.log(prob_new)).sum(1).mean()*T*T


def param_change_loss(model: nn.Module, multiplier: Dict, params_buffer: Dict) -> torch.Tensor:
    """
    Computes the parameter change loss for a given model.
    This function calculates the loss based on the difference between the current 
    parameters of the model and a buffer of previous parameters, weighted by a 
    multiplier. The loss is computed only for parameters that require gradients.

    Args:
        model (torch.nn.Module): The neural network model containing the parameters.
        multiplier (dict): A dictionary where keys are parameter names and values 
                           are the corresponding multipliers for the loss calculation.
        params_buffer (dict): A dictionary where keys are parameter names and values 
                              are the previous parameter values to compare against.z                              
    Returns:
        torch.Tensor: The computed parameter change loss.
    """

    loss = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            loss += (multiplier[name] * (p - params_buffer[name]) ** 2).sum()
    return loss