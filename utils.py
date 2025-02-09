import sys
import random
from copy import deepcopy
from datetime import datetime
import math
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


def set_seed(seed: int = 1994):
    """Sets random seed for reproducibility across NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def l4_regularization(model: nn.Module, lambda_l4: float) -> torch.Tensor:
    """
    Computes the L⁴ norm regularization penalty, normalized by the number of trainable parameters.

    Args:
        model (torch.nn.Module): The neural network model.
        lambda_l4 (float): Regularization strength.

    Returns:
        torch.Tensor: The L⁴ norm penalty term.
    """
    l4_penalty = sum(torch.sum(p.pow(4)) for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_params == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return (lambda_l4 / num_params) * l4_penalty



def compute_gradient_norm(network: nn.Module) -> float:
    """
    Computes the L2 norm of gradients for a given neural network.

    Args:
        network (torch.nn.Module): The neural network model.

    Returns:
        float: The computed L2 norm of gradients.
    """
    grad_norm = sum(p.grad.norm(2) ** 2 for p in network.parameters() if p.grad is not None)
    return grad_norm.sqrt().item()



def compute_param_change(initial_params: list[dict], networks: list[nn.Module], aggregation: str = "mean") -> float:
    """
    Computes a summary of parameter changes across networks.

    Args:
        initial_params (list): List of initial state_dicts for each network.
        networks (list): List of neural networks.
        aggregation (str): Aggregation method ('sum', 'mean', 'max').

    Returns:
        float: Aggregated parameter change across all networks.
    """
    param_changes = [
        sum(torch.norm(network.state_dict()[k] - init[k]) for k in init.keys())
        for init, network in zip(initial_params, networks)
    ]

    return {
        "sum": sum(param_changes).item(),
        "mean": (sum(param_changes) / len(param_changes)).item(),
        "max": max(param_changes).item(),
    }.get(aggregation, ValueError("Invalid aggregation method. Choose 'sum', 'mean', or 'max'."))



def sample_indices(
    memory_length: int, 
    batch_size: int, 
    sampling_type: str, 
    alpha: float = 1.0, 
    exp_start: float = 0.005
) -> torch.Tensor:
    """
    Samples indices from a hybrid exponential-uniform distribution.

    Args:
        memory_length (int): The total number of available indices.
        batch_size (int): The number of indices to sample.
        sampling_type (str): 'recent' for favoring recent indices, 'preceding' for older indices.
        alpha (float, optional): Weight factor between exponential and uniform distributions (0 = pure uniform, 1 = pure exponential). Default is 1.0.
        exp_start (float, optional): Lower bound for the exponential distribution. Default is 0.005.

    Returns:
        torch.Tensor: A tensor of sampled indices.
    """
    if sampling_type not in ["recent", "preceding"]:
        if alpha != 0.0:
            raise ValueError("sampling_type must be either 'recent' or 'preceding', or alpha = 0.")

    # Create an exponential distribution with customizable start
    exponential_distribution = torch.logspace(math.log10(exp_start), math.log10(1.0), steps=memory_length)

    # Reverse if sampling type is 'preceding'
    if sampling_type == "preceding":
        exponential_distribution = exponential_distribution.flip(dims=(0,))

    # Normalize the exponential distribution
    exponential_distribution /= exponential_distribution.sum()

    # Create a uniform distribution
    uniform_distribution = torch.full((memory_length,), 1.0 / memory_length)

    # Combine exponential and uniform distributions
    combined_distribution = alpha * exponential_distribution + (1 - alpha) * uniform_distribution
    combined_distribution /= combined_distribution.sum()

    # Sample indices based on the combined distribution
    sampled_indices = torch.multinomial(combined_distribution, batch_size, replacement=True)

    return sampled_indices

