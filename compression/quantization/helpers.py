import torch


def ste_floor(x: torch.Tensor, gradient_factor=1.0) -> torch.Tensor:
    """
    Return the floor values of a tensor.
    """
    return (torch.floor(x) - x * gradient_factor).detach() + x * gradient_factor


def ste_clip(x: torch.Tensor, min_val=-1.0, max_val=1.0) -> torch.Tensor:
    """
    Clip a variable between fixed values such that min_val<=output<=max_val
    Args:
        x: input variable
        min_val: minimum value for clipping
        max_val: maximum value for clipping
    Returns:
        clipped variable
    """
    return (torch.clip(x, min=min_val, max=max_val) - x).detach() + x
