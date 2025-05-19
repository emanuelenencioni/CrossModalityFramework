import torch
from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    CosineAnnealingWarmRestarts
)


def scheduler_factory(optimizer, config):
    """
    Factory method for creating PyTorch schedulers from a configuration dictionary.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be scheduled.
        config (dict): A dictionary containing the scheduler configuration.
                         Must contain a 'name' key specifying the scheduler type.
                         Other keys are scheduler-specific arguments.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: A PyTorch learning rate scheduler.

    Raises:
        ValueError: If the specified scheduler name is not supported.
    """

    scheduler_name = config.pop('name')

    if scheduler_name == 'LambdaLR':
        lr_lambda = config.pop('lr_lambda')
        scheduler = LambdaLR(optimizer, lr_lambda, **config)
    elif scheduler_name == 'MultiplicativeLR':
        lr_lambda = config.pop('lr_lambda')
        scheduler = MultiplicativeLR(optimizer, lr_lambda, **config)
    elif scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, **config)
    elif scheduler_name == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **config)
    elif scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, **config)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **config)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **config)
    elif scheduler_name == 'CyclicLR':
        scheduler = CyclicLR(optimizer, **config)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, **config)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler