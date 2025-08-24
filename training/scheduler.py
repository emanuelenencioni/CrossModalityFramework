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


def normalize_scheduler_name(name):
    """
    Normalize scheduler name by removing underscores, dashes, and converting to lowercase.
    
    Args:
        name (str): The scheduler name to normalize.
        
    Returns:
        str: Normalized scheduler name.
    """
    return name.lower().replace('_', '').replace('-', '')


def scheduler_builder(optimizer, config):
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
    normalized_name = normalize_scheduler_name(scheduler_name)
    
    # Create mapping of normalized names to actual scheduler classes
    scheduler_mapping = {
        'lambdalr': LambdaLR,
        'multiplicativelr': MultiplicativeLR,
        'steplr': StepLR,
        'multisteplr': MultiStepLR,
        'exponentiallr': ExponentialLR,
        'cosineannealinglr': CosineAnnealingLR,
        'reducelronplateau': ReduceLROnPlateau,
        'cycliclr': CyclicLR,
        'cosineannealingwarmrestarts': CosineAnnealingWarmRestarts
    }
    
    if normalized_name not in scheduler_mapping:
        # Provide helpful error message with suggestions
        available_names = list(scheduler_mapping.keys())
        raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. "
                        f"Available schedulers: {available_names}")
    
    scheduler_class = scheduler_mapping[normalized_name]
    
    # Handle special cases that require lr_lambda parameter
    if normalized_name in ['lambdalr', 'multiplicativelr']:
        lr_lambda = config.pop('lr_lambda')
        scheduler = scheduler_class(optimizer, lr_lambda, **config)
    else:
        scheduler = scheduler_class(optimizer, **config)

    return scheduler