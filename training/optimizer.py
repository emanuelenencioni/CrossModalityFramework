from torch import optim as op


def build_from_config(params, cfg):
    """
    Factory method to create and return an optimizer based on the provided configuration.

    Args:
        params (iterable):  Iterable of parameters to optimize or dicts defining parameter groups.
        cfg (dict): A dictionary containing the optimizer configuration
    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.
    Raises:
        ValueError: If the specified optimizer 'name' in the configuration is not supported.
    """
    opti = cfg['name'].lower()  # Convert to lowercase for case-insensitive matching
    if opti == 'adam':
        return op.Adam(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif opti == 'adamw':
        return op.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif opti == 'sgd':
        return op.SGD(params, lr=cfg['lr'], weight_decay=cfg['wd'], momentum=cfg.get('momentum', 0.0))  # Add momentum with a default value
    elif opti == 'rmsprop':
        return op.RMSprop(params, lr=cfg['lr'], weight_decay=cfg['wd'], momentum=cfg.get('momentum', 0.0)) # Add momentum with a default value
    elif opti == 'adagrad':
        return op.Adagrad(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif opti == 'adadelta':
        return op.Adadelta(params, lr=cfg['lr'], weight_decay=cfg['wd'], rho=cfg.get('rho', 0.9)) #Add rho with a default value
    elif opti == 'adamax':
        return op.Adamax(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif opti == 'asgd':
        return op.ASGD(params, lr=cfg['lr'], weight_decay=cfg['wd'], lambd=cfg.get('lambd', 1e-4)) # Add lambd with a default value
    elif opti == 'lbfgs':
        return op.LBFGS(params, lr=cfg['lr'])
    elif opti == 'rprop':
        return op.Rprop(params, lr=cfg['lr'])
    elif opti == 'sparseadam':
        return op.SparseAdam(params, lr=cfg['lr'])
    else: raise ValueError(f"Unsupported scheduler: {opti}")