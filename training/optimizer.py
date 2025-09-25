from torch import optim as op


def build_from_config(model, criterion, cfg):
    """
    Factory method to create and return an optimizer based on the provided configuration.

    Args:
        params (iterable):  Iterable of parameters to optimize or dicts defining parameter groups.
        opti_cfg (dict): A dictionary containing the optimizer configuration
    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.
    Raises:
        ValueError: If the specified optimizer 'name' in the configuration is not supported.
    """
    assert 'optimizer' in cfg.keys(), "'optimizer' params list missing in yaml file"
    opti_cfg = cfg['optimizer']
    if criterion is not None:
        params = list(model.parameters()) + list(criterion.parameters())
        print(criterion.parameters())
    else:
        params = model.parameters()

    opti = opti_cfg['name'].lower()  # Convert to lowercase for case-insensitive matching
    if opti == 'adam':
        return op.Adam(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'])
    elif opti == 'adamw':
        return op.AdamW(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'])
    elif opti == 'sgd':
        return op.SGD(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'], momentum=opti_cfg.get('momentum', 0.0))  # Add momentum with a default value
    elif opti == 'rmsprop':
        return op.RMSprop(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'], momentum=opti_cfg.get('momentum', 0.0)) # Add momentum with a default value
    elif opti == 'adagrad':
        return op.Adagrad(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'])
    elif opti == 'adadelta':
        return op.Adadelta(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'], rho=opti_cfg.get('rho', 0.9)) #Add rho with a default value
    elif opti == 'adamax':
        return op.Adamax(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'])
    elif opti == 'asgd':
        return op.ASGD(params, lr=opti_cfg['lr'], weight_decay=opti_cfg['wd'], lambd=opti_cfg.get('lambd', 1e-4)) # Add lambd with a default value
    elif opti == 'lbfgs':
        return op.LBFGS(params, lr=opti_cfg['lr'])
    elif opti == 'rprop':
        return op.Rprop(params, lr=opti_cfg['lr'])
    elif opti == 'sparseadam':
        return op.SparseAdam(params, lr=opti_cfg['lr'])
    else: raise ValueError(f"Unsupported Optimizer: {opti}")