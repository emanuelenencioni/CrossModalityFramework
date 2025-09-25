import yaml
import importlib

def check_backbone_params(cfg):
    """
    Check if the backbone parameters are correctly specified in the configuration file.
    :param cfg: The configuration dictionary.
    :return: True if both event and rgb backbones are specified, False otherwise.  Also returns the specified backbone
    """
    assert 'model' in cfg.keys(), "Error - specify the model architecture"
    assert 'backbone' in cfg['model'].keys(), "Error - specify the backbone parameters"
    cfg_b = cfg['model']['backbone']
    assert 'embed_dim' in cfg_b.keys(), "Error - specify the embed_dim"
    assert 'input_size' in cfg_b.keys(), "Error - specify the input_size"
    ev_bb = cfg_b['event_backbone'] if 'event_backbone' in cfg_b.keys() else None
    rgb_bb = cfg_b['rgb_backbone'] if 'rgb_backbone' in cfg_b.keys() else None

    
    assert ev_bb != None or rgb_bb != None, "Error - specify at least one backbone: event or rgb"
    if ev_bb is None:
        return False, rgb_bb
    elif rgb_bb is None:
        return False, ev_bb

    return True, None

def build_model_from_cfg(cfg):
    """Builds a model based on the provided configuration dictionary.
    Args:
        cfg (dict): Configuration dictionary containing model parameters.
    Returns:
        model (nn.Module): The initialized model.
    """



    dual_modality, backbone = check_backbone_params(cfg)
    cfg['dual_modality'] = dual_modality
    model_cfg = cfg['model']
    pretrained_weights = model_cfg['backbone']['pretrained_weights'] if 'pretrained_weights' in model_cfg['backbone'].keys() else None
    pretrained = model_cfg['backbone'].get('pretrained', True) if 'pretrained' in model_cfg['backbone'].keys() else True

    assert 'name' in model_cfg, "Error - model name must be specified in the config"
    assert 'num_classes' in model_cfg['head'], "Error - num_classes must be specified in the head config"

    if ('pretrained_weights' not in model_cfg or model_cfg.get('pretrained_weights', None) is None)  and model_cfg.get('pretrained', True):
        print("Warning - pretrained weights not specified, using default backbone pretrained weights")

    # Load the model dynamically based on the name
    module_name = f"model.{ model_cfg['name']}"
    model_class_name = model_cfg['name'].capitalize()
    module = importlib.import_module(module_name)
    model_class = getattr(module, model_class_name)
    
    # Initialize the model with the provided configuration
    model = model_class(**model_cfg)
    return model