import yaml
import importlib


def build_model_from_cfg(cfg):
    """Builds a model based on the provided configuration dictionary.
    Args:
        cfg (dict): Configuration dictionary containing model parameters.
    Returns:
        model (nn.Module): The initialized model.
    """
    assert 'name' in cfg, "Error - model name must be specified in the config"
    assert 'num_classes' in cfg['head'], "Error - num_classes must be specified in the head config"

    if ('pretrained_weights' not in cfg or cfg.get('pretrained_weights', None) is None)  and cfg.get('pretrained', True):
        print("Warning - pretrained weights not specified, using default backbone pretrained weights")

    # Load the model dynamically based on the name
    module_name = f"model.{cfg['name']}"
    model_class_name = cfg['name'].capitalize()
    module = importlib.import_module(module_name)
    model_class = getattr(module, model_class_name)
    
    # Initialize the model with the provided configuration
    model = model_class(**cfg)
    return model