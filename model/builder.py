import yaml
import importlib
from loguru import logger

def check_backbone_params(cfg):
    """
    Check if the backbone parameters are correctly specified in the configuration file.
    :param cfg: The configuration dictionary.
    :return: True if both event and rgb backbones are specified, False otherwise.  Also returns the specified backbone
    """
    assert 'model' in cfg.keys() or ('model1' in cfg.keys() and 'model2' in cfg.keys()), "Error - specify the model architecture"
    if 'model' in cfg.keys():
        assert 'backbone' in cfg['model'].keys(), "Error - specify the backbone parameters"
        cfg_b = cfg['model']['backbone']
        assert 'embed_dim' in cfg_b.keys(), "Error - specify the embed_dim"
        assert 'input_size' in cfg_b.keys(), "Error - specify the input_size"
        return False
    elif 'model1' in cfg.keys() and 'model2' in cfg.keys():
        assert 'backbone' in cfg['model1'].keys(), "Error - specify the backbone parameters for model1"
        assert 'backbone' in cfg['model2'].keys(), "Error - specify the backbone parameters for model2"
        cfg_b1 = cfg['model1']['backbone']
        cfg_b2 = cfg['model2']['backbone']
        assert 'embed_dim' in cfg_b1.keys(), "Error - specify the embed_dim for model1"
        assert 'input_size' in cfg_b1.keys(), "Error - specify the input_size for model1"
        assert 'embed_dim' in cfg_b2.keys(), "Error - specify the embed_dim for model2"
        assert 'input_size' in cfg_b2.keys(), "Error - specify the input_size for model2"
        return True
    else:
        assert False, "Error - specify the model architecture"



def build_model_from_cfg(cfg):
    """Builds a model based on the provided configuration dictionary.
    Args:
        cfg (dict): Configuration dictionary containing model parameters.
    Returns:
        model (nn.Module): The initialized model.
    """



    dual_modality =  check_backbone_params(cfg)
    cfg['dual_modality'] = dual_modality
    if dual_modality:
        model_cfg1 = cfg['model1']
        model_cfg2 = cfg['model2']
        pretrained_weights1 = model_cfg1['backbone']['pretrained_weights'] if 'pretrained_weights' in model_cfg1['backbone'].keys() else None
        pretrained1 = model_cfg1['backbone'].get('pretrained', True) if 'pretrained' in model_cfg1['backbone'].keys() else True
        pretrained_weights2 = model_cfg2['backbone']['pretrained_weights'] if 'pretrained_weights' in model_cfg2['backbone'].keys() else None
        pretrained2 = model_cfg2['backbone'].get('pretrained', True) if 'pretrained' in model_cfg2['backbone'].keys() else True
        assert 'name' in model_cfg1, "Error - model1 name must be specified in the config"
        assert 'name' in model_cfg2, "Error - model2 name must be specified in the config"
        assert 'num_classes' in model_cfg1['head'], "Error - num_classes must be specified in the head config of model1"
        assert 'num_classes' in model_cfg2['head'], "Error - num_classes must be specified in the head config of model2"
        if ('pretrained_weights' not in model_cfg1 or model_cfg1.get('pretrained_weights', None) is None)  and model_cfg1.get('pretrained', True):
            logger.warning("Pretrained weights not specified for model1, using default backbone pretrained weights")
        if ('pretrained_weights' not in model_cfg2 or model_cfg2.get('pretrained_weights', None) is None)  and model_cfg2.get('pretrained', True):
            logger.warning("Pretrained weights not specified for model2, using default backbone pretrained weights")
        module_name1 = f"model.{ model_cfg1['name']}"
        model_class_name1 = model_cfg1['name'].capitalize()
        module1 = importlib.import_module(module_name1)
        model_class1 = getattr(module1, model_class_name1)
        module_name2 = f"model.{ model_cfg2['name']}"
        model_class_name2 = model_cfg2['name'].capitalize()
        module2 = importlib.import_module(module_name2)
        model_class2 = getattr(module2, model_class_name2)
        model1 = model_class1(**model_cfg1)
        model2 = model_class2(**model_cfg2)
        return (model1, model2)
    else:
        model_cfg = cfg['model']
        pretrained_weights = model_cfg['backbone']['pretrained_weights'] if 'pretrained_weights' in model_cfg['backbone'].keys() else None
        pretrained = model_cfg['backbone'].get('pretrained', True) if 'pretrained' in model_cfg['backbone'].keys() else True

        assert 'name' in model_cfg, "Error - model name must be specified in the config"
        assert 'num_classes' in model_cfg['head'], "Error - num_classes must be specified in the head config"

        if ('pretrained_weights' not in model_cfg or model_cfg.get('pretrained_weights', None) is None)  and model_cfg.get('pretrained', True):
            logger.warning("Pretrained weights not specified, using default backbone pretrained weights")

        # Load the model dynamically based on the name
        module_name = f"model.{ model_cfg['name']}"
        model_class_name = model_cfg['name'].capitalize()
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_class_name)
        
        # Initialize the model with the provided configuration
        model = model_class(**model_cfg)
        return model