import yaml
import importlib
from loguru import logger
import torch
from utils.helpers import DEBUG
from pathlib import Path

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
        prtraind_w1_path = model_cfg1.get('pretrained_weights', None)
        prtraind_w2_path = model_cfg2.get('pretrained_weights', None)
        if (prtraind_w1_path is None) and model_cfg1.get('pretrained', True):
            logger.warning("Pretrained weights not specified for model1")
        if (prtraind_w2_path is None) and model_cfg2.get('pretrained', True):
            logger.warning("Pretrained weights not specified for model2")
        assert 'name' in model_cfg1, "Error - model1 name must be specified in the config"
        assert 'name' in model_cfg2, "Error - model2 name must be specified in the config"
        assert 'num_classes' in model_cfg1['head'], "Error - num_classes must be specified in the head config of model1"
        assert 'num_classes' in model_cfg2['head'], "Error - num_classes must be specified in the head config of model2"

        module_name1 = f"model.{ model_cfg1['name']}"
        model_class_name1 = model_cfg1['name'].capitalize()
        module1 = importlib.import_module(module_name1)
        model_class1 = getattr(module1, model_class_name1)
        model1 = model_class1(**model_cfg1)
        if prtraind_w1_path is not None:
            if DEBUG>=1: logger.info(f"Loading pretrained weights for model1 from {prtraind_w1_path}")
            prtraind_w1 = torch.load(prtraind_w1_path)
            if "config" in prtraind_w1.keys():
                if DEBUG>=1: logger.info("The pretrained weights contain a config file, using this to create the model, IGNORING the current config")
                #this ignoring is applied only on the backbone and head parameters, not on the top level parameters like freeze or name
                model_cfg1['backbone'] = prtraind_w1['config']['model']['backbone']
                model_cfg1['head'] = prtraind_w1['config']['model']['head']
                model1 = model_class1(**model_cfg1)

                model1.load_state_dict(get_model_dict(prtraind_w1), strict=False)

        module_name2 = f"model.{ model_cfg2['name']}"
        model_class_name2 = model_cfg2['name'].capitalize()
        module2 = importlib.import_module(module_name2)
        model_class2 = getattr(module2, model_class_name2)
        if prtraind_w2_path is not None:
            if DEBUG>=1: logger.info(f"Loading pretrained weights for model2 from {prtraind_w2_path}")
            prtraind_w2 = torch.load(prtraind_w2_path)
            if "config" in prtraind_w2.keys():
                if DEBUG>=1: logger.info("The pretrained weights contain a config file, using this to create the model, IGNORING the current config")
                #this ignoring is applied only on the backbone and head parameters, not on the top level parameters like freeze or name
                model_cfg2['backbone'] = prtraind_w2['config']['model']['backbone']
                model_cfg2['head'] = prtraind_w2['config']['model']['head']
                model2 = model_class2(**model_cfg2)

            model2.load_state_dict(get_model_dict(prtraind_w2), strict=False)
        model2 = model_class2(**model_cfg2)
        return (model1, model2)
    else:
        model_cfg = cfg['model']
        prtraind_w_path = model_cfg.get('pretrained_weights', None)

        assert 'name' in model_cfg, "Error - model name must be specified in the config"
        assert 'num_classes' in model_cfg['head'], "Error - num_classes must be specified in the head config"
        
        # Load the model dynamically based on the name
        module_name = f"model.{ model_cfg['name']}"
        model_class_name = model_cfg['name'].capitalize()
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_class_name)
        
        # Initialize the model with the provided configuration
        model = model_class(**model_cfg)
        if prtraind_w_path is not None:
            if DEBUG>=1: logger.info(f"Loading pretrained weights from {prtraind_w_path}")
            prtraind_w_path = torch.load(prtraind_w_path)
            model.load_state_dict(get_model_dict(prtraind_w_path), strict=False)
        return model
    

def get_model_dict(dict_):
    for key in ['model', 'model_state_dict']:
        if key in dict_.keys():
            return dict_[key]
    return dict_