from typing import Dict, Any
from .dsec_evaluator import DSECEvaluator
from .cityscapes_evaluator import CityscapesEvaluator

def build_from_config(data_loader, cfg: Dict[str, Any]):
    """
    Factory method to create evaluators based on configuration.
    
    Args:
        cfg: Configuration dictionary containing 'dataset' key
        
    Returns:
        An evaluator instance (DSECEvaluator or CityscapesEvaluator)
        
    Raises:
        ValueError: If dataset type is not supported
    """
    dataset_type = cfg['dataset']['name'].lower()
    in_size = cfg['model']['input_size'] if 'input_size' in cfg['model'].keys() else 512
    cfg_eval = cfg.get('evaluator', None)
    assert cfg_eval is not None, "Evaluator configuration cannot be empty"
    assert 'conf_threshold' in cfg_eval.keys(), " specify 'conf_threshold' evaluator param"
    assert 'nms_threshold' in cfg_eval.keys(), " specify 'nms_threshold' evaluator param"
    assert 'bb_num_classes' in cfg['dataset'].keys(), " specify 'bb_num_classes' dataset param"
    confthre = cfg_eval['conf_threshold']
    nmsthre = cfg_eval['nms_threshold']
    if dataset_type == 'dsec_night':
        return DSECEvaluator(data_loader, img_size=(in_size, in_size), confthre=confthre, nmsthre=nmsthre, num_classes=cfg['dataset']['bb_num_classes'], device=cfg['device'])
    elif dataset_type == 'cityscapes':
        return CityscapesEvaluator(data_loader, img_size=(in_size, in_size), confthre=confthre, nmsthre=nmsthre, num_classes=cfg['dataset']['bb_num_classes'], device=cfg['device'])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                         f"Supported types are: 'dsec', 'cityscapes'")