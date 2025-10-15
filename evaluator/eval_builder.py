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
    
    
    cfg_eval = cfg.get('evaluator', None)
    assert cfg_eval is not None, "Evaluator configuration cannot be empty"
    assert 'conf_threshold' in cfg_eval.keys(), " specify 'conf_threshold' evaluator param"
    assert 'nms_threshold' in cfg_eval.keys(), " specify 'nms_threshold' evaluator param"
    assert 'bb_num_classes' in cfg['dataset'].keys(), " specify 'bb_num_classes' dataset param"

    dataset_type = cfg['dataset']['name'].lower()

    confthre = cfg_eval['conf_threshold']
    nmsthre = cfg_eval['nms_threshold']
    if cfg.get('dual_modality', True):
        in_size1 = cfg['model1']['input_size'] if 'input_size' in cfg['model1'].keys() else 512
        in_size2 = cfg['model2']['input_size'] if 'input_size' in cfg['model2'].keys() else 512
        eval1 = _build_unimodal_evaluator(dataset_type, data_loader, img_size=(in_size1, in_size1), confthre=confthre, nmsthre=nmsthre, num_classes=cfg['dataset']['bb_num_classes'], device=cfg['device'])
        eval2 = _build_unimodal_evaluator(dataset_type, data_loader, img_size=(in_size2, in_size2), confthre=confthre, nmsthre=nmsthre, num_classes=cfg['dataset']['bb_num_classes'], device=cfg['device'])
        return eval1, eval2
    else:
        in_size = cfg['model']['input_size'] if 'input_size' in cfg['model'].keys() else 512
        return _build_unimodal_evaluator(dataset_type, data_loader, img_size=(in_size, in_size), confthre=confthre, nmsthre=nmsthre, num_classes=cfg['dataset']['bb_num_classes'], device=cfg['device'])




def _build_unimodal_evaluator(dataset_type, data_loader, img_size, confthre, nmsthre, num_classes, device):
    if dataset_type == 'dsec_night':
        return DSECEvaluator(data_loader, img_size=img_size, confthre=confthre, nmsthre=nmsthre, num_classes=num_classes, device=device)
    elif dataset_type == 'cityscapes':
        return CityscapesEvaluator(data_loader, img_size=img_size, confthre=confthre, nmsthre=nmsthre, num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types are: 'dsec', 'cityscapes'")