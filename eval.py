import yaml
import sys
import os
import argparse
from pathlib import Path

import torch
from loguru import logger

from model.builder import build_model_from_cfg
import dataset.dataset_builder as dataset_builder
from dataset import dataloader_builder as dl_builder
from evaluator import eval_builder
from utils.helpers import DEBUG, set_seed
from evaluator.dsec_evaluator import DSECEvaluator


def load_checkpoint(checkpoint_path):
    """
    Load checkpoint file and extract model state and config.
    Args: checkpoint_path: Path to the .pth checkpoint file
    Returns: checkpoint dict containing model state and config
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Verify checkpoint structure
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' key with training configuration")
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain 'model' key with model state dict")
    
    return checkpoint

def merge_configs(train_cfg, eval_dataset_cfg):
    """
    Merge training config with evaluation dataset config.
    Keep model architecture from training, but use new dataset for evaluation.
    Args:
        train_cfg: Original training configuration
        eval_dataset_cfg: New dataset configuration for evaluation
    Returns: merged config dict
    """
    cfg = train_cfg.copy()
    # Override dataset configuration
    cfg['dataset'] = eval_dataset_cfg['dataset']
    cfg['evaluator'] = eval_dataset_cfg['evaluator']
    # Remove training-specific configs that aren't needed for evaluation
    cfg.pop('optimizer', None)
    cfg.pop('scheduler', None)
    cfg.pop('trainer', None)
    cfg.pop('logger', None)
    return cfg

def parse_eval_arguments():
    """
    Parse command line arguments for evaluation.
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate trained model on new dataset')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--dataset-config',
        type=str,
        required=True,
        help='Path to YAML config file for evaluation dataset'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run evaluation on (cuda/cpu)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation (overrides config)'
    )
    
    parser.add_argument('--conf-thre', type=float, default=None,
        help='Confidence threshold for detection (overrides config)'
    )
    parser.add_argument('--nms-thre', type=float, default=None, help='NMS threshold for detection (overrides config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_eval_arguments()
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    train_cfg = checkpoint['config']
    
    logger.info(f"Loaded model trained with config: {train_cfg.get('run_name', 'unknown')}")
    
    # Load evaluation dataset config
    logger.info(f"Loading evaluation dataset config from: {args.dataset_config}")
    with open(args.dataset_config, 'r') as f:
        eval_dataset_cfg = yaml.safe_load(f)
    
    # Merge configurations
    cfg = merge_configs(train_cfg, eval_dataset_cfg)
    
    # Override device
    cfg['device'] = args.device
    
    # Override batch size if provided
    if args.batch_size is not None:
        if 'dataloader' not in cfg:
            cfg['dataloader'] = {}
        cfg['dataloader']['batch_size'] = args.batch_size
    
    # Set seed for reproducibility
    if args.seed is not None:
        cfg['seed'] = args.seed
    set_seed(cfg)
    
    # Build model from config
    logger.info("Building model from checkpoint config...")
    model = build_model_from_cfg(cfg)
    
    # Load model weights
    logger.info("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    device = torch.device(cfg['device'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    
    # Build evaluation dataset and dataloader
    logger.info("Building evaluation dataset...")
    test_ds = dataset_builder.build_test(cfg)
    
    if test_ds is None:
        raise ValueError("No test dataset created. Check your dataset config.")
    
    logger.info(f"Evaluation dataset size: {len(test_ds)}")
    
    # Build dataloader
    logger.info("Building dataloader...")
    _, test_dl = dl_builder.build_from_config(None, test_ds, cfg)
    
    # Override evaluator thresholds if provided
    if 'evaluator' not in cfg:
        cfg['evaluator'] = {}
    
    if args.conf_thre is not None:
        cfg['evaluator']['conf_thre'] = args.conf_thre
    
    if args.nms_thre is not None:
        cfg['evaluator']['nms_thre'] = args.nms_thre
    
    # Build evaluator
    logger.info("Building evaluator...")
    evaluator = eval_builder.build_from_config(test_dl, cfg)
    
    if evaluator is None:
        logger.warning("No evaluator built from config, creating default DSECEvaluator")
        
        img_size = cfg['model']['backbone'].get('input_size', 512)
        conf_thre = cfg['evaluator'].get('conf_thre', 0.3)
        nms_thre = cfg['evaluator'].get('nms_thre', 0.6)
        num_classes = cfg['dataset'].get('bb_num_classes', 8)
        
        evaluator = DSECEvaluator(
            test_dl,
            img_size=(img_size, img_size),
            confthre=conf_thre,
            nmsthre=nms_thre,
            num_classes=num_classes,
            device=device
        )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset_config}")
    logger.info("="*80)
    
    with torch.no_grad():
        stats, class_ap, class_ar = evaluator.evaluate(model)
    
    # Print results
    logger.info("="*80)
    logger.info("Evaluation Results:")
    logger.info(f"AP@[IoU=0.50:0.95]: {stats[0]:.4f}")
    logger.info(f"AP@[IoU=0.50]:     {stats[1]:.4f}")
    logger.info(f"AP@[IoU=0.75]:     {stats[2]:.4f}")
    logger.info(f"AR@[maxDets=100]:  {stats[8]:.4f}")
    logger.info("="*80)
    
    if class_ap is not None:
        logger.info("\nPer-class AP:")
        for class_name, ap_value in class_ap.items():
            logger.info(f"  {class_name}: {ap_value:.2f}")
    
    if class_ar is not None:
        logger.info("\nPer-class AR:")
        for class_name, ar_value in class_ar.items():
            logger.info(f"  {class_name}: {ar_value:.2f}")
    
    # Save results to file
    results_path = Path(args.checkpoint).parent / "eval_results.yaml"
    results = {
        'checkpoint': args.checkpoint,
        'dataset_config': args.dataset_config,
        'metrics': {
            'AP_IoU_0.50:0.95': float(stats[0]),
            'AP_IoU_0.50': float(stats[1]),
            'AP_IoU_0.75': float(stats[2]),
            'AR_maxDets_100': float(stats[8]),
        },
        'per_class_AP': class_ap,
        'per_class_AR': class_ar,
    }
    
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()