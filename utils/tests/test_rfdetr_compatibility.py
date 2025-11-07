#!/usr/bin/env python3
"""
Test script to verify RF-DETR wrapper compatibility with builder.py
"""

import sys
import yaml
import torch
from pathlib import Path

# Add project root to path (go up two levels: tests -> utils -> CrossModalityFramework)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.builder import build_model_from_cfg
from loguru import logger

def test_rfdetr_wrapper():
    """Test RF-DETR wrapper with builder.py"""
    
    logger.info("=" * 60)
    logger.info("Testing RF-DETR Wrapper Compatibility with builder.py")
    logger.info("=" * 60)
    
    # Load config
    config_path = project_root / "configs" / "rfdetr_unimodal.yaml"
    logger.info(f"Loading config from: {config_path}")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Check required fields
    logger.info("\n1. Checking required config fields...")
    assert 'model' in cfg, "Missing 'model' in config"
    assert 'name' in cfg['model'], "Missing 'name' in model config"
    assert 'backbone' in cfg['model'], "Missing 'backbone' in model config"
    assert 'head' in cfg['model'], "Missing 'head' in model config"
    assert 'embed_dim' in cfg['model']['backbone'], "Missing 'embed_dim' in backbone config"
    assert 'input_size' in cfg['model']['backbone'], "Missing 'input_size' in backbone config"
    assert 'num_classes' in cfg['model']['head'], "Missing 'num_classes' in head config"
    logger.success("✓ All required fields present")
    
    # Build model using builder.py
    logger.info("\n2. Building model with builder.py...")
    try:
        model = build_model_from_cfg(cfg)
        logger.success("✓ Model built successfully")
    except Exception as e:
        logger.error(f"✗ Failed to build model: {e}")
        raise
    
    # Check model type
    logger.info("\n3. Checking model type...")
    from model.rfdetrwrapper import Rfdetrwrapper
    assert isinstance(model, Rfdetrwrapper), f"Expected Rfdetrwrapper, got {type(model)}"
    logger.success("✓ Model type correct")
    
    # Check model attributes
    logger.info("\n4. Checking model attributes...")
    assert hasattr(model, 'model'), "Missing 'model' attribute"
    assert hasattr(model, 'criterion'), "Missing 'criterion' attribute"
    assert hasattr(model, 'loss_keys'), "Missing 'loss_keys' attribute"
    assert hasattr(model, 'get_name'), "Missing 'get_name' method"
    logger.success("✓ All required attributes present")
    
    # Test forward pass (training mode with targets)
    logger.info("\n5. Testing forward pass with targets...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 2
    input_size = cfg['model']['backbone']['input_size']
    x = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    # Create dummy targets [B, N, 5] format: [class_id, x1, y1, x2, y2]
    max_objects = 10
    targets = torch.zeros(batch_size, max_objects, 5).to(device)
    
    # First batch: 2 objects
    targets[0, 0] = torch.tensor([0, 100, 100, 200, 200])  # class 0
    targets[0, 1] = torch.tensor([1, 300, 300, 400, 400])  # class 1
    targets[0, 2:] = -1  # padding
    
    # Second batch: 1 object
    targets[1, 0] = torch.tensor([2, 150, 150, 250, 250])  # class 2
    targets[1, 1:] = -1  # padding
    
    try:
        output = model(x, targets)
        logger.success("✓ Forward pass successful")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        raise
    
    # Check output format
    logger.info("\n6. Checking output format...")
    assert isinstance(output, dict), "Output should be a dictionary"
    assert 'total_loss' in output, "Missing 'total_loss' in output"
    assert 'losses' in output, "Missing 'losses' in output"
    assert 'backbone_features' in output, "Missing 'backbone_features' in output"
    logger.success("✓ Output format correct")
    
    # Check losses
    logger.info("\n7. Checking losses...")
    assert isinstance(output['losses'], dict), "'losses' should be a dictionary"
    logger.info(f"   Loss keys: {list(output['losses'].keys())}")
    logger.info(f"   Total loss: {output['total_loss'].item():.4f}")
    for key, value in output['losses'].items():
        logger.info(f"   {key}: {value.item():.4f}")
    logger.success("✓ Losses computed correctly")
    
    # Test forward pass (inference mode without targets)
    logger.info("\n8. Testing forward pass without targets (inference)...")
    try:
        with torch.no_grad():
            output_inf = model(x, targets=None)
        assert 'predictions' in output_inf, "Missing 'predictions' in inference output"
        logger.success("✓ Inference pass successful")
    except Exception as e:
        logger.error(f"✗ Inference pass failed: {e}")
        raise
    
    # Test get_name method
    logger.info("\n9. Testing get_name method...")
    model_name = model.get_name()
    logger.info(f"   Model name: {model_name}")
    assert model_name == "RF-DETR", f"Expected 'RF-DETR', got '{model_name}'"
    logger.success("✓ get_name method works")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.success("✓ ALL TESTS PASSED - RF-DETR wrapper is compatible with builder.py")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_rfdetr_wrapper()
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
