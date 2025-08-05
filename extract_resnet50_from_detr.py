#!/usr/bin/env python3
"""
Script to extract ResNet50 backbone weights from DETR model.

This script loads the DETR model from Hugging Face, extracts the ResNet50 backbone weights,
and saves them in a format that can be used with other frameworks like PyTorch timm.

Usage:
    python extract_resnet50_from_detr.py [--output_path backbone_weights.pth] [--save_format state_dict]
"""

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection
import argparse
import os
from collections import OrderedDict


def extract_resnet50_backbone(model_name="facebook/detr-resnet-50", revision="no_timm"):
    """
    Extract ResNet50 backbone from DETR model.
    
    Args:
        model_name (str): The model name from Hugging Face
        revision (str): The model revision to use
        
    Returns:
        dict: State dict containing ResNet50 backbone weights
    """
    print(f"Loading DETR model: {model_name} (revision: {revision})")
    
    # Load the DETR model
    model = DetrForObjectDetection.from_pretrained(model_name, revision=revision)
    
    # Navigate to the actual ResNet50 backbone
    # DETR structure: model.model.backbone.backbone (the inner backbone is the ResNet50)
    print("Exploring DETR model structure...")
    
    # Try to find the ResNet50 backbone
    if hasattr(model.model.backbone, 'backbone'):
        # This should be the actual ResNet50
        resnet50_backbone = model.model.backbone.backbone
        print("Found ResNet50 at: model.model.backbone.backbone")
    elif hasattr(model.model.backbone, 'conv_encoder'):
        # Alternative path for some DETR versions
        if hasattr(model.model.backbone.conv_encoder, 'model'):
            resnet50_backbone = model.model.backbone.conv_encoder.model
            print("Found ResNet50 at: model.model.backbone.conv_encoder.model")
        else:
            resnet50_backbone = model.model.backbone.conv_encoder
            print("Found ResNet50 at: model.model.backbone.conv_encoder")
    else:
        # Fallback to the entire backbone
        resnet50_backbone = model.model.backbone
        print("Using entire backbone (structure may be different)")
    
    print("Extracting ResNet50 backbone weights...")
    
    # Get the state dict of the ResNet50 backbone
    backbone_state_dict = resnet50_backbone.state_dict()
    
    print(f"Extracted {len(backbone_state_dict)} parameters from ResNet50 backbone")
    
    # Print some key information about the backbone
    print("\nBackbone architecture:")
    print(resnet50_backbone)
    
    # Print a few sample keys to understand the structure
    print(f"\nSample parameter keys:")
    sample_keys = list(backbone_state_dict.keys())[:10]
    for key in sample_keys:
        print(f"  {key}")
    if len(backbone_state_dict) > 10:
        print(f"  ... and {len(backbone_state_dict) - 10} more")
    
    return backbone_state_dict, resnet50_backbone


def convert_to_timm_format(state_dict):
    """
    Convert DETR ResNet50 state dict to timm ResNet50 format.
    
    Args:
        state_dict (dict): Original DETR backbone state dict
        
    Returns:
        dict: Converted state dict compatible with timm ResNet50
    """
    converted_state_dict = OrderedDict()
    
    # Print some keys to understand the structure
    print(f"\nConverting DETR state dict to timm format...")
    print(f"Original keys sample:")
    sample_keys = list(state_dict.keys())[:5]
    for key in sample_keys:
        print(f"  {key}")
    
    # Mapping from DETR naming to timm naming
    for key, value in state_dict.items():
        new_key = key
        
        # Handle different DETR backbone structures
        if key.startswith('conv_encoder.model.'):
            # Remove the conv_encoder.model prefix
            new_key = key.replace('conv_encoder.model.', '')
        elif key.startswith('body.'):
            # Remove 'body.' prefix (some DETR versions)
            new_key = key.replace('body.', '')
        
        # Handle DETR-specific naming patterns
        if 'embedder.embedder.convolution.weight' in new_key:
            new_key = 'conv1.weight'
        elif 'embedder.embedder.normalization' in new_key:
            new_key = new_key.replace('embedder.embedder.normalization', 'bn1')
        
        # Handle encoder stages -> layers mapping
        if 'encoder.stages.' in new_key:
            # Map stages to layers
            new_key = new_key.replace('encoder.stages.0.layers.', 'layer1.')
            new_key = new_key.replace('encoder.stages.1.layers.', 'layer2.')
            new_key = new_key.replace('encoder.stages.2.layers.', 'layer3.')
            new_key = new_key.replace('encoder.stages.3.layers.', 'layer4.')
            
            # Handle shortcut -> downsample
            new_key = new_key.replace('.shortcut.convolution.weight', '.downsample.0.weight')
            new_key = new_key.replace('.shortcut.normalization', '.downsample.1')
            
            # Handle layer convolutions and normalizations
            new_key = new_key.replace('.layer.0.convolution.weight', '.conv1.weight')
            new_key = new_key.replace('.layer.0.normalization', '.bn1')
            new_key = new_key.replace('.layer.1.convolution.weight', '.conv2.weight')
            new_key = new_key.replace('.layer.1.normalization', '.bn2')
            new_key = new_key.replace('.layer.2.convolution.weight', '.conv3.weight')
            new_key = new_key.replace('.layer.2.normalization', '.bn3')
        
        # Clean up any remaining patterns
        new_key = new_key.replace('.weight', '.weight')
        new_key = new_key.replace('.bias', '.bias')
        new_key = new_key.replace('.running_mean', '.running_mean')
        new_key = new_key.replace('.running_var', '.running_var')
        new_key = new_key.replace('.num_batches_tracked', '.num_batches_tracked')
        
        # Only keep keys that look like standard ResNet50 keys
        if any(pattern in new_key for pattern in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
            converted_state_dict[new_key] = value
            if len(converted_state_dict) <= 5:  # Show first few conversions
                print(f"  {key} -> {new_key}")
    
    print(f"Converted {len(converted_state_dict)} parameters")
    return converted_state_dict


def save_backbone_weights(state_dict, backbone_model, output_path, save_format="timm_compatible"):
    """
    Save the extracted backbone weights.
    
    Args:
        state_dict (dict): The backbone state dict
        backbone_model (nn.Module): The backbone model
        output_path (str): Path to save the weights
        save_format (str): Format to save ('state_dict', 'full_model', 'timm_compatible')
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if save_format == "state_dict":
        # Save only the state dict
        torch.save(state_dict, output_path)
        print(f"Saved backbone state dict to: {output_path}")
        
    elif save_format == "full_model":
        # Save the full model
        torch.save(backbone_model, output_path)
        print(f"Saved full backbone model to: {output_path}")
        
    elif save_format == "timm_compatible":
        # Convert and save in timm-compatible format
        timm_state_dict = convert_to_timm_format(state_dict)
        torch.save(timm_state_dict, output_path)
        print(f"Saved timm-compatible state dict to: {output_path}")
        
    else:
        raise ValueError(f"Unknown save format: {save_format}")


def load_backbone_weights_example(weights_path):
    """
    Example of how to load the extracted weights into a timm ResNet50 model.
    
    Args:
        weights_path (str): Path to the saved weights
    """
    try:
        import timm
        
        print(f"\nExample: Loading weights into timm ResNet50...")
        
        # Create a timm ResNet50 model
        timm_model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        
        # Load the extracted weights
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Load weights into the model
        missing_keys, unexpected_keys = timm_model.load_state_dict(state_dict, strict=False)
        
        print(f"Successfully loaded weights!")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            
        return timm_model
        
    except ImportError:
        print("timm not available for loading example")
        return None


def print_backbone_info(backbone_model):
    """
    Print detailed information about the backbone model.
    
    Args:
        backbone_model (nn.Module): The backbone model
    """
    print("\n" + "="*50)
    print("BACKBONE MODEL INFORMATION")
    print("="*50)
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone_model.parameters())
    trainable_params = sum(p.numel() for p in backbone_model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print model structure
    print(f"\nModel structure:")
    print(backbone_model)
    
    # Test with dummy input
    print(f"\nTesting with dummy input (1, 3, 224, 224)...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        try:
            output = backbone_model(dummy_input)
            if isinstance(output, torch.Tensor):
                print(f"Output shape: {output.shape}")
            elif isinstance(output, (list, tuple)):
                print(f"Output shapes: {[o.shape for o in output]}")
            else:
                print(f"Output type: {type(output)}")
        except Exception as e:
            print(f"Error during forward pass: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet50 backbone weights from DETR model")
    parser.add_argument("--output_path", default="resnet50_backbone_from_detr.pth", 
                       help="Path to save the extracted weights")
    parser.add_argument("--save_format", choices=["state_dict", "full_model", "timm_compatible"], 
                       default="timm_compatible", help="Format to save the weights")
    parser.add_argument("--model_name", default="facebook/detr-resnet-50", 
                       help="DETR model name from Hugging Face")
    parser.add_argument("--revision", default="no_timm", 
                       help="Model revision")
    parser.add_argument("--show_info", action="store_true", 
                       help="Show detailed information about the backbone")
    parser.add_argument("--test_load", action="store_true", 
                       help="Test loading the weights into a timm model")
    
    args = parser.parse_args()
    
    try:
        # Extract backbone weights
        state_dict, backbone_model = extract_resnet50_backbone(args.model_name, args.revision)
        
        # Show backbone info if requested
        if args.show_info:
            print_backbone_info(backbone_model)
        
        # Save the weights
        save_backbone_weights(state_dict, backbone_model, args.output_path, args.save_format)
        
        # Test loading if requested
        if args.test_load:
            load_backbone_weights_example(args.output_path)
            
        print(f"\n✅ Successfully extracted and saved ResNet50 backbone weights!")
        print(f"📁 Saved to: {args.output_path}")
        print(f"💾 Format: {args.save_format}")
        
        # Show usage example
        print(f"\n📖 Usage example:")
        print(f"```python")
        print(f"import torch")
        print(f"import timm")
        print(f"")
        print(f"# Load the extracted weights")
        print(f"state_dict = torch.load('{args.output_path}', map_location='cpu')")
        print(f"")
        print(f"# Create a timm ResNet50 model")
        print(f"model = timm.create_model('resnet50', pretrained=False, num_classes=0)")
        print(f"")
        print(f"# Load the weights")
        print(f"model.load_state_dict(state_dict, strict=False)")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
