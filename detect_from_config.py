import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
import os
import glob
from pathlib import Path

import random

# Import your framework components
from model.builder import build_model_from_cfg
from model.backbone import DualModalityBackbone, UnimodalBackbone
from model.yolox_head import YOLOXHead
import dataset.dataset_builder as dataset_builder
from dataset.cityscapes import CityscapesDataset
from helpers import DEBUG
from torchvision.ops import nms

def get_args_parser():
    parser = argparse.ArgumentParser('Detection script using trained models', add_help=False)
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--input_image', type=str, help='Path to single input image')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./detection_results', help='Output directory for results')
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to process (if using dataset)')
    parser.add_argument('--use_dataset', action='store_true', help='Use dataset instead of input images')
    parser.add_argument('--dataset_split', type=str, default='val', help='Dataset split to use (train/val)')
    return parser

def load_model_from_config(cfg, checkpoint_path, device):
    """Load model from config and checkpoint."""
    print(f"Loading model from config and checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']
    # Build model from config
    if 'head' in cfg['model'].keys():
        model = build_model_from_cfg(cfg['model'])
    else:
        # Backbone only model
        dual_modality = 'rgb_backbone' in cfg['model']['backbone'] and 'event_backbone' in cfg['model']['backbone']
        pretrained_weights = cfg['model']['backbone'].get('pretrained_weights', None)
        pretrained = cfg['model']['backbone'].get('pretrained', True)
        
        if dual_modality:
            model = DualModalityBackbone(
                rgb_backbone=cfg['model']['backbone']['rgb_backbone'],
                event_backbone=cfg['model']['backbone']['event_backbone'],
                embed_dim=cfg['model']['backbone']['embed_dim'],
                img_size=cfg['model']['backbone']['input_size'],
                pretrained=pretrained
            )
        else:
            backbone_name = cfg['model']['backbone'].get('rgb_backbone') or cfg['model']['backbone'].get('event_backbone')
            model = UnimodalBackbone(
                backbone_name,
                pretrained_weights=pretrained_weights,
                embed_dim=cfg['model']['backbone']['embed_dim'],
                img_size=cfg['model']['backbone']['input_size']
            )
    
    # Load checkpoint
    #checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")
    return model

def get_transform(input_size):
    """Get image transformation pipeline."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def detect_image(model, image, transform, device, confidence_threshold=0.5):
    """Run detection on a single image."""
    
    original_size = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(img_tensor)
    
    # YOLOX-style outputs
    # This would need to be implemented based on your YOLOX head structure
    boxes, scores = process_yolox_outputs(outputs, original_size, confidence_threshold)
     # Convert boxes from [0,1] to image coordinates
    boxes = rescale_boxes(boxes, original_size)
    
    return boxes, scores

def rescale_boxes(boxes, original_size):
    """Convert normalized boxes to image coordinates."""
    img_w, img_h = original_size
    
    # Convert from center format to corner format if needed
    # Assume cx, cy, w, h format and divide by 512
    x_c, y_c, w, h = ((boxes/512).unbind(-1))
    x1 = (x_c - 0.5 * h) * img_w
    y1 = (y_c - 0.5 * w) * img_h
    x2 = (x_c + 0.5 * h) * img_w
    y2 = (y_c + 0.5 * w) * img_h
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    return boxes

def process_yolox_outputs(outputs, original_size, confidence_threshold):
    """Process YOLOX model outputs (implement based on your YOLOX head)."""
    # This is a placeholder - implement based on your actual YOLOX head output format
    # You'll need to decode the outputs from your YOLOXHead
    outputs = outputs[0]
    boxes = outputs[0,:,:4]  
    objectness = outputs[:,:, 4] 
    class_scores = outputs[:,:, 5:].softmax(dim=0)[0]  # [5120, 8] - Apply softmax!
    scores = (class_scores * objectness.unsqueeze(-1))[0]  # [5120, 8]
   
    return boxes, scores

def plot_detections(image, boxes, scores, classes, output_path, confidence_threshold=0.5):
    """Plot detection results on image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    detection_count = 0
    for box, score_vec in zip(boxes, scores):
        max_score = score_vec.max().item()
        if max_score < confidence_threshold:
            continue
            
        class_id = score_vec.argmax().item()
        class_name = classes[class_id] if class_id < len(classes) else f"Class_{class_id}"
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, 
                               facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = f'{class_name}: {max_score:.2f}'
        ax.text(x1, y1-5, label, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='white', fontweight='bold')
        
        detection_count += 1
    
    plt.title(f'Detections: {detection_count} objects found')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path,    bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved detection result: {output_path}")
    return detection_count

def process_from_dataset(cfg, model, transform, device, args):
    """Process images from dataset."""
    print("Loading dataset for inference...")
    
    # Modify config for inference
    cfg_inference = cfg.copy()
    cfg_inference['dataset']['batch_size'] = 1
    cfg_inference['dataset']['num_workers'] = 1
    
    # Build dataset
    train_ds, test_ds = dataset_builder.build_from_config(cfg_inference['dataset'])
    dataset = test_ds if test_ds is not None and args.dataset_split == 'val' else train_ds
    
    if dataset is None:
        print("❌ No dataset available")
        return
    
    # Get classes
    if hasattr(dataset, 'DSEC_DET_CLASSES'):
        classes = list(dataset.DSEC_DET_CLASSES.values())
    elif hasattr(dataset, 'CLASSES'):
        classes = dataset.CLASSES
    else:
        classes = [f"class_{i}" for i in range(8)]  # Default classes
    
    # Process random samples
    num_samples = min(args.num_samples, len(dataset))
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"Processing {num_samples} random samples from dataset...")
    
    for i, idx in enumerate(random_indices):
        print(f"\nProcessing sample {i+1}/{num_samples} (index {idx})")
        
        try:
            sample = dataset[idx]
            image = sample['image']
            filename = sample.get('img_info', {}).get('filename', f'sample_{idx}')
            
            # Convert tensor to PIL Image for processing
            if torch.is_tensor(image):
                # Denormalize if needed
                if image.max() <= 1.0 and image.min() >= -3.0:  # Likely normalized
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
                
                if image.dim() == 3 and image.shape[0] == 3:  # CHW
                    image = image.permute(1, 2, 0)
                
                image_np = (image.numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Run detection
            boxes, scores = detect_image(model, pil_image, transform, device, args.confidence_threshold)
            
            # Plot results
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_detection.png")
            
            detection_count = plot_detections(pil_image, boxes, scores, classes, output_path, args.confidence_threshold)
            print(f"Found {detection_count} detections in {filename}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

def main(args):
    """Main detection function."""
    print("Detection Script using Trained Framework Models")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    args.device = "cpu"
    # Load model
    model = load_model_from_config(cfg, args.checkpoint_path, args.device)
    
    # Get transform
    input_size = cfg['model']['backbone']['input_size']
    transform = get_transform(input_size)
    
    # Define classes (adapt based on your dataset)
    classes = [
        'person', 'rider', 'car', 'bus', 'truck', 
        'bicycle', 'motorcycle', 'train'
    ]
    
    if args.use_dataset:
        # Process from dataset
        process_from_dataset(cfg, model, transform, args.device, args)
    
    elif args.input_dir:
        # Process directory of images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        
        image_paths = sort(image_paths)
        print(f"Found {len(image_paths)} images in {args.input_dir}")
        
        for img_path in image_paths:
            print(f"\nProcessing: {img_path}")
            
            try:
                image = Image.open(img_path).convert('RGB')
                boxes, scores = detect_image(model, image, transform, args.device, args.confidence_threshold)
                
                # Save result
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(args.output_dir, f"{base_name}_detection.png")
                
                detection_count = plot_detections(image, boxes, scores, classes, output_path, args.confidence_threshold)
                print(f"Found {detection_count} detections")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    elif args.input_image:
        # Process single image
        print(f"Processing single image: {args.input_image}")
        image = Image.open(args.input_image).convert('RGB')

        boxes, scores = detect_image(model, image, transform, args.device, args.confidence_threshold)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_detection.png")
        
        detection_count = plot_detections(image, boxes, scores, classes, output_path, args.confidence_threshold)
        print(f"Found {detection_count} detections")
    
    else:
        print("Please specify either --input_image, --input_dir, or --use_dataset")
        return
    
    print(f"\n✓ Detection complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Detection script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


    