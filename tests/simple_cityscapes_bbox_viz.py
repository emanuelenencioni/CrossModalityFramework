#!/usr/bin/env python3
"""
Simple script to extract and visualize Cityscapes dataset with bounding boxes.
This script works independently and saves all visualizations to disk.

Usage:
    python simple_cityscapes_bbox_viz.py
    
Modify the paths at the top of the script to match your dataset location.
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import masks_to_boxes

# Disable matplotlib GUI backend for server usage
import matplotlib
matplotlib.use('Agg')

# Configuration - Modify these paths according to your setup
DATA_ROOT = "./data"  # Change this to your dataset root
IMG_DIR = "cityscapes/leftImg8bit/val"  # Relative to DATA_ROOT
ANN_DIR = "cityscapes/gtFine/val"       # Relative to DATA_ROOT
OUTPUT_DIR = "./bbox_visualizations"
MIN_BBOX_AREA = 500  # Minimum area for bounding boxes
NUM_SAMPLES = 5      # Number of samples to process


def find_cityscapes_samples(data_root, img_dir, ann_dir):
    """Find Cityscapes image and annotation pairs."""
    img_path = os.path.join(data_root, img_dir)
    ann_path = os.path.join(data_root, ann_dir)
    
    print(f"Looking for images in: {img_path}")
    print(f"Looking for annotations in: {ann_path}")
    
    if not os.path.exists(img_path):
        print(f"❌ Image directory not found: {img_path}")
        return []
    if not os.path.exists(ann_path):
        print(f"❌ Annotation directory not found: {ann_path}")
        return []
    
    samples = []
    
    # Walk through the image directory
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if file.endswith('_leftImg8bit.png'):
                img_file = os.path.join(root, file)
                
                # Find corresponding annotation
                rel_path = os.path.relpath(img_file, img_path)
                ann_file = rel_path.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                ann_full_path = os.path.join(ann_path, ann_file)
                
                if os.path.exists(ann_full_path):
                    samples.append({
                        'image_path': img_file,
                        'annotation_path': ann_full_path,
                        'filename': file
                    })
                else:
                    print(f"⚠ Missing annotation for: {file}")
    
    print(f"✓ Found {len(samples)} valid image-annotation pairs")
    return samples


def extract_bboxes_from_mask(segmentation_mask, min_area=500):
    """
    Extract bounding boxes from segmentation mask using masks_to_boxes.
    
    Args:
        segmentation_mask (np.ndarray): Segmentation mask with class IDs
        min_area (int): Minimum area for bounding boxes
        
    Returns:
        tuple: (bboxes, class_ids) where bboxes are [x1, y1, x2, y2] format
    """
    # Get unique classes (excluding background)
    unique_classes = np.unique(segmentation_mask)
    unique_classes = unique_classes[unique_classes != 255]  # Remove ignore label
    unique_classes = unique_classes[unique_classes > 0]     # Remove background
    
    all_bboxes = []
    all_class_ids = []
    
    print(f"  Processing classes: {unique_classes}")
    
    for class_id in unique_classes:
        # Create binary mask for this class
        class_mask = (segmentation_mask == class_id).astype(np.uint8)
        
        # Find connected components for multiple instances
        try:
            from scipy import ndimage
            labeled_mask, num_features = ndimage.label(class_mask)
            
            for instance_id in range(1, num_features + 1):
                # Create mask for this instance
                instance_mask = (labeled_mask == instance_id).astype(bool)
                
                # Check minimum area
                if np.sum(instance_mask) < min_area:
                    continue
                
                # Convert to torch tensor for masks_to_boxes
                instance_tensor = torch.from_numpy(instance_mask)
                
                # Extract bounding box using torchvision.ops.masks_to_boxes
                bbox = masks_to_boxes(instance_tensor.unsqueeze(0))  # Add batch dimension
                
                if bbox.numel() > 0:  # If bbox was found
                    bbox = bbox.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                    all_bboxes.append(bbox)
                    all_class_ids.append(class_id)
                    
        except ImportError:
            print("  SciPy not available, using simple method...")
            # Fallback: treat entire class as single object
            if np.sum(class_mask) >= min_area:
                # Convert to torch tensor for masks_to_boxes
                class_tensor = torch.from_numpy(class_mask.astype(bool))
                
                # Extract bounding box
                bbox = masks_to_boxes(class_tensor.unsqueeze(0))  # Add batch dimension
                
                if bbox.numel() > 0:  # If bbox was found
                    bbox = bbox.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                    all_bboxes.append(bbox)
                    all_class_ids.append(class_id)
    
    print(f"  Extracted {len(all_bboxes)} bounding boxes")
    return all_bboxes, all_class_ids


def create_colored_segmentation_mask(segmentation_mask):
    """Create colored segmentation mask using Cityscapes palette."""
    # Cityscapes palette
    PALETTE = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    
    colored_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(len(PALETTE)):
        mask = segmentation_mask == class_id
        colored_mask[mask] = PALETTE[class_id]
    
    return colored_mask


def visualize_and_save(image_np, segmentation_mask, bboxes, class_ids, save_path):
    """Create and save visualization with original image, segmentation, and bboxes."""
    
    # Cityscapes class names
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]
    
    # Create colored segmentation mask
    colored_mask = create_colored_segmentation_mask(segmentation_mask)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Segmentation mask
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Combined: image with bounding boxes
    axes[2].imshow(image_np, alpha=0.8)
    axes[2].imshow(colored_mask, alpha=0.3)
    axes[2].set_title(f'Image + Segmentation + Bboxes ({len(bboxes)} boxes)', 
                     fontsize=14, fontweight='bold')
    
    # Draw bounding boxes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
        x1, y1, x2, y2 = bbox
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, 
                               facecolor='none', alpha=0.8)
        axes[2].add_patch(rect)
        
        # Add class label
        if class_id < len(CLASSES):
            class_name = CLASSES[class_id]
            axes[2].text(x1, y1-5, class_name, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                       fontsize=10, color='white', fontweight='bold')
    
    axes[2].axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()  # Close to free memory
    
    print(f"  ✓ Saved visualization: {save_path}")


def save_bbox_info(bboxes, class_ids, info_path):
    """Save bounding box information to text file."""
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]
    
    with open(info_path, 'w') as f:
        f.write("Bounding Box Information\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total bounding boxes: {len(bboxes)}\n")
        f.write(f"Minimum area threshold: {MIN_BBOX_AREA}\n\n")
        
        if len(bboxes) == 0:
            f.write("No bounding boxes found.\n")
            return
        
        # Group by class
        class_counts = {}
        for class_id in class_ids:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        f.write("Bounding boxes by class:\n")
        for class_id, count in sorted(class_counts.items()):
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}"
            f.write(f"  {class_name} (ID {class_id}): {count} boxes\n")
        
        f.write("\nDetailed box information:\n")
        f.write("-" * 40 + "\n")
        
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}"
            
            f.write(f"Box {i+1}: {class_name}\n")
            f.write(f"  Coords: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
            f.write(f"  Size: {width:.1f} x {height:.1f} (area: {area:.1f})\n\n")


def process_sample(sample, output_dir, sample_num):
    """Process a single sample."""
    print(f"\nProcessing sample {sample_num}: {sample['filename']}")
    
    try:
        # Load image and annotation
        image = Image.open(sample['image_path']).convert('RGB')
        image_np = np.array(image)
        
        segmentation_mask = np.array(Image.open(sample['annotation_path']))
        
        print(f"  Image shape: {image_np.shape}")
        print(f"  Mask shape: {segmentation_mask.shape}")
        print(f"  Unique classes in mask: {len(np.unique(segmentation_mask))}")
        
    except Exception as e:
        print(f"  ❌ Error loading files: {e}")
        return False
    
    # Extract bounding boxes
    bboxes, class_ids = extract_bboxes_from_mask(segmentation_mask, MIN_BBOX_AREA)
    
    # Create output filename
    base_name = os.path.splitext(sample['filename'])[0]
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    visualize_and_save(image_np, segmentation_mask, bboxes, class_ids, viz_path)
    
    # Save individual images
    Image.fromarray(image_np).save(
        os.path.join(output_dir, f"{base_name}_original.png")
    )
    colored_mask = create_colored_segmentation_mask(segmentation_mask)
    Image.fromarray(colored_mask).save(
        os.path.join(output_dir, f"{base_name}_segmentation.png")
    )
    
    # Save bounding box info
    info_path = os.path.join(output_dir, f"{base_name}_bbox_info.txt")
    save_bbox_info(bboxes, class_ids, info_path)
    
    print(f"  ✓ Extracted {len(bboxes)} bounding boxes")
    return True


def main():
    """Main function."""
    print("Simple Cityscapes Bounding Box Visualization")
    print("=" * 50)
    
    # Check dependencies
    try:
        import torch
        import torchvision
        print("✓ PyTorch and torchvision available")
    except ImportError as e:
        print(f"❌ PyTorch/torchvision not available: {e}")
        return 1
    
    try:
        import scipy
        print("✓ SciPy available for connected components")
    except ImportError:
        print("⚠ SciPy not available - will use fallback method")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Find samples
    samples = find_cityscapes_samples(DATA_ROOT, IMG_DIR, ANN_DIR)
    
    if not samples:
        print("❌ No samples found. Please check your dataset paths:")
        print(f"   DATA_ROOT = {DATA_ROOT}")
        print(f"   IMG_DIR = {IMG_DIR}")
        print(f"   ANN_DIR = {ANN_DIR}")
        print("\nExpected structure:")
        print("   DATA_ROOT/IMG_DIR/<city>/<image>_leftImg8bit.png")
        print("   DATA_ROOT/ANN_DIR/<city>/<image>_gtFine_labelTrainIds.png")
        return 1
    
    # Process samples
    num_to_process = min(NUM_SAMPLES, len(samples))
    print(f"\nProcessing {num_to_process} samples...")
    
    success_count = 0
    for i in range(num_to_process):
        if process_sample(samples[i], OUTPUT_DIR, i + 1):
            success_count += 1
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Cityscapes Bounding Box Extraction Summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"Samples processed: {success_count}/{num_to_process}\n")
        f.write(f"Minimum bbox area: {MIN_BBOX_AREA}\n")
        f.write(f"Dataset root: {DATA_ROOT}\n")
        f.write(f"Image directory: {IMG_DIR}\n")
        f.write(f"Annotation directory: {ANN_DIR}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n")
    
    print(f"\n{'='*50}")
    print("Processing complete!")
    print(f"Successfully processed: {success_count}/{num_to_process} samples")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files for each sample:")
    print("  - *_visualization.png (combined view)")
    print("  - *_original.png (original image)")
    print("  - *_segmentation.png (colored segmentation)")
    print("  - *_bbox_info.txt (bounding box details)")
    print("  - summary.txt (overall summary)")
    
    return 0


if __name__ == "__main__":
    exit(main())
