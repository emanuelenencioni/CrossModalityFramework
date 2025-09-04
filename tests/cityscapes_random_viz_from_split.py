#!/usr/bin/env python3
"""
Test script for CityscapesDataset class using cs_train.txt split file.
This script creates a dataset from the split file and saves random visualizations with bboxes.

Usage:
    python cityscapes_random_viz_from_split.py
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import random

# Disable matplotlib GUI backend for server usage
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('tests')[0])
from helpers import DEBUG

# Import the CityscapesDataset class
from dataset.cityscapes import CityscapesDataset

# Configuration
DATA_ROOT = None  # Will use absolute paths from txt file
SPLIT_FILE = "./dataset/cs_train.txt"  # Path to the split file
OUTPUT_DIR = "./cityscapes_random_viz"
MIN_BBOX_AREA = 500  # Minimum area for bounding boxes
NUM_RANDOM_SAMPLES = 10  # Number of random samples to visualize
DET_CLASSES = (
        "pedestrian",
        "rider", 
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
        "train",
    )

def initialize_cityscapes_dataset_from_split(split_file, min_bbox_area):
    """Initialize the CityscapesDataset using the split file."""
    
    # Check if split file exists
    if not os.path.exists(split_file):
        print(f"❌ Split file not found: {split_file}")
        return None
    
    # Read a few lines to understand the structure
    with open(split_file, 'r') as f:
        sample_lines = [f.readline().strip() for _ in range(3)]
    
    print(f"Sample paths from split file:")
    for line in sample_lines[:2]:
        print(f"  {line}")
    
    # Extract directory structure from the first path
    if sample_lines:
        sample_path = sample_lines[0]
        # Extract base directories
        if '/cityscapes/leftImg8bit/' in sample_path:
            # Find the root path before 'cityscapes'
            root_idx = sample_path.find('/cityscapes/leftImg8bit/')
            data_root = sample_path[:root_idx]
            img_dir = sample_path[:root_idx] + '/cityscapes/leftImg8bit/train/'
            ann_dir = sample_path[:root_idx] + '/cityscapes/gtFine/train/'
            
            print(f"Detected paths:")
            print(f"  Data root: {data_root}")
            print(f"  Image dir: {img_dir}")
            print(f"  Annotation dir: {ann_dir}")
            print(f"  Split file: {split_file}")
        else:
            print(f"❌ Unexpected path structure in split file")
            return None
    else:
        print(f"❌ Empty split file")
        return None
    
    try:
        # Initialize CityscapesDataset with the split file
        dataset = CityscapesDataset(
            pipeline=[],  # Empty pipeline since we handle loading manually
            img_dir=img_dir,
            ann_dir=ann_dir,
            split=split_file,
            data_root=None,  # Using absolute paths
            test_mode=True,   # For visualization purposes
            load_bboxes=True,  # Enable bbox loading
            extract_bboxes_from_masks=True,
            bbox_min_area=min_bbox_area,
            custom_classes=True,  # Enable detection classes mapping
        )
        
        print(f"✓ Initialized Cityscapes dataset with {len(dataset)} samples")
        print(f"✓ Classes: {len(dataset.CLASSES)} classes")
        print(f"✓ Detection classes mapping: {len(dataset.DSEC_DET_CLASSES)} detection classes")
        return dataset
        
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_sample_data(dataset, idx):
    """Get sample data using dataset's __getitem__ method."""
    print(f"\n--- Getting Sample {idx} ---")
    
    try:
        # Use the dataset's __getitem__ method directly
        sample = dataset[idx]
        
        print(f"  Sample keys: {list(sample.keys())}")
        
        # Extract data from the sample
        image = sample.get('image')
        bboxes = sample.get('BB', [])
        class_ids = sample.get('BB_class_ids', [])
        img_info = sample.get('img_info', {})
        filename = img_info.get('filename', f'sample_{idx}')
        
        print(f"  Filename: {filename}")
        print(f"  Image shape: {image.shape if image is not None else 'None'}")
        print(f"  Image dtype: {image.dtype if image is not None else 'None'}")
        print(f"  Number of bboxes: {len(bboxes)}")
        print(f"  Bboxes shape: {bboxes.shape if hasattr(bboxes, 'shape') else type(bboxes)}")
        print(f"  Class IDs shape: {class_ids.shape if hasattr(class_ids, 'shape') else type(class_ids)}")
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_ids': class_ids,
            'filename': filename,
            'sample': sample  # Keep full sample for debugging
        }
        
    except Exception as e:
        print(f"❌ Error getting sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def denormalize_image(tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image that was normalized with ImageNet mean/std."""
    if torch.is_tensor(tensor_image):
        # Clone to avoid modifying original
        denorm_image = tensor_image.clone()
        
        if denorm_image.dim() == 3 and denorm_image.shape[0] == 3:  # CHW format
            # Denormalize each channel
            for i in range(3):
                denorm_image[i] = denorm_image[i] * std[i] + mean[i]
            
            # Convert to HWC format for visualization
            denorm_image = denorm_image.permute(1, 2, 0)
        
        # Clamp to [0, 1] range
        denorm_image = torch.clamp(denorm_image, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        image_np = (denorm_image.numpy() * 255).astype(np.uint8)
        
    else:
        # If it's already a numpy array, assume it's in the right format
        image_np = np.array(tensor_image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    
    return image_np


def visualize_sample_with_bboxes(dataset, sample_data, save_path):
    """Create visualization with bounding boxes using the dataset's built-in method."""
    
    image = sample_data['image']
    bboxes = sample_data['bboxes']
    class_ids = sample_data['class_ids']
    filename = sample_data['filename']
    
    # Denormalize the image properly
    image_np = denormalize_image(image)
    
    # Convert numpy back to tensor for the dataset's draw method
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC to CHW
    
    # Convert tensors to numpy arrays if needed for counting
    if torch.is_tensor(bboxes):
        bboxes_np = bboxes.numpy()
    else:
        bboxes_np = bboxes
        
    # Count valid bboxes
    valid_bboxes = 0
    if len(bboxes_np) > 0:
        for bbox in bboxes_np:
            if len(bbox) >= 5 and bbox[0] >= 0:  # Valid bbox has class_id >= 0
                valid_bboxes += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Original processed image
    axes[0].imshow(image_np)
    axes[0].set_title(f'Processed Image\n{os.path.basename(filename)}', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Image with bounding boxes using dataset's method
    try:
        # Use the dataset's draw_bounding_boxes_on_image method
        image_with_bboxes = dataset.draw_bounding_boxes_on_image(image_tensor, bboxes)
        
        # Convert back to numpy for display
        if torch.is_tensor(image_with_bboxes):
            if image_with_bboxes.dim() == 3 and image_with_bboxes.shape[0] == 3:  # CHW format
                image_with_bboxes_np = image_with_bboxes.permute(1, 2, 0).numpy()
            else:
                image_with_bboxes_np = image_with_bboxes.numpy()
        else:
            image_with_bboxes_np = image_with_bboxes
        
        axes[1].imshow(image_with_bboxes_np.astype(np.uint8))
        
    except Exception as e:
        print(f"  ⚠ Error using dataset's draw method: {e}")
        # Fallback to manual drawing
        axes[1].imshow(image_np)
        
        # Draw bounding boxes manually as fallback
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown']
        
        if len(bboxes_np) > 0:
            for i, bbox in enumerate(bboxes_np):
                if len(bbox) < 5 or bbox[0] < 0:  # Skip invalid bboxes
                    continue
                # bbox format: [class_id, x, y, w, h]
                class_id, x_center, y_center, w, h = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                x = x_center - w * 0.5
                y = y_center - h * 0.5
                color = colors[i % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none', alpha=0.8)
                axes[1].add_patch(rect)
                
                # Get class name
                if hasattr(dataset, 'DSEC_DET_CLASSES') and int(class_id) in dataset.DSEC_DET_CLASSES:
                    class_name = dataset.DSEC_DET_CLASSES[int(class_id)]
                else:
                    class_name = f"Class_{int(class_id)}"
                
                # Add class label
                axes[1].text(x, y-5, f"{class_name} ({int(class_id)})", 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                            fontsize=9, color='white', fontweight='bold')
    
    axes[1].set_title(f'With Bounding Boxes ({valid_bboxes} boxes)', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved visualization: {save_path}")


def save_sample_info(sample_data, info_path):
    """Save detailed sample information."""
    with open(info_path, 'w') as f:
        f.write("Cityscapes Dataset Random Sample Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Filename: {sample_data['filename']}\n")
        
        image = sample_data['image']
        if image is not None:
            f.write(f"Image shape: {image.shape}\n")
            f.write(f"Image dtype: {image.dtype}\n")
            if torch.is_tensor(image):
                f.write(f"Image range: [{image.min():.3f}, {image.max():.3f}]\n")
            else:
                f.write(f"Image range: [{np.min(image):.3f}, {np.max(image):.3f}]\n")
        
        # Bounding box details
        bboxes = sample_data['bboxes']
        
        if len(bboxes) > 0:
            valid_count = 0
            f.write(f"\nBounding Boxes Details:\n")
            f.write("-" * 25 + "\n")
            
            for i, bbox in enumerate(bboxes):
                if len(bbox) < 5 or bbox[0] < 0:  # Skip invalid bboxes
                    continue
                    
                valid_count += 1
                class_id, x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                area = w * h
                
                # Get class name
                if int(class_id) in CityscapesDataset.DSEC_DET_CLASSES:
                    class_name = CityscapesDataset.DSEC_DET_CLASSES[int(class_id)]
                else:
                    class_name = f"Class_{int(class_id)}"
                
                f.write(f"Box {valid_count}: {class_name} (ID: {int(class_id)})\n")
                f.write(f"  Coords: [{x:.1f}, {y:.1f}] + [{w:.1f}, {h:.1f}]\n")
                f.write(f"  Area: {area:.1f} pixels\n\n")
            
            f.write(f"Total valid bboxes: {valid_count}\n")
        else:
            f.write(f"\nNo bounding boxes found.\n")
        
        # Full sample info for debugging
        f.write(f"\nFull Sample Keys: {list(sample_data['sample'].keys())}\n")


def create_detection_classes_legend(output_dir):
    """Create a legend for DSEC detection classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create patches for detection classes
    patches_list = []
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    # Extract detection class names and IDs
    detection_items = [(k, v) for k, v in CityscapesDataset.DSEC_DET_CLASSES.items() if isinstance(k, int)]
    detection_items.sort()  # Sort by class ID
    
    for i, (class_id, class_name) in enumerate(detection_items):
        color = colors[i % len(colors)]
        patch = patches.Patch(color=color, label=f"{class_id}: {class_name}")
        patches_list.append(patch)
    
    # Create legend
    ax.legend(handles=patches_list, loc='center', ncol=2, 
             frameon=False, fontsize=12)
    ax.axis('off')
    
    plt.title('DSEC Detection Classes Legend', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    legend_path = os.path.join(output_dir, "dsec_detection_classes_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detection classes legend saved: {legend_path}")


def main():
    """Main function."""
    print("Cityscapes Random Visualization from Split File")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Initialize dataset from split file
    dataset = initialize_cityscapes_dataset_from_split(SPLIT_FILE, MIN_BBOX_AREA)
    
    if dataset is None:
        print("❌ Failed to initialize dataset. Please check:")
        print(f"   Split file: {SPLIT_FILE}")
        return 1
    
    # Get random sample indices
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), 
                                  min(NUM_RANDOM_SAMPLES, total_samples))
    
    print(f"\nSelected random samples: {random_indices}")
    print(f"Processing {len(random_indices)} random samples...")
    
    success_count = 0
    for i, idx in enumerate(tqdm(random_indices, desc="Processing samples")):
        print(f"\n{'='*60}")
        print(f"Processing sample {i+1}/{len(random_indices)} (index {idx})")
        
        # Get sample data using dataset[idx]
        sample_data = get_sample_data(dataset, idx)
        
        if sample_data is None:
            continue
        
        # Create output filename based on the original filename
        base_name = os.path.splitext(os.path.basename(sample_data['filename']))[0]
        
        # Save visualization using dataset's draw method
        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_random_viz.png")
        visualize_sample_with_bboxes(dataset, sample_data, viz_path)  # Pass dataset as parameter
        
        # Save processed image (denormalized)
        image = sample_data['image']
        image_np = denormalize_image(image)
            
        Image.fromarray(image_np).save(
            os.path.join(OUTPUT_DIR, f"{base_name}_processed.png")
        )
        
        # Save detailed info
        info_path = os.path.join(OUTPUT_DIR, f"{base_name}_info.txt")
        save_sample_info(sample_data, info_path)
        
        success_count += 1
        print(f"✓ Sample {idx} processed successfully")
    
    # Create detection classes legend
    create_detection_classes_legend(OUTPUT_DIR)
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Cityscapes Random Visualization Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Split file used: {SPLIT_FILE}\n")
        f.write(f"Total samples in dataset: {total_samples}\n")
        f.write(f"Random samples processed: {success_count}/{len(random_indices)}\n")
        f.write(f"Minimum bbox area: {MIN_BBOX_AREA}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n\n")
        f.write("Random indices processed:\n")
        for idx in random_indices[:success_count]:
            f.write(f"  {idx}\n")
        f.write(f"\nGenerated files per sample:\n")
        f.write(f"  - *_random_viz.png (visualization with bboxes)\n")
        f.write(f"  - *_processed.png (processed image)\n")
        f.write(f"  - *_info.txt (detailed information)\n")
    
    print(f"\n{'='*60}")
    print("Random Visualization Complete!")
    print(f"Successfully processed: {success_count}/{len(random_indices)} samples")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - Random visualizations with bboxes")
    print("  - Processed images")
    print("  - Detailed information files")
    print("  - Detection classes legend")
    print("  - Visualization summary")
    
    return 0


if __name__ == "__main__":
    exit(main())