#!/usr/bin/env python3
"""
Simple script to extract and visualize Cityscapes dataset with bounding boxes using CityscapesDataset class.
This script works independently and saves all visualizations to disk.

Usage:
    python simple_cityscapes_bbox_viz.py
    
Modify the paths at the top of the script to match your dataset location.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Disable matplotlib GUI backend for server usage
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('tests')[0])  # Add parent directory to path
from helpers import DEBUG
# Add the current directory to Python path for imports


# Import the CityscapesDataset class
from dataset.cityscapes import CityscapesDataset

# Configuration - Modify these paths according to your setup
DATA_ROOT = "./data"  # Change this to your dataset root
IMG_DIR = "cityscapes/leftImg8bit/train/aachen"  # Relative to DATA_ROOT
ANN_DIR = "cityscapes/gtFine/train/aachen"       # Relative to DATA_ROOT
OUTPUT_DIR = "./bbox_visualizations"
MIN_BBOX_AREA = 500  # Minimum area for bounding boxes
NUM_SAMPLES = 5      # Number of samples to process


def create_minimal_pipeline():
    """Create a minimal pipeline for loading images and annotations."""
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
    return pipeline


def initialize_cityscapes_dataset(data_root, img_dir, ann_dir, min_bbox_area):
    """Initialize the CityscapesDataset with bounding box extraction."""
    
    # Create absolute paths
    img_path = os.path.join(data_root, img_dir)
    ann_path = os.path.join(data_root, ann_dir)
    
    print(f"Looking for images in: {img_path}")
    print(f"Looking for annotations in: {ann_path}")
    
    if not os.path.exists(img_path):
        print(f"❌ Image directory not found: {img_path}")
        return None
    if not os.path.exists(ann_path):
        print(f"❌ Annotation directory not found: {ann_path}")
        return None
    
    try:
        # Create a simplified pipeline for visualization
        pipeline = create_minimal_pipeline()
        
        # Initialize CityscapesDataset with bounding box extraction
        dataset = CityscapesDataset(
            pipeline=pipeline,
            img_dir=img_path,
            ann_dir=ann_path,
            data_root=None,  # Already using absolute paths
            test_mode=True,   # For visualization purposes
            load_bboxes=True,
            extract_bboxes_from_masks=True,
            bbox_min_area=min_bbox_area,
            custom_classes=True,  # Assuming this is defined in dataset/custom.py
        )
        
        print(f"✓ Initialized Cityscapes dataset with {len(dataset)} samples")
        print(f"✓ Classes: {len(dataset.CLASSES)} classes")
        print(f"detection classes: {dataset.DSEC_DET_CLASSES}")
        return dataset
        
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        return None


def load_sample_data(dataset, idx):
    """Load image and segmentation data for a sample."""
    try:
        # Get image info
        img_info = dataset.img_infos[idx]
        img_path = os.path.join(dataset.img_dir, img_info['filename'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Load segmentation mask
        if 'ann' in img_info:
            seg_path = os.path.join(dataset.ann_dir, img_info['ann']['seg_map'])
            segmentation_mask = np.array(Image.open(seg_path))
        else:
            print(f"⚠ No annotation found for sample {idx}")
            segmentation_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        return image_np, segmentation_mask, img_info['filename']
        
    except Exception as e:
        print(f"❌ Error loading sample {idx}: {e}")
        return None, None, None


def extract_bboxes_using_dataset(dataset, idx, segmentation_mask):
    """Extract bounding boxes using the dataset's built-in methods."""
    try:
        # Try JSON polygon method first (most accurate)
        bboxes, class_ids, _ = dataset.extract_bboxes_from_json_polygons(idx)
        
        if len(bboxes) > 0:
            print(f"  ✓ Extracted {len(bboxes)} bboxes from JSON polygons")
            return bboxes, class_ids
        
        # Fallback to segmentation mask method
        bboxes, class_ids = dataset.extract_bboxes_from_mask(segmentation_mask)
        print(f"  ✓ Extracted {len(bboxes)} bboxes from segmentation mask")
        return bboxes, class_ids
        
    except Exception as e:
        print(f"  ❌ Error extracting bboxes: {e}")
        return [], []


def create_colored_segmentation_mask(segmentation_mask, palette):
    """Create colored segmentation mask using dataset palette."""
    colored_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(len(palette)):
        mask = segmentation_mask == class_id
        colored_mask[mask] = palette[class_id]
    
    return colored_mask


def visualize_and_save(dataset, image_np, segmentation_mask, bboxes, class_ids, save_path):
    """Create and save visualization with original image, segmentation, and bboxes."""
    
    # Create colored segmentation mask using dataset palette
    colored_mask = create_colored_segmentation_mask(segmentation_mask, dataset.PALETTE)
    
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
        if class_id < len(dataset.CLASSES):
            class_name = dataset.DSEC_DET_CLASSES[class_id]
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


def save_bbox_info(dataset, bboxes, class_ids, info_path):
    """Save bounding box information to text file."""
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
            class_name = dataset.CLASSES[class_id] if class_id < len(dataset.CLASSES) else f"class_{class_id}"
            f.write(f"  {class_name} (ID {class_id}): {count} boxes\n")
        
        f.write("\nDetailed box information:\n")
        f.write("-" * 40 + "\n")
        
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            class_name = dataset.CLASSES[class_id] if class_id < len(dataset.CLASSES) else f"class_{class_id}"
            
            f.write(f"Box {i+1}: {class_name}\n")
            f.write(f"  Coords: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
            f.write(f"  Size: {width:.1f} x {height:.1f} (area: {area:.1f})\n\n")


def process_sample(dataset, idx, output_dir):
    """Process a single sample using the dataset class."""
    print(f"\nProcessing sample {idx + 1}")
    
    # Load sample data
    image_np, segmentation_mask, filename = load_sample_data(dataset, idx)
    
    if image_np is None:
        return False
    
    print(f"  Filename: {filename}")
    print(f"  Image shape: {image_np.shape}")
    print(f"  Mask shape: {segmentation_mask.shape}")
    print(f"  Unique classes in mask: {len(np.unique(segmentation_mask))}")
    
    # Extract bounding boxes using dataset methods
    bboxes, class_ids = extract_bboxes_using_dataset(dataset, idx, segmentation_mask)
    
    # Create output filename
    base_name = os.path.splitext(filename)[0]
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    visualize_and_save(dataset, image_np, segmentation_mask, bboxes, class_ids, viz_path)
    
    # Save individual images
    Image.fromarray(image_np).save(
        os.path.join(output_dir, f"{base_name}_original.png")
    )
    
    colored_mask = create_colored_segmentation_mask(segmentation_mask, dataset.PALETTE)
    Image.fromarray(colored_mask).save(
        os.path.join(output_dir, f"{base_name}_segmentation.png")
    )
    
    # Save bounding box info
    info_path = os.path.join(output_dir, f"{base_name}_bbox_info.txt")
    save_bbox_info(dataset, bboxes, class_ids, info_path)
    
    print(f"  ✓ Extracted {len(bboxes)} bounding boxes")
    return True


def create_class_legend(dataset, output_dir):
    """Create a class legend using the dataset's palette."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color patches for each class
    patches_list = []
    for i, (class_name, color) in enumerate(zip(dataset.CLASSES, dataset.PALETTE)):
        color_normalized = [c/255.0 for c in color]  # Normalize to 0-1 range
        patch = patches.Patch(color=color_normalized, label=class_name)
        patches_list.append(patch)
    
    # Create legend
    ax.legend(handles=patches_list, loc='center', ncol=3, 
             frameon=False, fontsize=12)
    ax.axis('off')
    
    plt.title('Cityscapes Dataset Class Legend', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    legend_path = os.path.join(output_dir, "class_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Class legend saved: {legend_path}")


def main():
    """Main function."""
    print("Cityscapes Bounding Box Visualization (Using CityscapesDataset Class)")
    print("=" * 70)
    
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
    
    # Initialize dataset
    dataset = initialize_cityscapes_dataset(DATA_ROOT, IMG_DIR, ANN_DIR, MIN_BBOX_AREA)
    
    if dataset is None:
        print("❌ Failed to initialize dataset. Please check your dataset paths:")
        print(f"   DATA_ROOT = {DATA_ROOT}")
        print(f"   IMG_DIR = {IMG_DIR}")
        print(f"   ANN_DIR = {ANN_DIR}")
        print("\nExpected structure:")
        print("   DATA_ROOT/IMG_DIR/<city>/<image>_leftImg8bit.png")
        print("   DATA_ROOT/ANN_DIR/<city>/<image>_gtFine_labelTrainIds.png")
        print("   DATA_ROOT/ANN_DIR/<city>/<image>_gtFine_polygons.json (optional)")
        return 1
    
    # Create class legend
    create_class_legend(dataset, OUTPUT_DIR)
    
    # Process samples
    num_to_process = min(NUM_SAMPLES, len(dataset))
    print(f"\nProcessing {num_to_process} samples...")
    
    success_count = 0
    for i in tqdm(range(num_to_process), desc="Processing samples"):
        if process_sample(dataset, i, OUTPUT_DIR):
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
        f.write(f"Classes: {', '.join(dataset.CLASSES)}\n")
        f.write(f"Extraction methods: JSON polygons (preferred), segmentation masks (fallback)\n")
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"Successfully processed: {success_count}/{num_to_process} samples")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files for each sample:")
    print("  - *_visualization.png (combined view)")
    print("  - *_original.png (original image)")
    print("  - *_segmentation.png (colored segmentation)")
    print("  - *_bbox_info.txt (bounding box details)")
    print("  - class_legend.png (class color legend)")
    print("  - summary.txt (overall summary)")
    
    return 0


if __name__ == "__main__":
    exit(main())
