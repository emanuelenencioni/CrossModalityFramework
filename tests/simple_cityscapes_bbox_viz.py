#!/usr/bin/env python3
"""
Test script for CityscapesDataset class with bounding box extraction.
This script tests the dataset loading, image processing, and bbox extraction functionality.

Usage:
    python test_cityscapes_dataset.py
    
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('tests')[0])
from helpers import DEBUG

# Import the CityscapesDataset class
from dataset.cityscapes import CityscapesDataset

# Configuration - Modify these paths according to your setup
DATA_ROOT = "./data"  # Change this to your dataset root
IMG_DIR = "cityscapes/leftImg8bit/train/aachen"  # Relative to DATA_ROOT
ANN_DIR = "cityscapes/gtFine/train/aachen"       # Relative to DATA_ROOT
OUTPUT_DIR = "./cityscapes_dataset_test"
MIN_BBOX_AREA = 500  # Minimum area for bounding boxes
NUM_SAMPLES = 5      # Number of samples to process


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
        # Initialize CityscapesDataset with bounding box extraction and image loading
        dataset = CityscapesDataset(
            pipeline=[],  # Empty pipeline since we handle loading in pre_pipeline
            img_dir=img_path,
            ann_dir=ann_path,
            data_root=None,  # Already using absolute paths
            test_mode=True,   # For visualization purposes
            load_bboxes=True,  # Enable bbox loading
            extract_bboxes_from_masks=True,
            bbox_min_area=min_bbox_area,
            custom_classes=True,  # Enable detection classes mapping
            DETECTION_CLASSES=CityscapesDataset.DSEC_DET_CLASSES  # Pass detection classes
        )
        
        print(f"✓ Initialized Cityscapes dataset with {len(dataset)} samples")
        print(f"✓ Classes: {len(dataset.CLASSES)} classes")
        print(f"✓ Detection classes mapping: {dataset.DSEC_DET_CLASSES}")
        return dataset
        
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sample_loading(dataset, idx):
    """Test loading a single sample through the dataset interface."""
    print(f"\n--- Testing Sample {idx} ---")
    
    try:
        # Get sample data through dataset interface
        sample = dataset[idx]
        
        print(f"✓ Sample loaded successfully")
        print(f"  Keys in sample: {list(sample.keys())}")
        
        # Extract data from sample
        image = sample.get('image', None)
        bboxes = sample.get('BB', [])
        class_ids = sample.get('BB_class_ids', [])
        padding_info = sample.get('padding_info', {})
        
        # Print sample information
        if image is not None:
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        else:
            print(f"  ❌ No image found in sample")
            return None
        
        print(f"  Number of bounding boxes: {len(bboxes)}")
        print(f"  Class IDs: {class_ids}")
        
        if padding_info:
            print(f"  Padding info:")
            print(f"    Original size: {padding_info.get('original_size', 'N/A')}")
            print(f"    Padding: left={padding_info.get('pad_left', 0)}, top={padding_info.get('pad_top', 0)}")
            print(f"    Scale factor: {padding_info.get('scale_factor', 1.0):.3f}")
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_ids': class_ids,
            'padding_info': padding_info,
            'filename': sample['img_info']['filename']
        }
        
    except Exception as e:
        print(f"❌ Error loading sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_sample_with_dataset(dataset, sample_data, save_path):
    """Create visualization using dataset methods and sample data."""
    
    image = sample_data['image']
    bboxes = sample_data['bboxes']
    class_ids = sample_data['class_ids']
    filename = sample_data['filename']
    
    # Convert image from [0,1] to [0,255] for visualization
    if image.max() <= 1.0:
        vis_image = (image * 255).astype(np.uint8)
    else:
        vis_image = image.astype(np.uint8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Original padded/resized image
    axes[0].imshow(vis_image)
    axes[0].set_title(f'Processed Image (512x512)\n{filename}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Image with bounding boxes
    axes[1].imshow(vis_image)
    axes[1].set_title(f'With Bounding Boxes ({len(bboxes)} boxes)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Draw bounding boxes using detection class names
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, 
                               facecolor='none', alpha=0.8)
        axes[1].add_patch(rect)
        
        # Get class name from detection classes mapping
        if class_id in dataset.DSEC_DET_CLASSES:
            class_name = dataset.DSEC_DET_CLASSES[class_id]
        elif class_id < len(dataset.CLASSES):
            class_name = dataset.CLASSES[class_id]
        else:
            class_name = f"Class_{class_id}"
        
        # Add class label
        axes[1].text(x1, y1-5, class_name, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved visualization: {save_path}")


def test_bbox_extraction_methods(dataset, idx):
    """Test different bounding box extraction methods."""
    print(f"\n--- Testing Bbox Extraction Methods for Sample {idx} ---")
    
    try:
        # Test JSON polygon extraction
        json_bboxes, json_class_ids, json_masks = dataset.extract_bboxes_from_json_polygons(idx)
        print(f"  JSON polygons: {len(json_bboxes)} bboxes")
        
        # Test segmentation mask extraction
        seg_bboxes, seg_class_ids = dataset.get_bboxes_from_segmentation(idx)
        print(f"  Segmentation masks: {len(seg_bboxes)} bboxes")
        
        # Test combined method
        combined_bboxes = dataset.get_bbox_info(idx)
        print(f"  Combined method: {len(combined_bboxes)} bboxes")
        
        # Test padded and scaled bbox extraction
        transformed_bboxes, transformed_class_ids, padding_info = dataset.get_padded_and_scaled_bbox_info(idx)
        print(f"  Transformed (512x512): {len(transformed_bboxes)} bboxes")
        
        return {
            'json': (json_bboxes, json_class_ids),
            'segmentation': (seg_bboxes, seg_class_ids),
            'transformed': (transformed_bboxes, transformed_class_ids, padding_info)
        }
        
    except Exception as e:
        print(f"❌ Error testing bbox extraction: {e}")
        return None


def save_sample_info(dataset, sample_data, bbox_results, info_path):
    """Save detailed sample information to text file."""
    with open(info_path, 'w') as f:
        f.write("Cityscapes Dataset Sample Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Filename: {sample_data['filename']}\n")
        f.write(f"Image shape: {sample_data['image'].shape}\n")
        f.write(f"Image dtype: {sample_data['image'].dtype}\n")
        f.write(f"Image range: [{sample_data['image'].min():.3f}, {sample_data['image'].max():.3f}]\n\n")
        
        # Padding information
        padding_info = sample_data['padding_info']
        if padding_info:
            f.write("Padding/Scaling Information:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Original size: {padding_info.get('original_size', 'N/A')}\n")
            f.write(f"Square size: {padding_info.get('square_size', 'N/A')}\n")
            f.write(f"Padding left: {padding_info.get('pad_left', 0)}\n")
            f.write(f"Padding top: {padding_info.get('pad_top', 0)}\n")
            f.write(f"Scale factor: {padding_info.get('scale_factor', 1.0):.3f}\n\n")
        
        # Bounding box extraction results
        if bbox_results:
            f.write("Bounding Box Extraction Results:\n")
            f.write("-" * 35 + "\n")
            
            json_bboxes, json_class_ids = bbox_results['json']
            f.write(f"JSON polygons: {len(json_bboxes)} bboxes\n")
            
            seg_bboxes, seg_class_ids = bbox_results['segmentation']
            f.write(f"Segmentation masks: {len(seg_bboxes)} bboxes\n")
            
            transformed_bboxes, transformed_class_ids, _ = bbox_results['transformed']
            f.write(f"Transformed (final): {len(transformed_bboxes)} bboxes\n\n")
            
            # Detailed bbox information
            if transformed_bboxes:
                f.write("Final Bounding Boxes (512x512 space):\n")
                f.write("-" * 40 + "\n")
                
                for i, (bbox, class_id) in enumerate(zip(transformed_bboxes, transformed_class_ids)):
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Get class name
                    if class_id in dataset.DSEC_DET_CLASSES:
                        class_name = dataset.DSEC_DET_CLASSES[class_id]
                    elif class_id < len(dataset.CLASSES):
                        class_name = dataset.CLASSES[class_id]
                    else:
                        class_name = f"Class_{class_id}"
                    
                    f.write(f"Box {i+1}: {class_name} (ID: {class_id})\n")
                    f.write(f"  Coords: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
                    f.write(f"  Size: {width:.1f} x {height:.1f} (area: {area:.1f})\n\n")


def test_dataset_functionality():
    """Main test function."""
    print("Cityscapes Dataset Class Test")
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
    
    # Initialize dataset
    dataset = initialize_cityscapes_dataset(DATA_ROOT, IMG_DIR, ANN_DIR, MIN_BBOX_AREA)
    
    if dataset is None:
        print("❌ Failed to initialize dataset. Please check your dataset paths:")
        print(f"   DATA_ROOT = {DATA_ROOT}")
        print(f"   IMG_DIR = {IMG_DIR}")
        print(f"   ANN_DIR = {ANN_DIR}")
        return 1
    
    # Test samples
    num_to_test = min(NUM_SAMPLES, len(dataset))
    print(f"\nTesting {num_to_test} samples...")
    
    success_count = 0
    for i in tqdm(range(num_to_test), desc="Testing samples"):
        print(f"\n{'='*60}")
        
        # Test sample loading through dataset interface
        sample_data = test_sample_loading(dataset, i)
        
        if sample_data is None:
            continue
        
        # Test bbox extraction methods
        bbox_results = test_bbox_extraction_methods(dataset, i)
        
        # Create output filename
        base_name = os.path.splitext(sample_data['filename'])[0]
        
        # Save visualization
        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_dataset_test.png")
        visualize_sample_with_dataset(dataset, sample_data, viz_path)
        
        # Save processed image
        processed_image = sample_data['image']
        if processed_image.max() <= 1.0:
            processed_image = (processed_image * 255).astype(np.uint8)
        Image.fromarray(processed_image).save(
            os.path.join(OUTPUT_DIR, f"{base_name}_processed_512x512.png")
        )
        
        # Save detailed info
        info_path = os.path.join(OUTPUT_DIR, f"{base_name}_test_info.txt")
        save_sample_info(dataset, sample_data, bbox_results, info_path)
        
        success_count += 1
        print(f"✓ Sample {i} test completed successfully")
    
    # Create detection classes legend
    create_detection_classes_legend(dataset, OUTPUT_DIR)
    
    # Save test summary
    save_test_summary(dataset, success_count, num_to_test, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("Dataset Test Complete!")
    print(f"Successfully tested: {success_count}/{num_to_test} samples")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files for each sample:")
    print("  - *_dataset_test.png (visualization)")
    print("  - *_processed_512x512.png (processed image)")
    print("  - *_test_info.txt (detailed information)")
    print("  - detection_classes_legend.png (class legend)")
    print("  - test_summary.txt (overall summary)")
    
    return 0


def create_detection_classes_legend(dataset, output_dir):
    """Create a legend for detection classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create patches for detection classes
    patches_list = []
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    # Extract detection class names (string keys only)
    detection_classes = [v for k, v in dataset.DSEC_DET_CLASSES.items() if isinstance(k, int)]
    
    for i, class_name in enumerate(detection_classes):
        color = colors[i % len(colors)]
        patch = patches.Patch(color=color, label=class_name)
        patches_list.append(patch)
    
    # Create legend
    ax.legend(handles=patches_list, loc='center', ncol=3, 
             frameon=False, fontsize=12)
    ax.axis('off')
    
    plt.title('DSEC Detection Classes Legend', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    legend_path = os.path.join(output_dir, "detection_classes_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detection classes legend saved: {legend_path}")


def save_test_summary(dataset, success_count, total_count, output_dir):
    """Save test summary information."""
    summary_path = os.path.join(output_dir, "test_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Cityscapes Dataset Test Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Samples tested: {success_count}/{total_count}\n")
        f.write(f"Dataset size: {len(dataset)} total samples\n")
        f.write(f"Minimum bbox area: {MIN_BBOX_AREA}\n")
        f.write(f"Target image size: 512x512\n\n")
        
        f.write("Dataset Configuration:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Image directory: {IMG_DIR}\n")
        f.write(f"Annotation directory: {ANN_DIR}\n")
        f.write(f"Image suffix: {dataset.img_suffix}\n")
        f.write(f"Segmentation suffix: {dataset.seg_map_suffix}\n")
        f.write(f"Load bboxes: {dataset.load_bboxes}\n")
        f.write(f"Extract from masks: {dataset.extract_bboxes_from_masks}\n\n")
        
        f.write("Class Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total classes: {len(dataset.CLASSES)}\n")
        f.write(f"Classes: {', '.join(dataset.CLASSES)}\n\n")
        
        f.write("Detection Classes Mapping:\n")
        f.write("-" * 30 + "\n")
        for k, v in dataset.DSEC_DET_CLASSES.items():
            if isinstance(k, int):
                f.write(f"  {k} -> {v}\n")
        
        f.write(f"\nTest completed successfully!\n")


if __name__ == "__main__":
    exit(test_dataset_functionality())