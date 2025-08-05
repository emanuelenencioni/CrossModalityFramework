#!/usr/bin/env python3
"""
Quick test script to demonstrate bounding box extraction from segmentation masks.
This script creates sample data to test the masks_to_boxes functionality.
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

def create_test_segmentation_data():
    """Create test segmentation data similar to Cityscapes."""
    print("Creating test segmentation data...")
    
    # Create a test image and segmentation mask
    height, width = 300, 500
    
    # Create a sample RGB image
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structure to make it look more realistic
    # Sky (class 10)
    image[0:100, :] = [135, 206, 235]  # Sky blue
    
    # Buildings (class 2)
    image[100:200, 0:150] = [128, 128, 128]  # Gray buildings
    image[100:180, 300:500] = [96, 96, 96]   # Another building
    
    # Road (class 0)
    image[200:300, :] = [64, 64, 64]  # Dark gray road
    
    # Create corresponding segmentation mask
    segmentation = np.zeros((height, width), dtype=np.uint8)
    
    # Sky (class 10)
    segmentation[0:100, :] = 10
    
    # Buildings (class 2)
    segmentation[100:200, 0:150] = 2   # Building 1
    segmentation[100:180, 300:500] = 2  # Building 2
    
    # Road (class 0)
    segmentation[200:300, :] = 0
    
    # Add some cars (class 13)
    segmentation[220:260, 100:160] = 13  # Car 1
    segmentation[210:240, 250:300] = 13  # Car 2
    segmentation[225:255, 350:390] = 13  # Car 3
    
    # Add some people (class 11)
    segmentation[180:220, 200:215] = 11  # Person 1
    segmentation[190:230, 400:415] = 11  # Person 2
    
    # Add vegetation (class 8)
    segmentation[120:180, 450:500] = 8   # Trees
    
    return image, segmentation

def extract_bboxes_from_mask(segmentation_mask, min_area=100):
    """Extract bounding boxes using masks_to_boxes."""
    print("Extracting bounding boxes...")
    
    unique_classes = np.unique(segmentation_mask)
    unique_classes = unique_classes[unique_classes > 0]  # Remove background
    
    all_bboxes = []
    all_class_ids = []
    
    print(f"Processing classes: {unique_classes}")
    
    for class_id in unique_classes:
        # Create binary mask for this class
        class_mask = (segmentation_mask == class_id).astype(np.uint8)
        
        # Find connected components
        try:
            from scipy import ndimage
            labeled_mask, num_features = ndimage.label(class_mask)
            
            for instance_id in range(1, num_features + 1):
                instance_mask = (labeled_mask == instance_id).astype(bool)
                
                if np.sum(instance_mask) < min_area:
                    continue
                
                # Use masks_to_boxes
                instance_tensor = torch.from_numpy(instance_mask)
                bbox = masks_to_boxes(instance_tensor.unsqueeze(0))
                
                if bbox.numel() > 0:
                    bbox = bbox.squeeze(0).numpy()
                    all_bboxes.append(bbox)
                    all_class_ids.append(class_id)
                    print(f"  Class {class_id}, instance {instance_id}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    
        except ImportError:
            print("  Using fallback method (no scipy)...")
            if np.sum(class_mask) >= min_area:
                class_tensor = torch.from_numpy(class_mask.astype(bool))
                bbox = masks_to_boxes(class_tensor.unsqueeze(0))
                
                if bbox.numel() > 0:
                    bbox = bbox.squeeze(0).numpy()
                    all_bboxes.append(bbox)
                    all_class_ids.append(class_id)
                    print(f"  Class {class_id}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    
    return all_bboxes, all_class_ids

def visualize_results(image, segmentation_mask, bboxes, class_ids, output_dir):
    """Create and save visualization."""
    print("Creating visualization...")
    
    # Cityscapes class names and colors
    CLASS_NAMES = {
        0: 'road', 2: 'building', 8: 'vegetation', 10: 'sky', 
        11: 'person', 13: 'car'
    }
    
    PALETTE = {
        0: [128, 64, 128],   # road
        2: [70, 70, 70],     # building  
        8: [107, 142, 35],   # vegetation
        10: [70, 130, 180],  # sky
        11: [220, 20, 60],   # person
        13: [0, 0, 142]      # car
    }
    
    # Create colored segmentation mask
    colored_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        mask = segmentation_mask == class_id
        colored_mask[mask] = color
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Test Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Combined with bounding boxes
    axes[2].imshow(image, alpha=0.7)
    axes[2].imshow(colored_mask, alpha=0.4)
    axes[2].set_title(f'Bounding Boxes ({len(bboxes)} boxes)', fontsize=14, fontweight='bold')
    
    # Draw bounding boxes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=3, edgecolor=color,
                               facecolor='none', alpha=0.8)
        axes[2].add_patch(rect)
        
        # Add label
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        axes[2].text(x1, y1-5, class_name,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   fontsize=10, color='white', fontweight='bold')
    
    axes[2].axis('off')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'test_bbox_extraction.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {save_path}")
    
    # Save individual components
    Image.fromarray(image).save(os.path.join(output_dir, 'test_image.png'))
    Image.fromarray(colored_mask).save(os.path.join(output_dir, 'test_segmentation.png'))
    
    # Save info
    info_path = os.path.join(output_dir, 'test_bbox_info.txt')
    with open(info_path, 'w') as f:
        f.write("Test Bounding Box Extraction Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total bounding boxes: {len(bboxes)}\n")
        f.write(f"Image size: {image.shape}\n")
        f.write(f"Segmentation classes: {np.unique(segmentation_mask)}\n\n")
        
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = bbox
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            f.write(f"Box {i+1}: {class_name}\n")
            f.write(f"  Coordinates: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
            f.write(f"  Size: {x2-x1:.1f} x {y2-y1:.1f}\n\n")
    
    print(f"✓ Info saved: {info_path}")

def main():
    """Main test function."""
    print("Testing Bounding Box Extraction from Segmentation Masks")
    print("=" * 60)
    
    # Check dependencies
    try:
        import torch
        import torchvision
        print("✓ PyTorch and TorchVision available")
    except ImportError as e:
        print(f"❌ PyTorch/TorchVision missing: {e}")
        return 1
    
    try:
        import scipy
        print("✓ SciPy available")
    except ImportError:
        print("⚠ SciPy not available - using fallback method")
    
    # Create test data
    image, segmentation = create_test_segmentation_data()
    print(f"✓ Created test data: {image.shape} image, {segmentation.shape} segmentation")
    
    # Extract bounding boxes
    bboxes, class_ids = extract_bboxes_from_mask(segmentation, min_area=100)
    print(f"✓ Extracted {len(bboxes)} bounding boxes")
    
    # Visualize results
    output_dir = "./test_bbox_output"
    visualize_results(image, segmentation, bboxes, class_ids, output_dir)
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"Check output directory: {output_dir}")
    print("Generated files:")
    print("  - test_bbox_extraction.png (combined visualization)")
    print("  - test_image.png (original test image)")
    print("  - test_segmentation.png (colored segmentation)")
    print("  - test_bbox_info.txt (bounding box details)")
    
    print(f"\nThis demonstrates the masks_to_boxes() functionality")
    print("You can now run the main scripts on your Cityscapes dataset!")
    
    return 0

if __name__ == "__main__":
    exit(main())
