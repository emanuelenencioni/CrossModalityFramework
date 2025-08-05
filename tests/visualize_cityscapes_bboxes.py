#!/usr/bin/env python3
"""
Script to extract images from Cityscapes dataset and visualize segmentation with bounding boxes.
This script is designed to run on servers without GUI - it only saves images to disk.

Usage:
    python visualize_cityscapes_bboxes.py --data_root /path/to/cityscapes --output_dir ./visualizations --num_samples 5
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import masks_to_boxes
import sys

# Disable matplotlib GUI backend for server usage
import matplotlib
matplotlib.use('Agg')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.cityscapes import CityscapesDataset
from helpers import DEBUG


class CityscapesBBoxVisualizer:
    """Visualizer for Cityscapes dataset with bounding box extraction and visualization."""
    
    def __init__(self, data_root, img_dir="leftImg8bit/val", ann_dir="gtFine/val", 
                 output_dir="./bbox_visualizations", min_bbox_area=500):
        """
        Initialize the visualizer.
        
        Args:
            data_root (str): Root directory of Cityscapes dataset
            img_dir (str): Image directory relative to data_root
            ann_dir (str): Annotation directory relative to data_root
            output_dir (str): Output directory for visualizations
            min_bbox_area (int): Minimum area for bounding boxes to be included
        """
        self.data_root = data_root
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.output_dir = output_dir
        self.min_bbox_area = min_bbox_area
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataset
        try:
            # Create a simple pipeline for loading data
            pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
            ]
            
            # For now, we'll load manually without pipeline
            self.dataset = self._load_dataset_manually()
            print(f"Dataset loaded successfully with {len(self.dataset)} samples")
            
        except Exception as e:
            print(f"Error initializing dataset: {e}")
            self.dataset = None
    
    def _load_dataset_manually(self):
        """Load dataset manually without complex pipeline."""
        img_path = os.path.join(self.data_root, self.img_dir)
        ann_path = os.path.join(self.data_root, self.ann_dir)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image directory not found: {img_path}")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation directory not found: {ann_path}")
        
        # Find all image files
        samples = []
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
        
        return samples
    
    def extract_bboxes_from_mask(self, segmentation_mask):
        """
        Extract bounding boxes from segmentation mask using masks_to_boxes.
        
        Args:
            segmentation_mask (np.ndarray): Segmentation mask with class IDs
            
        Returns:
            tuple: (bboxes, class_ids) where bboxes are [x1, y1, x2, y2] format
        """
        # Get unique classes (excluding background)
        unique_classes = np.unique(segmentation_mask)
        unique_classes = unique_classes[unique_classes > 0]  # Remove background
        
        all_bboxes = []
        all_class_ids = []
        
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
                    if np.sum(instance_mask) < self.min_bbox_area:
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
                # Fallback: treat entire class as single object
                if np.sum(class_mask) >= self.min_bbox_area:
                    # Convert to torch tensor for masks_to_boxes
                    class_tensor = torch.from_numpy(class_mask.astype(bool))
                    
                    # Extract bounding box
                    bbox = masks_to_boxes(class_tensor.unsqueeze(0))  # Add batch dimension
                    
                    if bbox.numel() > 0:  # If bbox was found
                        bbox = bbox.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                        all_bboxes.append(bbox)
                        all_class_ids.append(class_id)
        
        return all_bboxes, all_class_ids
    
    def visualize_sample(self, sample_idx, save_prefix=None):
        """
        Visualize a single sample with segmentation and bounding boxes.
        
        Args:
            sample_idx (int): Index of the sample to visualize
            save_prefix (str, optional): Prefix for saved files
            
        Returns:
            dict: Information about the visualization
        """
        if self.dataset is None or sample_idx >= len(self.dataset):
            raise ValueError(f"Invalid sample index: {sample_idx}")
        
        sample = self.dataset[sample_idx]
        
        # Load image and annotation
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image_np = np.array(image)
            
            segmentation_mask = np.array(Image.open(sample['annotation_path']))
            
        except Exception as e:
            print(f"Error loading sample {sample_idx}: {e}")
            return None
        
        # Extract bounding boxes
        bboxes, class_ids = self.extract_bboxes_from_mask(segmentation_mask)
        
        # Create filename
        if save_prefix is None:
            save_prefix = f"sample_{sample_idx:04d}"
        
        base_filename = os.path.splitext(sample['filename'])[0]
        
        # Visualize and save
        self._save_visualization(
            image_np, segmentation_mask, bboxes, class_ids,
            save_prefix=f"{save_prefix}_{base_filename}"
        )
        
        # Return info
        return {
            'sample_idx': sample_idx,
            'filename': sample['filename'],
            'num_bboxes': len(bboxes),
            'unique_classes': len(set(class_ids)) if class_ids else 0,
            'image_shape': image_np.shape,
            'mask_shape': segmentation_mask.shape
        }
    
    def _save_visualization(self, image, segmentation_mask, bboxes, class_ids, save_prefix):
        """Save visualization with three subplots: original, segmentation, combined."""
        
        # Cityscapes class names and colors
        CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                   'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                   'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                   'bicycle')
        
        PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                   [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                   [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                   [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                   [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # 2. Segmentation mask
        colored_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
        for class_id in range(len(CLASSES)):
            if class_id < len(PALETTE):
                mask = segmentation_mask == class_id
                colored_mask[mask] = PALETTE[class_id]
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask', fontsize=14)
        axes[1].axis('off')
        
        # 3. Combined: image with bounding boxes and segmentation overlay
        axes[2].imshow(image, alpha=0.7)
        axes[2].imshow(colored_mask, alpha=0.4)
        axes[2].set_title(f'Combined (Bboxes: {len(bboxes)})', fontsize=14)
        
        # Draw bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = bbox
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none')
            axes[2].add_patch(rect)
            
            # Add label
            if class_id < len(CLASSES):
                class_name = CLASSES[class_id]
                axes[2].text(x1, y1-5, class_name, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                           fontsize=10, color='white', fontweight='bold')
        
        axes[2].axis('off')
        
        # Save the figure
        save_path = os.path.join(self.output_dir, f"{save_prefix}_visualization.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"Visualization saved: {save_path}")
        
        # Also save individual components
        self._save_individual_images(image, colored_mask, save_prefix)
        
        # Save bounding box info
        self._save_bbox_info(bboxes, class_ids, save_prefix, CLASSES)
    
    def _save_individual_images(self, image, colored_mask, save_prefix):
        """Save individual images separately."""
        # Save original image
        Image.fromarray(image).save(
            os.path.join(self.output_dir, f"{save_prefix}_original.png")
        )
        
        # Save colored segmentation mask
        Image.fromarray(colored_mask).save(
            os.path.join(self.output_dir, f"{save_prefix}_segmentation.png")
        )
    
    def _save_bbox_info(self, bboxes, class_ids, save_prefix, class_names):
        """Save bounding box information to text file."""
        info_path = os.path.join(self.output_dir, f"{save_prefix}_bbox_info.txt")
        
        with open(info_path, 'w') as f:
            f.write(f"Bounding Box Information for {save_prefix}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total bounding boxes: {len(bboxes)}\n")
            f.write(f"Minimum area threshold: {self.min_bbox_area}\n\n")
            
            for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                f.write(f"Bbox {i+1}:\n")
                f.write(f"  Class: {class_name} (ID: {class_id})\n")
                f.write(f"  Coordinates: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
                f.write(f"  Size: {width:.1f} x {height:.1f}\n")
                f.write(f"  Area: {area:.1f}\n\n")
    
    def process_multiple_samples(self, num_samples=5, start_idx=0):
        """Process multiple samples and save visualizations."""
        if self.dataset is None:
            print("Dataset not loaded properly")
            return
        
        max_samples = min(num_samples, len(self.dataset) - start_idx)
        print(f"Processing {max_samples} samples starting from index {start_idx}")
        
        results = []
        
        for i in range(max_samples):
            sample_idx = start_idx + i
            print(f"\nProcessing sample {sample_idx + 1}/{start_idx + max_samples}")
            
            try:
                result = self.visualize_sample(sample_idx)
                if result:
                    results.append(result)
                    print(f"  ✓ Processed: {result['filename']}")
                    print(f"    - Bboxes: {result['num_bboxes']}")
                    print(f"    - Classes: {result['unique_classes']}")
                else:
                    print(f"  ✗ Failed to process sample {sample_idx}")
                    
            except Exception as e:
                print(f"  ✗ Error processing sample {sample_idx}: {e}")
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def _save_summary(self, results):
        """Save summary of all processed samples."""
        summary_path = os.path.join(self.output_dir, "processing_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Cityscapes Bounding Box Extraction Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total samples processed: {len(results)}\n")
            f.write(f"Minimum bbox area: {self.min_bbox_area}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            total_bboxes = sum(r['num_bboxes'] for r in results)
            f.write(f"Total bounding boxes extracted: {total_bboxes}\n")
            f.write(f"Average bboxes per image: {total_bboxes/len(results):.2f}\n\n")
            
            f.write("Per-sample details:\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['filename']}\n")
                f.write(f"   Bboxes: {result['num_bboxes']}, Classes: {result['unique_classes']}\n")
                f.write(f"   Image shape: {result['image_shape']}\n\n")
        
        print(f"\nSummary saved: {summary_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize Cityscapes dataset with bounding boxes')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Cityscapes dataset')
    parser.add_argument('--img_dir', type=str, default='leftImg8bit/val',
                       help='Image directory relative to data_root (default: leftImg8bit/val)')
    parser.add_argument('--ann_dir', type=str, default='gtFine/val',
                       help='Annotation directory relative to data_root (default: gtFine/val)')
    parser.add_argument('--output_dir', type=str, default='./bbox_visualizations',
                       help='Output directory for visualizations (default: ./bbox_visualizations)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process (default: 5)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for sample processing (default: 0)')
    parser.add_argument('--min_bbox_area', type=int, default=500,
                       help='Minimum area for bounding boxes (default: 500)')
    
    args = parser.parse_args()
    
    print("Cityscapes Bounding Box Visualization")
    print("=" * 40)
    print(f"Data root: {args.data_root}")
    print(f"Image dir: {args.img_dir}")
    print(f"Annotation dir: {args.ann_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Samples to process: {args.num_samples}")
    print(f"Starting index: {args.start_idx}")
    print(f"Min bbox area: {args.min_bbox_area}")
    
    # Check if scipy is available for connected components
    try:
        import scipy
        print("✓ SciPy available for connected components analysis")
    except ImportError:
        print("⚠ SciPy not available - using fallback method for instance separation")
    
    # Initialize visualizer
    try:
        visualizer = CityscapesBBoxVisualizer(
            data_root=args.data_root,
            img_dir=args.img_dir,
            ann_dir=args.ann_dir,
            output_dir=args.output_dir,
            min_bbox_area=args.min_bbox_area
        )
        
        # Process samples
        results = visualizer.process_multiple_samples(
            num_samples=args.num_samples,
            start_idx=args.start_idx
        )
        
        print("\n" + "=" * 40)
        print("Processing complete!")
        print(f"Check output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
