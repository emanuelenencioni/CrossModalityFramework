#!/usr/bin/env python3
"""
DSEC Dataset Visualization Script
Visualizes events, images, segmentation labels, and bounding boxes from the DSEC dataset.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
import random
import cv2

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.dsec import DSECDataset

class DSECVisualizer:
    """Visualizer for DSEC dataset with events, images, segmentation, and bounding boxes."""
    
    # DSEC segmentation class names (for semantic segmentation)
    SEG_CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                   'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                   'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                   'bicycle')
    
    # DSEC detection class names (for bounding boxes)
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
    
    # Color palette for segmentation (from the dataset)
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    
    # Colors for bounding boxes (normalized to [0,1] for matplotlib, will convert for CV2)
    BBOX_COLORS_NORM = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 0.0, 1.0],  # blue
        [0.0, 1.0, 0.0],  # green
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # purple
        [1.0, 0.5, 0.0],  # orange
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 0.0, 0.5],  # magenta
    ])
    
    def __init__(self, dataset_txt_path, output_dir="./dsec_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataset with all outputs we want to visualize
        # Remove 'label' from outputs to avoid the flip_flag issue
        self.dataset = DSECDataset(
            dataset_txt_path=dataset_txt_path,
            outputs={'events_vg', 'image', 'BB', 'img_metas'},  # Removed 'label'
            events_bins=5,
            events_clip_range=None,
            test_mode=True  # No data augmentation for visualization
        )
        
        print(f"Initialized DSEC dataset with {len(self.dataset)} samples")
        print(f"Detection classes: {self.DET_CLASSES}")
        print(f"Output directory: {output_dir}")
    
    def events_to_image(self, events_vg):
        """Convert events voxel grid to RGB image for visualization."""
        if events_vg.dim() == 3:
            # Average across time bins or take the mean
            events_img = torch.mean(events_vg, dim=0)
        else:
            events_img = events_vg
        
        # Normalize to [0, 1]
        events_img = (events_img + 1) / 2
        
        # Convert to RGB
        if events_img.dim() == 2:
            events_img = events_img.unsqueeze(0).repeat(3, 1, 1)
        
        # Convert to numpy and transpose for matplotlib
        events_img = events_img.numpy()
        events_img = np.transpose(events_img, (1, 2, 0))
        events_img = np.clip(events_img, 0, 1)
        
        return events_img
    
    def label_to_color(self, label):
        """Convert segmentation label to color image using segmentation classes."""
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        # Create RGB image
        h, w = label.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(self.PALETTE):
            mask = (label == class_id)
            color_img[mask] = color
        
        return color_img
    
    def draw_bounding_boxes(self, ax, bboxes, img_width=640, img_height=440):
        """Draw bounding boxes on the given axis."""
        bbox_count = 0
        
        if len(bboxes) == 0:
            return bbox_count
            
        for i, bbox in enumerate(bboxes):
            # bbox format from DSEC: [class_id, x, y, width, height]
            if len(bbox) != 5:
                continue
                
            class_id, x, y, w, h = bbox
            
            # Convert to float and check for valid values
            try:
                class_id = float(class_id)
                x = float(x)
                y = float(y) 
                w = float(w)
                h = float(h)
            except (ValueError, TypeError):
                continue
            
            # Skip empty bounding boxes (all zeros or invalid)
            if class_id == 0 and x == 0 and y == 0 and w == 0 and h == 0:
                continue
                
            # Skip if any dimension is invalid
            if w <= 0 or h <= 0:
                continue
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = max(1, min(w, img_width - x))
            h = max(1, min(h, img_height - y))
            
            # Draw the bounding box
            color = self.BBOX_COLORS_NORM[bbox_count % len(self.BBOX_COLORS_NORM)]
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add class label using DETECTION classes (not segmentation classes)
            class_id_int = int(class_id)
            if 0 <= class_id_int < len(self.DET_CLASSES):
                class_name = self.DET_CLASSES[class_id_int]
                ax.text(x, max(0, y-5), f'{class_name}', 
                       bbox=dict(facecolor=color, alpha=0.7),
                       fontsize=8, color='white')
            else:
                # Show class ID if not in known detection classes
                ax.text(x, max(0, y-5), f'Det_Class_{class_id_int}', 
                       bbox=dict(facecolor=color, alpha=0.7),
                       fontsize=8, color='white')
            
            bbox_count += 1
        
        return bbox_count
    
    def draw_bbox_on_img(self, img, bboxes, scores=None, conf=0.5, scale=1, linewidth=2, show_conf=True):
        """
        Draw bounding boxes on image using OpenCV.
        
        Args:
            img: numpy array image (H, W, 3) in range [0,1] or [0,255]
            bboxes: tensor/array of bboxes in format [class_id, x, y, w, h]
            scores: confidence scores (optional)
            conf: confidence threshold
            scale: scaling factor
            linewidth: line thickness
            show_conf: whether to show confidence scores
        """
        # Convert image to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        else:
            img = img.copy()
        
        if len(bboxes) == 0:
            return img
            
        for i, bbox in enumerate(bboxes):
            if len(bbox) != 5:
                continue
                
            class_id, x, y, w, h = bbox
            
            # Skip if confidence too low
            if scores is not None and i < len(scores) and scores[i] < conf:
                continue
            
            # Convert to integers
            try:
                x0 = int(scale * x)
                y0 = int(scale * y)
                x1 = int(scale * (x + w))
                y1 = int(scale * (y + h))
                cls_id = int(class_id)
            except (ValueError, TypeError):
                continue
            
            # Skip empty bounding boxes
            if cls_id == 0 and x == 0 and y == 0 and w == 0 and h == 0:
                continue
                
            # Skip if dimensions are invalid
            if x1 <= x0 or y1 <= y0:
                continue
            
            # Get color for this class
            color_idx = cls_id % len(self.BBOX_COLORS_NORM)
            color = (self.BBOX_COLORS_NORM[color_idx] * 255).astype(np.uint8).tolist()
            
            # Create label text
            if 0 <= cls_id < len(self.DET_CLASSES):
                class_name = self.DET_CLASSES[cls_id]
            else:
                class_name = f"Class_{cls_id}"
            
            text = f"{class_name}"
            if scores is not None and i < len(scores) and show_conf:
                text += f":{scores[i] * 100:.1f}%"
            
            # Draw bounding box rectangle
            cv2.rectangle(img, (x0, y0), (x1, y1), color, linewidth)
            
            # Prepare text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            txt_bk_color = [int(c * 0.7) for c in color]  # Darker background
            txt_height = int(1.5 * text_height)
            
            cv2.rectangle(
                img,
                (x0, y0 - txt_height),
                (x0 + text_width + 4, y0 + 2),
                txt_bk_color,
                -1
            )
            
            # Choose text color based on background brightness
            avg_color = np.mean(color)
            txt_color = (0, 0, 0) if avg_color > 127 else (255, 255, 255)
            
            # Draw text
            cv2.putText(
                img, 
                text, 
                (x0 + 2, y0 - txt_height + text_height + 2), 
                font, 
                font_scale, 
                txt_color, 
                thickness
            )
        
        return img
    
    def visualize_sample(self, idx, save_individual=True):
        """Visualize a single sample from the dataset."""
        try:
            sample = self.dataset[idx]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return False
        
        # Get image metadata
        img_metas = sample.get('img_metas', {})
        if hasattr(img_metas, 'data'):
            img_metas = img_metas.data
        
        filename = img_metas.get('ori_filename', f'sample_{idx}')
        print(f"Visualizing sample {idx}: {filename}")
        
        # Prepare images
        images_to_plot = []
        titles = []
        
        # Events
        if 'events_vg' in sample:
            events_img = self.events_to_image(sample['events_vg'])
            images_to_plot.append(events_img)
            titles.append('Events Visualization')
        
        # RGB Image
        if 'image' in sample:
            image = sample['image']
            if isinstance(image, torch.Tensor):
                # Denormalize the image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = torch.clamp(image, 0, 1)
                image = image.numpy().transpose(1, 2, 0)
            images_to_plot.append(image)
            titles.append('RGB Image')
        
        # Segmentation
        if 'label' in sample:
            label_color = self.label_to_color(sample['label'])
            label_color = label_color.astype(np.float32) / 255.0  # Normalize to [0,1]
            images_to_plot.append(label_color)
            titles.append('Segmentation Labels')
        
        # Get bounding boxes
        bboxes = sample.get('BB', torch.tensor([]))
        
        if len(images_to_plot) == 0:
            print(f"No valid data to visualize for sample {idx}")
            return False
        
        # Draw bounding boxes on images (not on segmentation)
        images_with_bboxes = []
        bbox_counts = []
        
        for i, (img, title) in enumerate(zip(images_to_plot, titles)):
            if 'Segmentation' not in title:  # Don't draw bboxes on segmentation
                # Draw bboxes using OpenCV function
                img_with_bbox = self.draw_bbox_on_img(img.copy(), bboxes, scale=1.0)
                
                # Count valid bboxes for title
                bbox_count = 0
                for bbox in bboxes:
                    if len(bbox) == 5:
                        class_id, x, y, w, h = bbox
                        if not (class_id == 0 and x == 0 and y == 0 and w == 0 and h == 0):
                            bbox_count += 1
                
                if bbox_count > 0:
                    title = f"{title} ({bbox_count} boxes)"
                
                bbox_counts.append(bbox_count)
                images_with_bboxes.append(img_with_bbox)
            else:
                images_with_bboxes.append(img)
                bbox_counts.append(0)
        
        # Create figure and plot
        num_plots = len(images_with_bboxes)
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images_with_bboxes, titles)):
            # Update title with bbox count
            if bbox_counts[i] > 0 and 'Segmentation' not in title:
                title = f"{title} ({bbox_counts[i]} boxes)"
            
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_individual:
            output_path = os.path.join(self.output_dir, f'sample_{idx:06d}_{filename}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.close()
        return True
    
    def create_dataset_overview(self, num_samples=9):
        """Create an overview grid of multiple samples."""
        print(f"Creating dataset overview with {num_samples} samples...")
        
        # Select random samples
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        if grid_size == 1:
            axes = [[axes]]
        elif axes.ndim == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            row = i // grid_size
            col = i % grid_size
            
            try:
                sample = self.dataset[idx]
                
                # Prefer events, fallback to image
                if 'events_vg' in sample:
                    img = self.events_to_image(sample['events_vg'])
                    title_prefix = "Events"
                elif 'image' in sample:
                    img = sample['image']
                    if isinstance(img, torch.Tensor):
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img = img * std + mean
                        img = torch.clamp(img, 0, 1).numpy().transpose(1, 2, 0)
                    title_prefix = "Image"
                else:
                    continue
                
                axes[row][col].imshow(img)
                
                # Add bounding boxes
                bboxes = sample.get('BB', torch.tensor([]))
                bbox_count = self.draw_bounding_boxes(axes[row][col], bboxes)
                
                title = f"{title_prefix} {idx}"
                if bbox_count > 0:
                    title += f" ({bbox_count} boxes)"
                
                axes[row][col].set_title(title, fontsize=10)
                axes[row][col].axis('off')
                
            except Exception as e:
                axes[row][col].text(0.5, 0.5, f'Error loading\nsample {idx}', 
                                   ha='center', va='center', transform=axes[row][col].transAxes)
                axes[row][col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row][col].axis('off')
        
        plt.tight_layout()
        overview_path = os.path.join(self.output_dir, 'dataset_overview.png')
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Dataset overview saved: {overview_path}")
    
    def analyze_dataset_statistics(self):
        """Analyze and print dataset statistics."""
        print("Analyzing dataset statistics...")
        
        total_samples = len(self.dataset)
        bbox_counts = []
        class_counts = {}
        
        for i in tqdm(range(min(100, total_samples)), desc="Analyzing samples"):
            try:
                sample = self.dataset[i]
                bboxes = sample.get('BB', torch.tensor([]))
                
                # Count non-empty bounding boxes
                valid_bboxes = 0
                for bbox in bboxes:
                    if len(bbox) != 5:
                        continue
                    class_id, x, y, w, h = bbox
                    if not (class_id == 0 and x == 0 and y == 0 and w == 0 and h == 0):
                        valid_bboxes += 1
                        class_id = int(class_id)
                        # Use detection classes for bbox statistics
                        if 0 <= class_id < len(self.DET_CLASSES):
                            class_name = self.DET_CLASSES[class_id]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                bbox_counts.append(valid_bboxes)
                
            except Exception as e:
                print(f"Error analyzing sample {i}: {e}")
        
        # Print statistics
        print(f"\nDataset Statistics (based on {len(bbox_counts)} samples):")
        print(f"Total samples: {total_samples}")
        print(f"Average bboxes per sample: {np.mean(bbox_counts):.2f}")
        print(f"Max bboxes per sample: {np.max(bbox_counts) if bbox_counts else 0}")
        print(f"Samples with bboxes: {sum(1 for x in bbox_counts if x > 0)}")
        
        if class_counts:
            print(f"\nDetection Class Distribution:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")
        
        print(f"\nAvailable detection classes: {', '.join(self.DET_CLASSES)}")
        print(f"Available segmentation classes: {', '.join(self.SEG_CLASSES)}")
def main():
    parser = argparse.ArgumentParser(description='Visualize DSEC dataset with events, images, segmentation, and bounding boxes')
    parser.add_argument('--dataset_txt', type=str, required=True,
                       help='Path to dataset txt file (e.g., night_dataset_warp.txt)')
    parser.add_argument('--output_dir', type=str, default='./dsec_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--sample_indices', type=int, nargs='+',
                       help='Specific sample indices to visualize')
    parser.add_argument('--create_overview', action='store_true',
                       help='Create dataset overview grid')
    parser.add_argument('--analyze_stats', action='store_true',
                       help='Analyze dataset statistics')
    
    args = parser.parse_args()
    
    # Check if dataset txt file exists
    if not os.path.exists(args.dataset_txt):
        print(f"❌ Dataset txt file not found: {args.dataset_txt}")
        return 1
    
    # Initialize visualizer
    try:
        visualizer = DSECVisualizer(args.dataset_txt, args.output_dir)
    except Exception as e:
        print(f"❌ Error initializing visualizer: {e}")
        return 1
    
    # Analyze statistics if requested
    if args.analyze_stats:
        visualizer.analyze_dataset_statistics()
    
    # Create overview if requested
    if args.create_overview:
        visualizer.create_dataset_overview()
    
    # Visualize specific samples
    if args.sample_indices:
        indices = args.sample_indices
    else:
        # Select random samples
        total_samples = len(visualizer.dataset)
        indices = random.sample(range(total_samples), min(args.num_samples, total_samples))
    
    print(f"Visualizing {len(indices)} samples...")
    success_count = 0
    
    for idx in tqdm(indices, desc="Creating visualizations"):
        if visualizer.visualize_sample(idx):
            success_count += 1
    
    print(f"\n✓ Successfully visualized {success_count}/{len(indices)} samples")
    print(f"Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())