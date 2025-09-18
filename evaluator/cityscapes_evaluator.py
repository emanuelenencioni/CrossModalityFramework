#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import numpy as np
from helpers import DEBUG
import torch

from .dsec_evaluator import DSECEvaluator  # Inherit from DSECEvaluator
from dataset.cityscapes import CityscapesDataset

class CityscapesEvaluator(DSECEvaluator):
    """
    Cityscapes AP Evaluation class.
    This class evaluates the model on the Cityscapes dataset using COCO-style metrics.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
        device = "cpu",
        input_format = "cxcywh"  # or "xyxy"
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        super().__init__(dataloader, img_size, confthre, nmsthre, num_classes, 
                         testdev, per_class_AP, per_class_AR, device, input_format)
        
        # Override class names for Cityscapes
        self.class_names = self._get_cityscapes_class_names()
        
    def _get_cityscapes_class_names(self):
        """Get Cityscapes detection class names"""
        if hasattr(self.dataloader.dataset, 'DSEC_DET_CLASSES'):
            # Use detection classes if available
            det_classes = self.dataloader.dataset.DSEC_DET_CLASSES
            return [det_classes[i] for i in [key  for key in det_classes.keys() if isinstance(key, str)]]
        elif hasattr(self.dataloader.dataset, 'DETECTION_CLASSES'):
            # Fallback to segmentation classes
            return list(self.dataloader.dataset.DETECTION_CLASSES)
        else:
            # Generic class names
            return [f"class_{i}" for i in range(self.num_classes)]

    def create_coco_gt_from_batch(self, targets_list, images_info):
        """
        Create COCO-style ground truth annotations from batch data for Cityscapes.
        
        Args:
            targets: Ground truth bounding boxes and labels
            images_info: Image metadata
            
        Returns:
            dict: COCO-style dataset with images, annotations, and categories
        """
        images = []
        annotations = []
        annotation_id = 1

        for targets,img_info, img in zip(targets_list,images_info[0], images_info[1]):
            # Add image info
            orig_height, orig_width = img_info.data['orig_shape'] if hasattr(img_info, 'data') else (1024, 2048)
            img_id = int(img_info.data['idx'])
            
            # Ensure unique image IDs
            # while any(img['id'] == img_id for img in images):
            #     img_id += 1000
                
            image_data = {
                "id": img_id,
                "width": int(orig_width),
                "height": int(orig_height),
                "file_name": img_info.data.get('ori_filename')
            }
            images.append(image_data)
            boxes = []
            ids = []
            for target in targets:
                if target is None:
                    continue
                
                # Process target bounding boxes
                target = target.cpu() if hasattr(target, 'cpu') else target
                
                # Assuming target format is [class_id, x1, y1, w, h]
                class_id, x_center, y_center, w, h = target[:5]
                if class_id < 0 or w <= 0 or h <= 0:
                    continue
                # Convert from center format to corner format
                x1 = x_center - w * 0.5
                y1 = y_center - h * 0.5
                x2 = x_center + w * 0.5
                y2 = y_center + h * 0.5
                # Skip invalid bboxes
                
                # Convert to COCO format (x, y, width, height)
                coco_bbox = [float(x1), float(y1), float(x2), float(y2)]
                area = float(w * h)
                
                # Skip very small bounding boxes
                if area <= 1:
                    continue
                boxes.append(coco_bbox)
                ids.append(int(class_id))
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                }
                annotations.append(annotation)
                annotation_id += 1
            
            # Create categories based on Cityscapes detection classes
            categories = []
            for i in range(self.num_classes):
                if i < len(self.class_names):
                    cat_name = self.class_names[i]
                else:
                    cat_name = f"class_{i}"
                    
                categories.append({
                    "id": i,
                    "name": cat_name,
                    "supercategory": "object"
                })
            if DEBUG >= 3:
                import cv2
                self.dataloader.dataset.vis(img, boxes, [1] * len(boxes), cls_ids=ids, conf=0.5, class_names=self.dataloader.dataset.DSEC_DET_CLASSES)
                cv2.imwrite(f"debug_gt_{img_id}.jpg", img)
            
        coco_gt = {
            "info": {
                "description": "Cityscapes Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Cityscapes",
                "date_created": "2024-01-01"
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        return coco_gt

        