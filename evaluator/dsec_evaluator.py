#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# took from yolox/evaluators/coco_evaluator.py

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

from .coco_classes import COCO_CLASSES
from .dsec_det_classes import DSEC_DET_CLASSES
from utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=DSEC_DET_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class DSECEvaluator:
    """
    DSEC AP Evaluation class. From YOLOX COCOEvaluator.
    This class evaluates the model on the DSEC dataset using COCO-style metrics.
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
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        # if half:
        #     model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
        #     model(x)
        #     model = model_trt
        ap_50s = []
        ap_50_95s = []
        for cur_iter, batch in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                input_frame = torch.stack([item["events_vg"] for item in batch])#.to(self.device)
                targets = torch.stack([item["BB"] for item in batch])#.to(self.device)
                img_info = [item["img_metas"] for item in batch]
                #imgs = input_frame.type(tensor_type)
                # skip the last iters since batchsize might be not enough for batch inference
                ap_50, ap_50_95, summary = self.evaluate_single_batch(model, input_frame, targets, img_info)
                if DEBUG>=1:print(summary)
                ap_50s.append(ap_50)
                ap_50_95s.append(ap_50_95)

        return np.mean(ap_50_95s), np.mean(ap_50s), None

    def convert_to_coco_format(self, outputs, targets, images_info, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_info) in zip(
            outputs, images_info
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize, 0 -> height, 1 -> width
            scale = min(
                self.img_size[0] / float(img_info.data['orig_shape'][0]), self.img_size[1] / float(img_info.data['orig_shape'][1])
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # image_wise_data.update({
            #     int(img_id): {
            #         "bboxes": [box.numpy().tolist() for box in bboxes],
            #         "scores": [score.numpy().item() for score in scores],
            #         "categories": [int(cls[ind]) for ind in range(bboxes.shape[0]) # TODO: check if this is correct
            #         ],
            #     }
            # })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind]) # TODO: check if this is correct
                pred_data = {
                    "image_id": int(img_info.data['idx']),  # Need image ID for COCO format
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def create_coco_gt_from_batch(self, targets, images_info):
        """
        Create COCO-style ground truth annotations from batch data.
        
        Args:
            targets: Ground truth bounding boxes and labels
            images_info: Image metadata
            
        Returns:
            dict: COCO-style dataset with images, annotations, and categories
        """
        images = []
        annotations = []
        annotation_id = 1
        
        for idx, (target, img_info) in enumerate(zip(targets, images_info)):
            if target is None:
                continue
                
            # Add image info
            orig_height, orig_width = img_info.data['orig_shape']
            img_id = int(img_info.data['idx'])  # Ensure it's an integer
            
            # Ensure unique image IDs
            if any(img['id'] == img_id for img in images):
                img_id = len(images) + idx + 1
                
            image_data = {
                "id": img_id,
                "width": int(orig_width),
                "height": int(orig_height),
                "file_name": img_info.data['ori_filename']
            }
            images.append(image_data)
            
            # Process target bounding boxes
            target = target.cpu() if hasattr(target, 'cpu') else target
            
            # Assuming target format is [class_id, x1, y1, x2, y2] or similar
            # Adjust this based on your actual target format
            if len(target.shape) == 2:
                for ann_idx in range(target.shape[0]):
                    bbox = target[ann_idx]
                    if len(bbox) >= 5:  # class_id, x1, y1, x2, y2, 
                        class_id, x1, y1, w, h= bbox[:5]

                        # Convert to COCO format (x, y, width, height)
                        coco_bbox = [
                            float(x1), 
                            float(y1), 
                            float(w), 
                            float(h)
                        ]

                        area = float(w * h)
                        # Skip very small bounding boxes
                        if area <= 0:
                            continue
                        
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
        categories = []
        for i in range(self.num_classes):
            categories.append({
                "id": i,
                "name": f"class_{i}",
                "supercategory": "object"
            })
        
        coco_gt = {
            "info": {
                "description": "DSEC Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "DSEC",
                "date_created": "2024-01-01"
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        return coco_gt

    def calculate_coco_metrics(self, predictions, targets, images_info):
        """
        Calculate COCO AP metrics (AP@50:95 and AP@50) from predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            images_info: Image metadata
            
        Returns:
            tuple: (ap50_95, ap50, detailed_info)
        """
        # Convert predictions to COCO format
        pred_data = self.convert_to_coco_format(predictions, targets, images_info)
        
        if len(pred_data) == 0:
            return 0.0, 0.0, "No predictions to evaluate"
        
        # Create COCO ground truth
        coco_gt_data = self.create_coco_gt_from_batch(targets, images_info)
        
        if len(coco_gt_data['annotations']) == 0:
            return 0.0, 0.0, "No ground truth annotations"
        
        try:
            # Create temporary COCO objects
            import tempfile
            import json
            from pycocotools.coco import COCO
            
            if DEBUG >= 2:
                print(f"Creating COCO GT with {len(coco_gt_data['images'])} images and {len(coco_gt_data['annotations'])} annotations")
            
            # Save GT to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
                json.dump(coco_gt_data, gt_file)
                gt_path = gt_file.name
            
            # Save predictions to temporary file  
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
                json.dump(pred_data, pred_file)
                pred_path = pred_file.name
            
            if DEBUG >= 2:
                print(f"Saved GT to {gt_path}, predictions to {pred_path}")
            
            # Load COCO objects
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(pred_path)
            
            # Import COCOeval
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval
                logger.warning("Using standard COCOeval.")
            
            # Evaluate
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            # Get metrics
            # stats[0] = AP@0.5:0.95
            # stats[1] = AP@0.5
            ap50_95 = coco_eval.stats[0]
            ap50 = coco_eval.stats[1]
            
            # Generate summary
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            summary_info = redirect_string.getvalue()
            
            # Clean up temporary files
            import os
            os.unlink(gt_path)
            os.unlink(pred_path)
            
            return float(ap50_95), float(ap50), summary_info
            
        except Exception as e:
            import traceback
            logger.error(f"Error calculating COCO metrics: {e}")
            if DEBUG >= 1:
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"GT data keys: {coco_gt_data.keys() if 'coco_gt_data' in locals() else 'Not created'}")
                logger.error(f"Predictions count: {len(pred_data) if 'pred_data' in locals() else 'Not created'}")
            return 0.0, 0.0, f"Error: {str(e)}"

    def evaluate_single_batch(self, model, input_frame, targets, img_info):
        """
        Evaluate a single batch and return AP@50:95 and AP@50 metrics.
        
        Args:
            model: The detection model
            input_frame: Input tensor for the model
            targets: Ground truth bounding boxes and labels
            img_info: Image metadata
            
        Returns:
            tuple: (ap50_95, ap50, summary_info)
        """
        model.eval()
        with torch.no_grad():
            # Forward pass
            outputs, _ = model(input_frame)
            
            # Post-process outputs
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            
            # Calculate COCO metrics
            ap50_95, ap50, summary = self.calculate_coco_metrics(outputs, targets, img_info)
            
            return ap50_95, ap50, summary

###### TODO: CONVERSIONE
"""
Il problema sta che lui va a caricare la gt da file, quando invece io la do per la singola batch. Quindi primo passo convertire questa cosa.
Ci sarà sicuramente bisogno di modificare ka evaluate prediction t.c. prenda anche le gt dalla batch e non da file.
Inoltre, bisogna modificare la parte di convert_to_coco_format per prendere le gt.

In realtà se converto la funzione evaluate_prediciton in modo che prenda direttamente  
- input_frame = torch.stack([item["events_vg"] for item in imgs]).to(self.device)
- targets = torch.stack([item["BB"] for item in imgs]).to(self.device),
Non si ha bisogno di alcun altro metodo, se non quello per convertire tutto in formato COCO. Vediamo.
"""


# Example usage:
"""
# How to use the new COCO metrics calculation:

# 1. During evaluation loop:
evaluator = DSECEvaluator(dataloader, img_size, confthre, nmsthre, num_classes)

for batch in dataloader:
    input_frame = torch.stack([item["events_vg"] for item in batch])
    targets = torch.stack([item["BB"] for item in batch])
    img_info = [item["img_metas"] for item in batch]
    
    # Method 1: Evaluate single batch
    ap50_95, ap50, summary = evaluator.evaluate_single_batch(model, input_frame, targets, img_info)
    print(f"Batch AP@50:95: {ap50_95:.4f}, AP@50: {ap50:.4f}")
    
    # Method 2: Calculate metrics directly from predictions
    model.eval()
    with torch.no_grad():
        outputs, _ = model(input_frame)
        outputs = postprocess(outputs, num_classes, confthre, nmsthre)
        
    ap50_95, ap50, summary = evaluator.calculate_coco_metrics(outputs, targets, img_info)
    print(f"Direct calculation - AP@50:95: {ap50_95:.4f}, AP@50: {ap50:.4f}")
    print(summary)

# Important notes:
# 1. Make sure your targets are in the correct format: [x1, y1, x2, y2, class_id]
# 2. Make sure img_info contains 'orig_shape' and optionally 'img_id' and 'filename'
# 3. The method handles batch processing automatically
# 4. Results are the same metrics as COCO evaluation (AP@IoU=0.5:0.95 and AP@IoU=0.5)
"""
