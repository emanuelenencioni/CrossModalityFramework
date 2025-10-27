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
import shutil
import numpy as np
from utils.helpers import DEBUG
import cv2
import wandb
from PIL import Image

from utils import visualization as viz

import torch
from torchvision.ops import nms, batched_nms

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .coco_classes import COCO_CLASSES
from .dsec_det_classes import DSEC_DET_CLASSES
from dataset.dsec import DSECDataset

from utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

ROOT_FOLDER = Path(__file__).resolve().parent.parent

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
        per_class_AR["class/AR/"+name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    if DEBUG >= 1: print(table)
    return per_class_AR


def per_class_AP_table(coco_eval, class_names=DSEC_DET_CLASSES, headers=["class", "AP"], colums=6):
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
        per_class_AP["class/AP/"+name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    if DEBUG >= 1: print(table)
    return per_class_AP

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
        device = "cpu",
        input_format = "xywh",  # or "xyxy"
        input_type = None
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
        self.class_names = DSECDataset.DETECTION_CLASSES
        self.dataloader = dataloader
        self.img_size = img_size
        self.conf_thre = confthre
        self.nms_thre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.device = device
        self.input_format = input_format
        if input_type is not None:
            self.input_type = input_type
        else:
            self.input_type = 'events' if 'events' in dataloader.dataset[0] else 'image'
        self.step = 0
        if DEBUG >= 3:
            self.debug_images_folder = ROOT_FOLDER / "debug_images"
            if self.debug_images_folder.exists():
                shutil.rmtree(self.debug_images_folder)
            self.debug_images_folder.mkdir(exist_ok=True, parents=True)
        else: self.debug_images_folder = None
        

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
        progress_bar = tqdm if is_main_process() else iter
        
        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        random_idx = torch.randint(1, len(self.dataloader) + 1, (1,)).item()
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
        #     model(x)
        #     model = model_trt
        outputs, gts, imgs_info = [], [], []

        for cur_iter, batch in enumerate(
            progress_bar(self.dataloader)):
            with torch.no_grad():
                input_frame = torch.stack([item[self.input_type] for item in batch]).to(self.device)
                assert "BB" in batch[0], "Batch must contain 'BB' key for targets"
                targets = torch.stack([item["BB"] for item in batch]) #TODO: now only for obj detection
                img_info = [item["img_metas"] for item in batch]
                #imgs = input_frame.type(tensor_type)
                # skip the last iters since batchsize might be not enough for batch inference
                out_dict = model(input_frame)
                output = out_dict['head_outputs']
                
                gts.append(targets.cpu())
                if DEBUG >= 3 and cur_iter == 0: #only for the first batch
                    imgs_info.append((img_info, input_frame))
                else:
                    imgs_info.append((img_info, None))
                outputs.append(self.postprocess(output, images_info=imgs_info[-1] if DEBUG >= 3 else None))
                if cur_iter == random_idx:
                    if wandb.run is not None and output is not None:
                        if DEBUG >= 1:
                            logger.info(f"Example image idx {img_info[0].data['idx']} processed, gt boxes {targets[0].shape[0]}, pred boxes {outputs[-1][0].shape[0] if outputs[-1][0] is not None else 0}")
                        img = self.visual(viz.tensor_to_cv2_image(input_frame[0]),outputs[-1][0], img_info[0].data['orig_shape'], cls_conf=self.conf_thre, classes=self.dataloader.dataset.DSEC_DET_CLASSES)
                        # wandb expects images in RGB format. cv2 uses BGR.
                        wandb.log({"eval/sample_img_with_bb": wandb.Image(Image.fromarray(cv2.cvtColor(img.get(), cv2.COLOR_BGR2RGB)), caption=f"idx_{img_info[0].data['idx']}")})
                    self.step += 1
        #Evaluate all the test set at once
        return self.calculate_coco_metrics(outputs, gts, imgs_info)

    def postprocess(self, prediction, class_agnostic=False, images_info=None):
        box_corner = prediction.new(prediction.shape)
        ####### WARNING : Convert from (cx, cy, w, h) to (x1, y1, x2, y2) format
        logger.warning("Converting from (cx, cy, w, h) to (x1, y1, x2, y2) format, BE AWARE: OLD CODE has cx,cy,h,w")
        
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    self.nms_thre,
                )
            else:
                nms_out_index = batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    self.nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

            if DEBUG >= 3 and self.debug_images_folder is not None and images_info[1] is not None:
                img, img_info = images_info[1][i], images_info[0][i]
                vis_img = self.visual(viz.tensor_to_cv2_image(img), output[i], img_info.data['orig_shape'], cls_conf=self.conf_thre)
                cv2.imwrite(str(self.debug_images_folder / f"{img_info.data['idx']}.png"), vis_img)
        return output

    def visual(self,img, output, orig_size, cls_conf=0.35, classes=None):
        
        if output is None:
            return img
        output = output.cpu()

        bboxes = output #rescale_boxes(output[:, :4], orig_size)
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = self.dataloader.dataset.vis(img, bboxes, scores, cls, cls_conf)
        return vis_res

    def convert_to_coco_format(self, output_batch, images_info):
        data_list = []
        
        image_wise_data = defaultdict(dict)
        for (output, img_info) in zip(
            output_batch, images_info[0]
        ):
            if output is None:
                continue

            output = output.cpu()
            bboxes = output[:, 0:4]
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            

            for ind in range(bboxes.shape[0]):
                area = float((bboxes[ind][2]-bboxes[ind][0])*(bboxes[ind][3]-bboxes[ind][1]))
                if area < 0: continue
                label = int(cls[ind]) # TODO: check if this is correct
                pred_data = {
                    "image_id": int(img_info.data['idx']),  # Need image ID for COCO format
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "area": area,
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        return data_list

    def create_coco_gt_from_batch(self, targets_batch, images_info):
        """
        Create COCO-style ground truth annotations from batch data
        Args:
            targets: Ground truth bounding boxes and labels
            images_info: Image metadata
        Returns: dict: COCO-style dataset with images, annotations, and categories
        """
        images = []
        annotations = []
        annotation_id = 1

        for targets, image_info in zip(targets_batch, images_info):
            # Add image info
            orig_height, orig_width = image_info.data['orig_shape']  # TODO only for testing it's ok
            img_id = int(image_info.data['idx'])  # Ensure it's an integer
                
            image_data = {
                "id": img_id,
                "width": int(orig_width),
                "height": int(orig_height),
                "file_name": image_info.data['ori_filename']
            }
            images.append(image_data)
            # Process target bounding boxes
            
            # Assuming target format is [class_id, x1, y1, x2, y2] or similar
            # Adjust this based on your actual target format
            for target in targets:
                if target is None:
                    continue
                target = target.cpu() if hasattr(target, 'cpu') else target
                if len(target) >= 5:  # convert to class_id, x1, y1, x2, y2
                    if self.input_format == "xyxy":
                        class_id, x1, y1, w, y2 = target[:5]
                        w = x2 - x1
                        h = y2 - y1
                    elif self.input_format == "cxcywh":   
                        class_id, xc, yc, w, h= target[:5]
                        x1 = xc - w * 0.5
                        y1 = yc - h * 0.5
                    elif self.input_format == "xywh":   
                        class_id, x1, y1, w, h= target[:5]
                        x2 = x1 + w
                        y2 = y1 + h

                    # Convert to COCO format (x1, y1, x2, y2)
                    if class_id < 0: continue
                    coco_bbox = [
                        float(x1), 
                        float(y1), 
                        float(x2), 
                        float(y2)
                    ]

                    area = float(w * h)
                    # Skip very small bounding boxes
                    if area <= 0: continue
                    
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
                "name": f"{self.class_names[i]}",
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
            "categories": categories,
            "annotations": annotations,
        }
        
        return coco_gt

    def calculate_coco_metrics(self, predictions, targets, images_info):
        """
        Calculate COCO AP metrics (AP@50:95 and AP@50) from predictions and targets.
        """
        coco_eval = None

        # Accumulate ALL predictions and ground truth data
        all_pred_data = []
        all_images = []
        all_annotations = []
        annotation_id = 1

        for idx, (pred_batch, t_batch, img_info) in enumerate(zip(predictions, targets, images_info)):
            # Convert predictions to COCO format
            batch_pred_data = self.convert_to_coco_format(pred_batch, img_info)
            all_pred_data.extend(batch_pred_data)
            
            # Create ground truth for this batch
            batch_gt_data = self.create_coco_gt_from_batch(t_batch, img_info[0])
            
            # Accumulate images and annotations
            all_images.extend(batch_gt_data['images'])
            
            # Re-index annotations to ensure unique IDs
            for ann in batch_gt_data['annotations']:
                ann['id'] = annotation_id
                all_annotations.append(ann)
                annotation_id += 1
    
        # Create single COCO GT dataset with ALL data
        coco_gt_data = {
            "info": {
                "description": "DSEC Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "DSEC",
                "date_created": "2024-01-01"
            },
            "licenses": [],
            "images": all_images,
            "categories": [
                {"id": i, "name": self.class_names[i], "supercategory": "object"}
                for i in range(self.num_classes)
            ],
            "annotations": all_annotations,
        }
        
        if len(all_pred_data) == 0:
            logger.warning("No predictions across entire dataset")
            return [0.0]*12, None, None
        
        if len(all_annotations) == 0:
            logger.warning("No ground truth annotations")
            return [0.0]*12, None, None

        try:
            if DEBUG >= 2:
                logger.info(f"Creating COCO GT with {len(all_images)} images and {len(all_annotations)} annotations")
                logger.info(f"Total predictions: {len(all_pred_data)}")
            
            # Create COCO objects once with all data
            if DEBUG >= 3:
                coco_gt = COCO()
                coco_gt.dataset = coco_gt_data
                coco_gt.createIndex()
                coco_dt = coco_gt.loadRes(all_pred_data)
            else:
                with contextlib.redirect_stdout(open('/dev/null', 'w')):
                    coco_gt = COCO()
                    coco_gt.dataset = coco_gt_data
                    coco_gt.createIndex()
                    coco_dt = coco_gt.loadRes(all_pred_data)
            
            # Evaluate once on all data
            if DEBUG >= 3:
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            else:
                with contextlib.redirect_stdout(open('/dev/null', 'w')):
                    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()

        except Exception as e:
            import traceback
            logger.error(f"Error calculating COCO metrics: {e}")
            if DEBUG >= 1:
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                logger.warning(f"GT data images: {len(all_images)}")
                logger.warning(f"GT data annotations: {len(all_annotations)}")
                logger.warning(f"Predictions count: {len(all_pred_data)}")
            return [0.0]*12, None, None
        
        # Log metrics to wandb once
        if wandb.run is not None and coco_eval is not None:
            wandb.log({
                "eval/AP_IoU=0.50:0.95": coco_eval.stats[0],
                "eval/AP_IoU=0.50": coco_eval.stats[1],
                "eval/AP_IoU=0.75": coco_eval.stats[2],
                "eval/AP_small": coco_eval.stats[3],
                "eval/AP_medium": coco_eval.stats[4],
                "eval/AP_large": coco_eval.stats[5],
                "eval/AR_maxDets=1": coco_eval.stats[6],
                "eval/AR_maxDets=10": coco_eval.stats[7],
                "eval/AR_maxDets=100": coco_eval.stats[8],
                "eval/AR_small": coco_eval.stats[9],
                "eval/AR_medium": coco_eval.stats[10],
                "eval/AR_large": coco_eval.stats[11],
                "epoch": self.step  # Changed from test_batch to epoch
            })
        
        classAP = per_class_AP_table(coco_eval, class_names=DSEC_DET_CLASSES) if self.per_class_AP and coco_eval is not None else None
        classAR = per_class_AR_table(coco_eval, class_names=DSEC_DET_CLASSES) if self.per_class_AR and coco_eval is not None else None
        
        return coco_eval.stats if coco_eval is not None else [0.0]*12, classAP, classAR