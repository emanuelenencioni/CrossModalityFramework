# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging
# Took from CMDA - https://github.com/CMDA/CMDA

from ctypes.wintypes import RGB
import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
import numpy as np
from prettytable import PrettyTable
import json
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import Compose
from torchvision.ops import masks_to_boxes
import torchvision.transforms as standard_transforms
from .utils.data_container import DataContainer
from .utils import visualization as visual

# Import albumentations
try:
    import albumentations as A
except ImportError:
    A = None

#from .builder import DATASETS
from utils.helpers import DEBUG
from utils import parse, boxes



class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure is as followed.
    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir=None,
                 events_dir=None,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 load_bboxes=False,
                 bbox_min_area=100,
                 use_augmentations=False, **kwargs):
        #self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.events_dir = events_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.image_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*self.mean_std)])
        self.event_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*self.mean_std)])
        self.split = split
        self.outputs = kwargs.get("outputs", ["rgb"])
        self.event_keys = ["events", "events_vg", "events_frames"] # possible event keys
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.custom_classes = kwargs.get('custom_classes', False)
        if self.custom_classes: 
            self.DETECTION_CLASSES = kwargs.get('DETECTION_CLASSES', None)
            assert self.DETECTION_CLASSES is not None, "DETECTION_CLASSES must be provided when custom_classes is True"
        else:
            assert kwargs.get('bb_num_classes', None) is None, "bb_num_classes should not be set if custom_classes is False"
        # Bounding box support
        self.load_bboxes = load_bboxes
        self.bbox_min_area = bbox_min_area
        self.max_labels = kwargs.get('max_labels', 100)  # Maximum number of bounding boxes per image

        # --- Data Augmentation ---
        self.use_augmentations = use_augmentations
        self.augmentations = None
        if self.use_augmentations and not self.test_mode:
            if A is None:
                raise ImportError("Please install albumentations: pip install albumentations")
            # Define a standard augmentation pipeline for object detection
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(p=0.5),
                # This crop is safe for bounding boxes, it tries to keep them in the frame
                A.RandomSizedBBoxSafeCrop(width=2048, height=1024, erosion_rate=0.2, p=0.5),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels'], min_visibility=0.3))
            print("✓ Albumentations pipeline for training enabled.")
            self.event_augmentations = self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                # This crop is safe for bounding boxes, it tries to keep them in the frame
                A.RandomSizedBBoxSafeCrop(width=2048, height=1024, erosion_rate=0.2, p=0.5),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels'], min_visibility=0.3))
        
        # join paths if data_root is specified
        if self.data_root is not None and self.img_dir is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
        elif self.data_root is not None:
            if not osp.isabs(self.events_dir):
                self.events_dir = osp.join(self.data_root, self.events_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        if self.img_dir is not None:
            self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

        if self.events_dir is not None:
            self.events_infos = self.load_annotations(self.events_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)
        self._COLORS = boxes._COLORS

        self.debug_gt = True

        self.orig_height, self.orig_width = None, None

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) if self.img_dir is not None else len(self.events_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns: list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip().split("/")[-1]
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(subfolder= img_name.split("_")[0], seg_map=seg_map)  # Add subfolder info
                    img_infos.append(img_info)
        else:
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.endswith(img_suffix):
                    img_info = dict(filename=entry.name)
                    if ann_dir is not None:
                        seg_map = entry.name.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(subfolder= img_name.split("_")[0], seg_map=seg_map)  # Add subfolder info
                    img_infos.append(img_info)

        if DEBUG>=1: print(
            f'Loaded {len(img_infos)} images from {img_dir}')
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args: idx (int): Index of data.
        Returns: dict: Annotation info of specified index.
        """
        return self.img_infos[idx]['ann'] if self.img_dir is not None else self.events_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        
        # Check if augmented data is provided
        # TODO: switch ifs (use aug internal to the others)
        if self.use_augmentations and not self.test_mode: # and 'image' in results and 'BB' in results:
            if 'image' in self.outputs:
                image_pil = results['image']
                padded_resized_image, _, padding_info = self.pad_and_resize_pil_image(image_pil, target_size=(512, 512))
                results['image'] = self.image_transform(padded_resized_image)
            
            if any(event_key in self.outputs for event_key in self.event_keys) and self.events_dir is not None:
                event_pil = results['events']
                padded_resized_events, _, padding_info = self.pad_and_resize_pil_image(event_pil, target_size=(512, 512))
                results['events'] = self.event_transform(padded_resized_events)

            if 'BB' in self.outputs:
                bboxes = results['BB']
                transformed_bboxes = self.transform_bboxes_with_padding(bboxes, padding_info)
                results['BB'] = transformed_bboxes
            
            results['padding_info'] = padding_info
            
        # Add bounding box information if available (original logic for test mode or no augmentations)
        else: 
            idx = results.get('idx', 0)
            transformed_bboxes, _ = self.get_padded_and_scaled_bbox_info(idx, target_size=(512, 512))
            if 'BB' in self.outputs: results['BB'] = transformed_bboxes
            # Load image with padding and scaling to 512x512
            if 'image' in self.outputs:
                padded_resized_image, scale_factor, padding_info = self.load_and_resize_image(idx, target_size=(512, 512))
            
                if padded_resized_image is not None:
                    # Convert PIL image to numpy array and normalize to [0, 1]
                    results['padding_info'] = padding_info
                    if 'image' in self.outputs: 
                        results['image'] = self.image_transform(padded_resized_image)
                    
                    # Get transformed bounding boxe
                    
                    if DEBUG >= 2:
                        orig_size = padding_info.get('original_size', (0, 0))
                        print(f"Loaded image: {orig_size} -> padded to square -> resized to (512, 512) "
                            f"with {len(transformed_bboxes)} transformed bboxes")
                else:
                    # Fallback: create empty image
                    results['image'] = torch.zeros((3, 512, 512), dtype=torch.float32)
                    results['padding_info'] = {}
                    results['BB'] = torch.zeros((self.max_labels, 5), dtype=torch.float32)
            
            if any(event_key in self.outputs for event_key in self.event_keys) and self.events_dir is not None:
                padded_resized_events, _, _ = self.load_and_resize_events(idx, target_size=(512, 512))
                if padded_resized_events is not None:
                    results['events'] = self.image_transform(padded_resized_events)
                else:
                    # Fallback: create empty events image
                    results['events'] = torch.zeros((3, 512, 512), dtype=torch.float32)
                    results['padding_info'] = {}
                    results['BB'] = torch.zeros((self.max_labels, 5), dtype=torch.float32)
        
        if self.custom_classes:
            results['label_map'] = self.DETECTION_CLASSES

        # Add img_metas similar to DSEC dataset
        idx = results.get('idx', 0)
        info = self.img_infos[idx] if self.img_dir is not None else self.events_infos[idx]
        
        # Create img_metas dictionary
        img_metas = dict()
        img_metas['img_norm_cfg'] = dict()
        img_metas['img_norm_cfg']['mean'] = [123.675, 116.28, 103.53]
        img_metas['img_norm_cfg']['std'] = [58.395, 57.12, 57.375]
        img_metas['img_norm_cfg']['to_rgb'] = True
        
        # Get original image dimensions
        padding_info = results.get('padding_info', {})
        if padding_info and 'original_size' in padding_info:
            self.orig_width, self.orig_height = padding_info['original_size']
        elif self.orig_height is None and self.orig_width is None:
            # Fallback to default or try to get from image file
            self.orig_width, self.orig_height = 2048, 1024  # Default for Cityscapes
            
            # Try to get actual size from image
            path = self.img_dir if self.img_dir is not None else self.events_dir
            path = osp.join(path, info['filename'])
            if osp.exists(path):
                try:
                    from PIL import Image
                    with Image.open(path) as pil_img:
                        self.orig_width, self.orig_height = pil_img.size
                except Exception:
                    pass
        
        img_metas['img_shape'] = (512, 512)  # Current processed size
        img_metas['pad_shape'] = (512, 512)
        img_metas['ori_shape'] = (512, 512)  # Size after initial processing
        img_metas['orig_shape'] = (self.orig_width, self.orig_height)  # Original image size
        img_metas['ori_filename'] = info['filename']
        img_metas['idx'] = idx
        
        img_metas['flip'] = False
        if img_metas['flip']:
            img_metas['flip_direction'] = 'horizontal'

        if DEBUG>=3:
            if self.debug_gt:
                aug = ""
                if self.use_augmentations: aug = "augmented"
                else: aug = "original"
                if 'image' in self.outputs:
                    rgb = results['image']
                    cvimg = self.gt_to_vis(rgb, results)
                    cv2.imwrite(f"debug_{aug}_{idx}_{info['filename']}.jpg", cvimg)
                if 'events' in self.outputs:     
                    rgb = results['events']              
                    cvimg = self.gt_to_vis(rgb, results)
                    cv2.imwrite(f"debug_{aug}_events_{idx}_{info['filename']}.jpg", cvimg)
                if DEBUG.value in [3, 4]: self.debug_gt = False # Only once if DEBUG isn't that high
        
        # Wrap in DataContainer similar to DSEC
        results['img_metas'] = DataContainer(img_metas, cpu_only=True)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args: idx (int): Index of data.
        Returns: dict: Training/test data (with annotation if `test_mode` is set False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args: idx (int): Index of data.
        Returns: dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        img_info, event_info = None, None
        if 'image' in self.outputs:
            img_info = self.img_infos[idx]
        if any(event_key in self.outputs for event_key in self.event_keys) and self.events_dir is not None:
            event_info = self.events_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, event_info=event_info, ann_info=ann_info, idx=idx)
        if 'BB' in self.outputs:
            bboxes_cxcywh, _ = self.extract_bboxes_from_json_polygons(idx)
                # 2. Convert bboxes to albumentations format (pascal_voc)
            
        # --- Apply Augmentations if enabled ---
        if self.use_augmentations and self.augmentations is not None:
            # 1. Extract bboxes and labels for albumentations
            bboxes_pascal_voc = []
            bbox_labels = []
            valid_mask = bboxes_cxcywh[:, 0] >= 0
            for bbox in bboxes_cxcywh[valid_mask]:
                class_id, xc, yc, w, h = bbox
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                bboxes_pascal_voc.append([x1, y1, x2, y2])
                bbox_labels.append(class_id)

            # 3. Apply augmentations
            try:
                if "image" in self.outputs:
                    img_path = osp.join(self.img_dir, img_info['ann']['subfolder'], img_info['filename'])
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    transformed = self.augmentations(image=image, bboxes=bboxes_pascal_voc, bbox_labels=bbox_labels)
                    image = transformed['image'] # Use augmented image
                    bboxes_pascal_voc = transformed['bboxes']
                    bbox_labels = transformed['bbox_labels']
                    results['image'] = Image.fromarray(image)
                if any(event_key in self.outputs for event_key in self.event_keys) and self.events_dir is not None:
                    events_path = osp.join(self.events_dir, event_info['ann']['subfolder'], event_info['filename'])
                    events_image = cv2.imread(events_path)
                    events_image = cv2.cvtColor(events_image, cv2.COLOR_BGR2RGB)
                    transformed_events = self.event_augmentations(image=events_image, bboxes=bboxes_pascal_voc, bbox_labels=bbox_labels)
                    events_image = transformed_events['image'] # Use augmented events image
                    results['events'] = Image.fromarray(events_image)
                    #Use always rgb in case of both events and rgb are present
                    bboxes_pascal_voc = transformed_events['bboxes'] if not 'image' in self.outputs else bboxes_pascal_voc
                    bbox_labels = transformed_events['bbox_labels'] if not 'image' in self.outputs else bbox_labels
                    results['events'] = Image.fromarray(events_image) 
            except Exception as e:
                if DEBUG >= 1: print(f"Warning: Albumentations failed for index {idx}, using original image. Error: {e}")

            # 4. Convert augmented bboxes back to cxcywh format for the rest of the pipeline
            final_bboxes = torch.full((self.max_labels, 5), -1, dtype=torch.float32)
            for i, (bbox, label) in enumerate(zip(bboxes_pascal_voc, bbox_labels)):
                if i >= self.max_labels: break
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                xc, yc = x1 + w * 0.5, y1 + h * 0.5
                final_bboxes[i] = torch.tensor([label, xc, yc, w, h])
            
            # 5. Pass augmented data to the pre_pipeline

            if "BB" in self.outputs: results['BB'] = final_bboxes

        else:
            if "BB" in self.outputs: results['BB'] = bboxes_cxcywh

        self.pre_pipeline(results)
        return results#self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args: idx (int): Index of data.
        Returns: dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info, idx=idx)
        self.pre_pipeline(results)
        return results#self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = np.array(Image.open(seg_map))
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def extract_bboxes_from_json_polygons(self, idx):
        """
        Extract bounding boxes from JSON polygon annotations.
        Each polygon represents an individual instance of a class.
        Args: idx (int): Index of the sample
        Returns:
            tuple: (bboxes, instance_masks) where:
                - bboxes are [class_id, x, y, w, h] format
                - instance_masks are individual binary masks for each instance
        """
        if self.ann_dir is None: return [], [], []
            
        # Get JSON file path
        info = self.img_infos[idx] if self.img_dir is not None else self.events_infos[idx]
        seg_map_path = osp.join(self.ann_dir, info['ann']['subfolder'], info['ann']['seg_map'])
        json_file_path = seg_map_path.replace(self.seg_map_suffix, '_gtFine_polygons.json')
        
        if not osp.exists(json_file_path):
            if DEBUG >= 1: print(f"JSON annotation file not found: {json_file_path}")
            return [], [], []
        all_bboxes = torch.zeros((self.max_labels, 5), dtype=torch.float32)
        all_bboxes[:, 0] = -1  # Initialize with -1 for empty cells
        try:
            # Load JSON annotation
            f = open(json_file_path, 'r')
            annotation_data = json.load(f)
            
            img_height = annotation_data.get('imgHeight', 1024)
            img_width = annotation_data.get('imgWidth', 2048)
            objects = annotation_data.get('objects', [])

            # Initialize tensor for bounding boxes [max_labels, 5] where 5 = [class_id, x, y, w, h]

            all_instance_masks = []
            bbox_count = 0
            
            for obj_idx, obj in enumerate(objects):
                label = obj.get('label', '')
                polygon = obj.get('polygon', [])
                
                if not polygon or len(polygon) < 3: continue
                # Map label to class index
                class_id = -1
                if self.DETECTION_CLASSES is not None and label in self.DETECTION_CLASSES.keys():
                    class_id = self.DETECTION_CLASSES[label]

                # Skip if class not found or is ignore class
                if class_id == -1 or class_id == self.ignore_index or class_id not in self.DETECTION_CLASSES.values(): 
                    continue
                # Create binary mask from polygon
                try:
                    # Convert polygon to binary mask
                    mask = Image.new('L', (img_width, img_height), 0)
                    if polygon:
                        # Flatten polygon points for PIL
                        polygon_points = [tuple(point) for point in polygon]
                        ImageDraw.Draw(mask).polygon(polygon_points, outline=1, fill=1)
                    
                    # Convert to numpy array
                    mask_array = np.array(mask, dtype=bool)
                    
                    # Check minimum area
                    if np.sum(mask_array) < self.bbox_min_area: continue
                    
                    # Convert to torch tensor for masks_to_boxes
                    mask_tensor = torch.from_numpy(mask_array)
                    
                    # Extract bounding box using torchvision.ops.masks_to_boxes
                    bbox = masks_to_boxes(mask_tensor.unsqueeze(0))  # Add batch dimension
                    
                    if bbox.numel() > 0:  # If bbox was found
                        bbox = bbox.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                        
                        # Validate bbox coordinates
                        x1, y1, x2, y2 = bbox
                        if x2 > x1 and y2 > y1:  # Valid bboxtensor_to_cv2_image
                            # Convert to [class_id, xc, yc, w, h] format
                            h = y2 - y1
                            w = x2 - x1
                            bbox_formatted = torch.tensor([class_id, x1+w*0.5, y1+h*0.5, w, h], dtype=torch.float32)

                            # Check if we exceed max_labels
                            if bbox_count >= self.max_labels:
                                if DEBUG >= 1:
                                    print(f"Warning: Exceeded max_labels ({self.max_labels}) for JSON polygon extraction")
                                break
                            
                            all_bboxes[bbox_count] = bbox_formatted
                            all_instance_masks.append(mask_array)
                            bbox_count += 1
                    elif DEBUG >= 1: print(f"No valid bbox found for object {obj_idx} in {json_file_path}")
                    
                except Exception as e:
                    if DEBUG >= 1: print(f"Error processing polygon for object {obj_idx} in {json_file_path}: {e}")
                    continue
                
                # Break outer loop if max_labels exceeded
                if bbox_count >= self.max_labels:
                    break
            
            # Always return tensor with max_labels dimension, filled cells have class_id > 0, empty cells have class_id = -1
            # Initialize empty cells with class_id = -
            return all_bboxes, all_instance_masks
            
        except Exception as e:
            if DEBUG >= 1: print(f"Error loading JSON file {json_file_path}: {e}")
            return all_bboxes, []

    def get_bbox_info(self, idx):
        """
        Get bounding box information for a sample.
        Priority: JSON polygons > segmentation masks > empty tensor
        Args: idx (int): Index of the sample
        Returns: torch.Tensor: Tensor of bounding boxes in [max_labels, 5] format where 5 = [class_id, x_center, y_center, w, h]
                              Empty cells have class_id = -1
        """
        bboxes, _ = self.extract_bboxes_from_json_polygons(idx)
        
        # If no valid bboxes found, check if we need to return a tensor with all -1s
        if torch.sum(bboxes[:, 0] > 0) == 0:
            # Initialize empty tensor with class_id = -1
            empty_bboxes = torch.full((self.max_labels, 5), -1, dtype=torch.float32)
            return empty_bboxes
        
        return bboxes

    def draw_bounding_boxes_on_image(self, image, bboxes, colors=None):
        """
        Draw bounding boxes on image using torchvision.utils.draw_bounding_boxes.
        Args:
            image (np.ndarray or torch.Tensor): Input image
            bboxes (list or torch.Tensor): Bounding boxes in [class_id, x_center, y_center, w, h] format
            colors (list, optional): Colors for each bbox
        Returns: torch.Tensor: Image with drawn bounding boxes
        """
        if len(bboxes) == 0:
            if torch.is_tensor(image): return image
            else:
                return torch.from_numpy(image).permute(2, 0, 1).byte()
        
        # Convert image to tensor
        if not torch.is_tensor(image):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).byte()
        else:
            image_tensor = image.byte()
        
        # Convert bboxes from [class_id, x_center, y_center, w, h] to [x1, y1, x2, y2] and extract labels
        # Filter out empty bboxes (class_id = -1)
        bbox_list = []
        labels = []
        
        for bbox in bboxes:
            class_id, x_center, y_center, w, h = bbox
            
            # Skip empty bboxes
            if class_id < 0:
                continue
            x1, y1 = x_center-w*0.5, y_center-h*0.5
            x2, y2 = x1 + w, y1 + h
            bbox_list.append([x1, y1, x2, y2])
            
            # Create label
            if self.DETECTION_CLASSES is not None and class_id in self.DETECTION_CLASSES.values():
                labels.append([key for key, value in self.DETECTION_CLASSES.items() if value == class_id][0])
            else:
                labels.append(f"Class_{int(class_id)}")
        
        # Convert bboxes to tensor
        if len(bbox_list) == 0:
            return image_tensor
            
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        
        # Default colors
        if colors is None:
            colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"] * 10
        
        # Draw bounding boxes
        try:
            result_image = draw_bounding_boxes(
                image_tensor,
                bbox_tensor,
                labels=labels,
                colors=colors[:len(bbox_tensor)] if colors else None,
                width=2
            )
            return result_image
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error drawing bounding boxes: {e}")
            return image_tensor

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            if cls_id < 0: continue
            cls_name = [key for key, value in self.DETECTION_CLASSES.items() if value == cls_id][0] if self.DETECTION_CLASSES is not None else str(cls_id)
            cls_id = class_names[cls_name]
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(cls_name, score * 100)
            txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img
    
    def  gt_to_vis(self,frame, results):
        """
        Visualize ground truth bounding boxes on the original image. the vis method need x1, y1, x2, y2 format
        Args: frame (torch.Tensor): Original image tensor
              results (dict): Results dictionary containing 'BB' key with bounding boxes
        Returns: cv2  Image with drawn bounding boxes
        """
        labels = []
        # Convert bboxes from [class_id, x_center, y_center, w, h] to [x1, y1, x2, y2]
        bboxes_xyxy = []
        for bbox in results['BB']:
            class_id, x_center, y_center, w, h = bbox
            if class_id < 0: continue
            x1, y1 = x_center-w*0.5, y_center-h*0.5
            x2, y2 = x1 + w, y1 + h
            bboxes_xyxy.append([x1, y1, x2, y2])
            labels.append(int(class_id))
        
        frame_ = visual.tensor_to_cv2_image(frame)
        
        cv_frame = self.vis(frame_, bboxes_xyxy, [1]*len(bboxes_xyxy), labels, conf=0.1, class_names=self.DETECTION_CLASSES)

        return frame_.get()



    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = parse.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def pad_and_resize_pil_image(self, image, target_size=(512, 512)):
        """Pads a PIL image to be square and resizes it."""
        original_size = image.size # (width, height)
        max_dim = max(original_size)
        
        pad_width = max_dim - original_size[0]
        pad_height = max_dim - original_size[1]
        
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        
        square_image = Image.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
        square_image.paste(image, (pad_left, pad_top))
        
        scale_factor = target_size[0] / max_dim
        resized_image = square_image.resize(target_size, Image.LANCZOS)
        
        padding_info = {
            'original_size': original_size,
            'square_size': max_dim,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'scale_factor': scale_factor
        }
        return resized_image, scale_factor, padding_info

    def load_and_resize_image(self, idx, target_size=(512, 512)):
        return self.load_and_resize(self.img_infos, self.img_dir, idx, target_size)

    def load_and_resize_events(self, idx, target_size=(512, 512)):
        return self.load_and_resize(self.events_infos, self.events_dir, idx, target_size)

    def load_and_resize(self, infos, dir, idx, target_size=(512, 512)):
        """
        Load events, add black padding to make it square, then resize to target size.
        Args:
            idx (int): Index of the sample
            target_size (tuple): Target size (width, height). Default: (512, 512)
        Returns:
            tuple: (resized_image, scale_factor, padding_info) where:
                   - resized_image is PIL Image resized to target_size
                   - scale_factor is the uniform scale applied after padding
                   - padding_info is dict with padding details
        """
        # Get events path
        info = infos[idx]
        path = osp.join(dir, info['ann']['subfolder'], info['filename'])
        input_ = "event" if "event" in path.lower() else "image"
        
         # Check if file exists
        if not osp.exists(path):
            if DEBUG >= 1:
                print(f"Events {input_} not found: {path}")
            return None, 1.0, {}
        
        try:
            # Load image
            image = Image.open(path).convert('RGB')
            return self.pad_and_resize_pil_image(image, target_size)
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error loading/processing image {path}: {e}")
            return None, 1.0, {}

    def transform_bboxes_with_padding(self, bboxes, padding_info):
        """
        Transform bounding boxes to account for padding and scaling.
        Args:
            bboxes (torch.Tensor): Tensor of bounding boxes in [N, 5] format where 5 = [class_id, x_center, y_center, w, h]
            padding_info (dict): Padding information from load_and_resize_image
        Returns: torch.Tensor: Transformed bounding boxes in [N, 5] format
        """
        if bboxes is None or len(bboxes) == 0 or not padding_info:
            return torch.zeros((0, 5), dtype=torch.float32)
        
        pad_left = padding_info['pad_left']
        pad_top = padding_info['pad_top']
        scale_factor = padding_info['scale_factor']
        
        # Initialize output tensor with max_labels dimension
        transformed_all = torch.zeros((self.max_labels, 5), dtype=torch.float32)
        transformed_all[:, 0] = -1  # Fill with class_id = -1

        # Find valid bboxes (class_id > 0)
        valid_mask = bboxes[:, 0] >= 0
        valid_bboxes = bboxes[valid_mask]
        
        if len(valid_bboxes) > 0:
            # Transform coordinates for valid bboxes
            transformed = valid_bboxes.clone()
            
            # Step 1: Add padding offset
            transformed[:, 1] = valid_bboxes[:, 1] + pad_left  # x
            transformed[:, 2] = valid_bboxes[:, 2] + pad_top   # y
            
            # Step 2: Apply uniform scaling
            transformed[:, 1] = transformed[:, 1] * scale_factor  # x
            transformed[:, 2] = transformed[:, 2] * scale_factor  # y
            transformed[:, 3] = valid_bboxes[:, 3] * scale_factor  # h
            transformed[:, 4] = valid_bboxes[:, 4] * scale_factor  # w
            
            # Copy transformed bboxes to output tensor
            num_valid = min(len(transformed), self.max_labels)
            transformed_all[:num_valid] = transformed[:num_valid]
        
        return transformed_all

    def get_padded_and_scaled_bbox_info(self, idx, target_size=(512, 512)):
        """
        Get bounding box information transformed for padded and scaled image.
        Args:
            idx (int): Index of the sample
            target_size (tuple): Target size (width, height). Default: (512, 512)
        Returns:
            tuple: (transformed_bboxes, padding_info) where:
                   - transformed_bboxes are bounding boxes in [class_id, x_center, y_center, w, h] format for the final image
                   - padding_info contains transformation details
        """
        # Get original bounding boxes
        bboxes, _ = self.extract_bboxes_from_json_polygons(idx)
        
        # Get padding and scaling information
        _, _, padding_info = self.load_and_resize_image(idx, target_size) if self.img_dir is not None else self.load_and_resize_events(idx, target_size)
        
        # Transform bounding boxes
        transformed_bboxes = self.transform_bboxes_with_padding(bboxes, padding_info)
        
        return transformed_bboxes, padding_info

    def reverse_bbox_transformation(self, transformed_bboxes, padding_info):
        """
        Reverse the bbox transformation to get original coordinates.
        Args:
            transformed_bboxes (torch.Tensor): Tensor of bboxes in [N, 5] format [class_id, x_center, y_center, w, h] transformed space
            padding_info (dict): Padding information from load_and_resize_image
        Returns: torch.Tensor: Bounding boxes in [N, 5] format original image coordinates
        """
        if transformed_bboxes is None or len(transformed_bboxes) == 0 or not padding_info:
            return torch.zeros((0, 5), dtype=torch.float32)
        
        pad_left = padding_info['pad_left']
        pad_top = padding_info['pad_top']
        scale_factor = padding_info['scale_factor']
        
        # Initialize output tensor with max_labels dimension
        original_all = torch.full((self.max_labels, 5), -1, dtype=torch.float32)
        
        # Find valid bboxes (class_id > 0)
        valid_mask = transformed_bboxes[:, 0] > 0
        valid_bboxes = transformed_bboxes[valid_mask]
        
        if len(valid_bboxes) > 0:
            # Reverse transformation for valid bboxes
            original = valid_bboxes.clone()
            
            # Step 1: Reverse scaling
            original[:, 1] = valid_bboxes[:, 1] / scale_factor  # x
            original[:, 2] = valid_bboxes[:, 2] / scale_factor  # y
            original[:, 3] = valid_bboxes[:, 3] / scale_factor  # h
            original[:, 4] = valid_bboxes[:, 4] / scale_factor  # w
            
            # Step 2: Remove padding offset
            original[:, 1] = original[:, 1] - pad_left  # x
            original[:, 2] = original[:, 2] - pad_top   # y
            
            # Copy reversed bboxes to output tensor
            num_valid = min(len(original), self.max_labels)
            original_all[:num_valid] = original[:num_valid]
        
        return original_all
