# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging
# Took from CMDA - https://github.com/CMDA/CMDA

import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
import numpy as np
from prettytable import PrettyTable
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import Compose
from torchvision.ops import masks_to_boxes

#from .builder import DATASETS
from helpers import DEBUG
from utils import parse



class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

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
                 img_dir,
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
                 extract_bboxes_from_masks=True,
                 bbox_min_area=100, **kwargs):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
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

        # Bounding box support
        self.load_bboxes = load_bboxes
        self.extract_bboxes_from_masks = extract_bboxes_from_masks
        self.bbox_min_area = bbox_min_area

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.endswith(img_suffix):
                    img_info = dict(filename=entry.name)
                    if ann_dir is not None:
                        seg_map = entry.name.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)

        if DEBUG>=1: print(
            f'Loaded {len(img_infos)} images from {img_dir}')
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        
        # Add bounding box information if available
        if self.load_bboxes:
            idx = results.get('idx', 0)  # Get index if available
            bboxes = self.get_bbox_info(idx) if hasattr(self, 'get_bbox_info') else []
            results['bboxes'] = bboxes
            results['bbox_fields'] = ['bboxes'] if bboxes else []
        else:
            results['bboxes'] = []
            results['bbox_fields'] = []
            
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, idx=idx)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info, idx=idx)
        self.pre_pipeline(results)
        return self.pipeline(results)

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

    def extract_bboxes_from_mask(self, segmentation_mask, min_area=None):
        """
        Extract bounding boxes from segmentation mask using torchvision.ops.masks_to_boxes.
        
        Args:
            segmentation_mask (np.ndarray): Segmentation mask with class IDs
            min_area (int, optional): Minimum area threshold for bounding boxes
            
        Returns:
            tuple: (bboxes, class_ids) where bboxes are [x1, y1, x2, y2] format
        """
        if min_area is None:
            min_area = self.bbox_min_area
            
        # Ensure mask is numpy array
        if torch.is_tensor(segmentation_mask):
            segmentation_mask = segmentation_mask.numpy()
        
        # Get unique classes (excluding ignore_index)
        unique_classes = np.unique(segmentation_mask)
        unique_classes = unique_classes[unique_classes != self.ignore_index]
        
        all_bboxes = []
        all_class_ids = []
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
                
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
                    if np.sum(instance_mask) < min_area:
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
                if np.sum(class_mask) >= min_area:
                    # Convert to torch tensor for masks_to_boxes
                    class_tensor = torch.from_numpy(class_mask.astype(bool))
                    
                    # Extract bounding box using torchvision.ops.masks_to_boxes
                    bbox = masks_to_boxes(class_tensor.unsqueeze(0))  # Add batch dimension
                    
                    if bbox.numel() > 0:  # If bbox was found
                        bbox = bbox.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                        all_bboxes.append(bbox)
                        all_class_ids.append(class_id)
        
        return all_bboxes, all_class_ids

    def extract_bboxes_from_json_polygons(self, idx):
        """
        Extract bounding boxes from JSON polygon annotations.
        Each polygon represents an individual instance of a class.
        Args: idx (int): Index of the sample
        Returns:
            tuple: (bboxes, class_ids, instance_masks) where:
                - bboxes are [x1, y1, x2, y2] format
                - class_ids are the class indices 
                - instance_masks are individual binary masks for each instance
        """
        if self.ann_dir is None: return [], [], []
            
        # Get JSON file path
        img_info = self.img_infos[idx]
        seg_map_path = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        json_file_path = seg_map_path.replace(self.seg_map_suffix, '_gtFine_polygons.json')
        
        if not osp.exists(json_file_path):
            if DEBUG >= 1: print(f"JSON annotation file not found: {json_file_path}")
            return [], [], []
        
        try:
            # Load JSON annotation
            with open(json_file_path, 'r') as f:
                annotation_data = json.load(f)
            
            img_height = annotation_data.get('imgHeight', 1024)
            img_width = annotation_data.get('imgWidth', 2048)
            objects = annotation_data.get('objects', [])
            
            all_bboxes = []
            all_class_ids = []
            all_instance_masks = []
            
            for obj_idx, obj in enumerate(objects):
                label = obj.get('label', '')
                polygon = obj.get('polygon', [])
                
                if not polygon or len(polygon) < 3:
                    continue
                
                # Map label to class index
                class_id = -1
                if self.CLASSES is not None and label in self.CLASSES:
                    class_id = self.CLASSES.index(label)
                elif hasattr(self, 'label_map') and self.label_map is not None:
                    # Handle custom class mapping if needed
                    original_class_id = self.CLASSES.index(label) if label in self.CLASSES else -1
                    class_id = self.label_map.get(original_class_id, -1) if original_class_id != -1 else -1
                
                # Skip if class not found or is ignore class
                if class_id == -1 or class_id == self.ignore_index or class_id not in self.DETECTION_CLASSES.keys(): continue
                
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
                    if np.sum(mask_array) < self.bbox_min_area:
                        continue
                    
                    # Convert to torch tensor for masks_to_boxes
                    mask_tensor = torch.from_numpy(mask_array)
                    
                    # Extract bounding box using torchvision.ops.masks_to_boxes
                    bbox = masks_to_boxes(mask_tensor.unsqueeze(0))  # Add batch dimension
                    
                    if bbox.numel() > 0:  # If bbox was found
                        bbox = bbox.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                        
                        # Validate bbox coordinates
                        x1, y1, x2, y2 = bbox
                        if x2 > x1 and y2 > y1:  # Valid bbox
                            all_bboxes.append(bbox)
                            all_class_ids.append(class_id)
                            all_instance_masks.append(mask_array)
                    elif DEBUG >= 1: print(f"No valid bbox found for object {obj_idx} in {json_file_path}")
                    
                except Exception as e:
                    if DEBUG >= 1: print(f"Error processing polygon for object {obj_idx} in {json_file_path}: {e}")
                    continue
            
            return all_bboxes, all_class_ids, all_instance_masks
            
        except Exception as e:
            if DEBUG >= 1: print(f"Error loading JSON file {json_file_path}: {e}")
            return [], [], []

    def get_bboxes_from_segmentation(self, idx):
        """
        Load segmentation mask and extract bounding boxes using masks_to_boxes.
        Args: idx (int): Index of the sample  
        Returns: tuple: (bboxes, class_ids) where bboxes are [x1, y1, x2, y2] format
        """
        if not self.extract_bboxes_from_masks or self.ann_dir is None:
            return [], []
            
        # Get segmentation file path
        img_info = self.img_infos[idx]
        seg_map_path = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        json_file_path = seg_map_path.replace(self.seg_map_suffix, '.json')
        if not osp.exists(json_file_path):
            if DEBUG >= 1: print(f"Segmentation json file not found: {json_file_path}")
            return [], []

        if not osp.exists(seg_map_path):
            if DEBUG >= 1: print(f"Segmentation file not found: {seg_map_path}")
            return [], []
        
        # Load segmentation mask
        try:
            seg_mask = np.array(Image.open(seg_map_path))
            return self.extract_bboxes_from_mask(seg_mask)
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error loading segmentation mask {seg_map_path}: {e}")
            return [], []
    def get_bboxes_from_json_segmentation(self, idx):
        """
        Get bounding box information for a sample from JSON polygon annotations.
        Args: idx (int): Index of the sample
        Returns: list: List of bounding boxes in [x1, y1, x2, y2] format
        """
        bboxes, class_ids, _ = self.extract_bboxes_from_json_polygons(idx)
        return bboxes

    def get_bbox_info(self, idx):
        """
        Get bounding box information for a sample.
        Priority: JSON polygons > segmentation masks > empty list
        Args: idx (int): Index of the sample
        Returns: list: List of bounding boxes in [x1, y1, x2, y2] format
        """
        bboxes, _, _ = self.extract_bboxes_from_json_polygons(idx)
        
        if len(bboxes) > 0: return bboxes
        elif self.extract_bboxes_from_masks:
            return self.get_bboxes_from_segmentation(idx)
        else: return []

    def draw_bounding_boxes_on_image(self, image, bboxes, class_ids=None, colors=None):
        """
        Draw bounding boxes on image using torchvision.utils.draw_bounding_boxes.
        Args:
            image (np.ndarray or torch.Tensor): Input image
            bboxes (list or torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format
            class_ids (list, optional): Class IDs for each bbox
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
        
        # Convert bboxes to tensor
        if not torch.is_tensor(bboxes):
            bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
        else:
            bbox_tensor = bboxes.float()
        
        # Create labels
        labels = None
        if class_ids is not None:
            if self.CLASSES is not None:
                labels = [self.CLASSES[cid] if cid < len(self.CLASSES) else f"Class_{cid}" 
                         for cid in class_ids]
            else:
                labels = [f"Class_{cid}" for cid in class_ids]
        
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

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])
        if DEBUG >= 1:
            print('per class results:')
            print('\n' + class_table_data.get_string())
            print('Summary:')
            print('\n' + summary_table_data.get_string())

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
