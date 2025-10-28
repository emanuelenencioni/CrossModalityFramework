import os
import cv2
import albumentations as A

import sys
from pathlib import Path
from utils.helpers import DEBUG


from .dsec import DSECDataset  # make sure this import path is correct based on your project structure
from .cityscapes import CityscapesDataset

def build_from_config(cfg):
    """
    Factory method. Given the dataset configuration dictionary, instantiate and return the desired dataset train and val split.

    Currently implemented: DSEC_Night, cityscapes
    """
    assert "dataset" in cfg.keys(), "'dataset' params list missing from config file "
    dataset_cfg = cfg.get("dataset")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_name = dataset_cfg.get("name", None)
    
    # Default data augmentatations
    aug =  A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(p=0.5),
                # This crop is safe for bounding boxes, it tries to keep them in the frame
                A.RandomSizedBBoxSafeCrop(width=2048, height=1024, erosion_rate=0.2, p=0.5),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels'], min_visibility=0.3))

    if dataset_name is None:
        raise ValueError("Specify the 'name' parameter under dataset in the cfg.")
    
    aug = None
    if dataset_cfg.get('use_augmentations', True):
        if 'augmentations' in dataset_cfg.keys() and dataset_cfg['augmentations'] is not None:
            # Build the augmentation pipeline from config
            aug_cfg = dataset_cfg['augmentations']
            aug = augmentation_builder(aug_cfg)
            if DEBUG>=1: print(f"Using augmentations with {len(aug.transforms)} transforms")
        else:
            if DEBUG>=1: print("No augmentations specified, using default")
    else:
        if DEBUG>=1: print("Augmentations disabled")
        aug = None

            

    if dataset_name.lower() in ["dsec_night", "dsec_night_dataset", "dsecnight"]:
        # Determine the project root by navigating two levels up from this file's directory.
        
        txt_dir = project_root + "/dataset/"
        # Construct the dataset_txt_path (adjust the filename if needed)
        dataset_txt_path = os.path.join(txt_dir, dataset_cfg.get("train_split", "night_dataset.txt"))
        if not os.path.exists(dataset_txt_path):
            raise FileNotFoundError(f"Dataset file {dataset_txt_path} does not exist. Please check the data_dir and filename.")
        
        dataset_txt_val_path = None
        if dataset_cfg.get("val_split") is not None:
            dataset_txt_val_path = os.path.join(txt_dir, dataset_cfg["val_split"])
            if not os.path.exists(dataset_txt_val_path):
                raise FileNotFoundError(f"Validation dataset file {dataset_txt_val_path} does not exist. Please check the data_dir and filename.")
        
        outputs = dataset_cfg.get("outputs", {"events_vg", "image"})
        events_bins = dataset_cfg.get("events_bins", 1)
        events_clip_range = dataset_cfg.get("events_clip_range", None)
        events_bins_5_avg_1 = dataset_cfg.get("events_bins_5_avg_1", False)
        
        if dataset_txt_val_path is not None:
            return DSECDataset(
                dataset_txt_path=dataset_txt_path,
                outputs=outputs,
                events_bins=events_bins,
                events_clip_range=events_clip_range,
                events_bins_5_avg_1=events_bins_5_avg_1
            ), DSECDataset(
                dataset_txt_path=dataset_txt_val_path,
                outputs=outputs,
                events_bins=events_bins,
                events_clip_range=events_clip_range,
                events_bins_5_avg_1=events_bins_5_avg_1
            )
        else:
            return DSECDataset(
                dataset_txt_path=dataset_txt_path,
                outputs=outputs,
                events_bins=events_bins,
                events_clip_range=events_clip_range,
                events_bins_5_avg_1=events_bins_5_avg_1
            ), None
        
    elif dataset_name.lower() in ["cityscape", "cityscapes", "cityscapes_dataset", "cityscape_dataset", "cityscapesdataset", "cityscapedataset"]:
        event_keys = ["events", "events_vg", "events_frames"]
        rgb_keys = ["rgb", "images", "image"]

        assert any(key in dataset_cfg["outputs"] for key in [*rgb_keys, *event_keys]), "At least one of 'rgb', 'image' or event modalities must be specified in 'outputs' for CityscapesDataset"
        
        if any(key in dataset_cfg["outputs"] for key in event_keys):
            use_events = True
        else: use_events = False
        
        if any(key in dataset_cfg["outputs"] for key in rgb_keys):
            use_rgb = True
        else: use_rgb = False
        
        dataset_cfg["data_root"] = dataset_cfg["data_dir"]
        if dataset_cfg.get("custom_classes", False) == True: 
            dataset_cfg["extract_bboxes_from_masks"] = True
            dataset_cfg["load_bboxes"] = True
        else:
            print("\033[93m"+"WARNING: custom_classes is set to False, using default Cityscapes classes"+"\033[0m")
        dataset_cfg["pipeline"] = dataset_cfg.get("pipeline", [])
        
        if use_events: dataset_cfg["events_dir"] = "cityscapes/leftImg8bitEvents/train/"
        if use_rgb: dataset_cfg["img_dir"] = "cityscapes/leftImg8bit/train/"
        
        dataset_cfg["ann_dir"] = "cityscapes/gtFine/train/"
        dataset_cfg["augmentations"] = aug

        train_txt = os.path.join(project_root,"dataset", dataset_cfg.get("train_split", "train.txt"))
        val_txt = os.path.join(project_root,"dataset", dataset_cfg.get("val_split", "val.txt"))
        train_ds =  CityscapesDataset(**dataset_cfg, split=train_txt)
        if dataset_cfg.get("build_test", False):
            dataset_cfg["ann_dir"] = dataset_cfg["ann_dir"].replace("train", "test")
            if use_events: dataset_cfg["events_dir"] = dataset_cfg["events_dir"].replace("train", "test")
            if use_rgb: dataset_cfg["img_dir"] = dataset_cfg["img_dir"].replace("train", "test")
        else:
            dataset_cfg["ann_dir"] = dataset_cfg["ann_dir"].replace("train", "val")
            if use_events: dataset_cfg["events_dir"] = dataset_cfg["events_dir"].replace("train", "val")
            if use_rgb: dataset_cfg["img_dir"] = dataset_cfg["img_dir"].replace("train", "val")
        # For test dataset NO aug
        dataset_cfg['use_augmentations'] = False
        dataset_cfg["augmentations"] = None
        test_ds = CityscapesDataset(**dataset_cfg, split=val_txt)
        return train_ds, test_ds
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

def build_test(cfg):
    """
    Factory method. Given the dataset configuration dictionary, instantiate and return the desired TEST dataset split.

    Currently implemented: same as build_from_config but only returns test split (not val split, test split).
    """
    assert "test_split" in cfg['dataset'].keys(), "'test_split' params missing from dataset config file "
    cfg['dataset']['val_split'] = cfg['dataset']['test_split']
    cfg['dataset']['build_test'] = True
    _, test_ds = build_from_config(cfg)
    return test_ds


def augmentation_builder(aug_cfg):
    """Builds an augmentation pipeline based on the provided configuration.
    
    Supported augmentations:
    - hflip_prob: Horizontal flip probability
    - vflip_prob: Vertical flip probability
    - resize: [height, width] for resizing
    - scale: [min, max] scale range for ShiftScaleRotate
    - shift_limit: Shift limit for ShiftScaleRotate
    - rotate_limit: Rotation limit in degrees
    - brightness_contrast: Enable random brightness and contrast
    - brightness_contrast_prob: Probability for brightness/contrast augmentation
    - blur_prob: Probability for blur augmentation
    - crop_size: [height, width] for random crop
    - safe_crop: Enable safe bbox crop (preserves bboxes)
    - erosion_rate: Erosion rate for safe crop
    - normalize: Normalize images (mean, std)
    - min_visibility: Minimum bbox visibility after transforms
    """
    import cv2
    
    transforms = []
    
    # Horizontal Flip
    if 'hflip_prob' in aug_cfg.keys() and aug_cfg['hflip_prob'] > 0:
        transforms.append(A.HorizontalFlip(p=aug_cfg['hflip_prob']))
    
    # Vertical Flip
    if 'vflip_prob' in aug_cfg.keys() and aug_cfg['vflip_prob'] > 0:
        transforms.append(A.VerticalFlip(p=aug_cfg['vflip_prob']))
    
    # Shift, Scale, Rotate
    if 'scale' in aug_cfg.keys() or 'shift_limit' in aug_cfg.keys() or 'rotate_limit' in aug_cfg.keys():
        scale_limit = aug_cfg.get('scale', 0.0)
        if isinstance(scale_limit, list) and len(scale_limit) == 2:
            scale_limit = (scale_limit[0] - 1.0, scale_limit[1] - 1.0)  # Convert to relative scale
        transforms.append(A.ShiftScaleRotate(
            shift_limit=aug_cfg.get('shift_limit', 0.0),
            scale_limit=scale_limit,
            rotate_limit=aug_cfg.get('rotate_limit', 0),
            p=aug_cfg.get('shift_scale_rotate_prob', 0.5),
            border_mode=cv2.BORDER_CONSTANT
        ))
    
    # Brightness and Contrast
    if aug_cfg.get('brightness_contrast', False):
        transforms.append(A.RandomBrightnessContrast(
            p=aug_cfg.get('brightness_contrast_prob', 0.5)
        ))
    
    # Blur
    if aug_cfg.get('blur_prob', 0) > 0:
        transforms.append(A.Blur(blur_limit=aug_cfg.get('blur_limit', 3), p=aug_cfg['blur_prob']))
    
    # Safe Crop for bounding boxes
    if aug_cfg.get('safe_crop', False) and 'resize' in aug_cfg.keys():
        resize = aug_cfg['resize']
        if isinstance(resize, list) and len(resize) == 2:
            height, width = resize[0], resize[1]
            transforms.append(A.RandomSizedBBoxSafeCrop(
                width=width,
                height=height,
                erosion_rate=aug_cfg.get('erosion_rate', 0.2),
                p=aug_cfg.get('safe_crop_prob', 0.5)
            ))
    
    # Random Crop
    elif 'crop_size' in aug_cfg.keys():
        crop = aug_cfg['crop_size']
        if isinstance(crop, list) and len(crop) == 2:
            transforms.append(A.RandomCrop(height=crop[0], width=crop[1]))
    
    # Resize (applied if not using safe crop)
    if 'resize' in aug_cfg and not aug_cfg.get('safe_crop', False):
        resize = aug_cfg['resize']
        if isinstance(resize, list) and len(resize) == 2:
            transforms.append(A.Resize(height=resize[0], width=resize[1]))
    
    # Normalize
    if aug_cfg.get('normalize', False):
        mean = aug_cfg.get('mean', [0.485, 0.456, 0.406])
        std = aug_cfg.get('std', [0.229, 0.224, 0.225])
        transforms.append(A.Normalize(mean=mean, std=std))
    
    if len(transforms) == 0:
        raise ValueError("No valid augmentations specified in configuration")
    
    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['bbox_labels'],
        min_visibility=aug_cfg.get('min_visibility', 0.3)
    ))