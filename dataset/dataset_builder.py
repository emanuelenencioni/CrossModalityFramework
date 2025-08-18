import os


from .dsec import DSECDataset  # make sure this import path is correct based on your project structure
from .cityscapes import CityscapesDataset
def build_from_config(cfg):
    """
    Factory method. Given the dataset configuration dictionary, instantiate and return the desired dataset train and test split.
    
    Currently implemented: DSEC_Night dataset.
    """

    dataset_name = cfg.get("name", None)
    if dataset_name is None:
        raise ValueError("Specify the 'name' parameter under dataset in the cfg.")

    if dataset_name.lower() in ["dsec_night", "dsec_night_dataset", "dsecnight"]:
        # Determine the project root by navigating two levels up from this file's directory.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        txt_dir = project_root + "/dataset/"
        # Construct the dataset_txt_path (adjust the filename if needed)
        dataset_txt_path = os.path.join(txt_dir, cfg.get("train_split", "night_dataset.txt"))
        if not os.path.exists(dataset_txt_path):
            raise FileNotFoundError(f"Dataset file {dataset_txt_path} does not exist. Please check the data_dir and filename.")
        
        dataset_txt_val_path = None
        if cfg.get("val_split") is not None:
            dataset_txt_val_path = os.path.join(txt_dir, cfg["val_split"])
            if not os.path.exists(dataset_txt_val_path):
                raise FileNotFoundError(f"Validation dataset file {dataset_txt_val_path} does not exist. Please check the data_dir and filename.")
        
        outputs = cfg.get("outputs", {"events_vg", "image"})
        events_bins = cfg.get("events_bins", 1)
        events_clip_range = cfg.get("events_clip_range", None)
        events_bins_5_avg_1 = cfg.get("events_bins_5_avg_1", False)
        
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
        
        cfg["data_root"] = cfg["data_dir"]
        cfg["custom_classes"] = cfg.get("DSEC_classes", False)
        if cfg["custom_classes"] == True: cfg["extract_bboxes_from_masks"] = True
        cfg["pipeline"] = cfg.get("pipeline", [])
        cfg["img_dir"] = "cityscapes/leftImg8bit/train/aachen"
        cfg["ann_dir"] = "cityscapes/gtFine/train/aachen"
        return CityscapesDataset(**cfg), None
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
