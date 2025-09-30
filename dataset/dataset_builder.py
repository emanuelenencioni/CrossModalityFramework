import os


from .dsec import DSECDataset  # make sure this import path is correct based on your project structure
from .cityscapes import CityscapesDataset
def build_from_config(cfg):
    """
    Factory method. Given the dataset configuration dictionary, instantiate and return the desired dataset train and test split.
    
    Currently implemented: DSEC_Night dataset.
    """
    assert "dataset" in cfg.keys(), "'dataset' params list missing from config file "
    dataset_cfg = cfg.get("dataset")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_name = dataset_cfg.get("name", None)
    if dataset_name is None:
        raise ValueError("Specify the 'name' parameter under dataset in the cfg.")

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

        assert any(key in dataset_cfg["outputs"] for key in ["rgb", "image", *event_keys]), "At least one of 'rgb', 'image' or event modalities must be specified in 'outputs' for CityscapesDataset"
        
        if any(key in dataset_cfg["outputs"] for key in event_keys):
            use_events = True
        else: use_events = False
        if "rgb" in dataset_cfg["outputs"] or "image" in dataset_cfg["outputs"]:
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

        train_txt = os.path.join(project_root,"dataset", dataset_cfg.get("train_split", "train.txt"))
        val_txt = os.path.join(project_root,"dataset", dataset_cfg.get("val_split", "val.txt"))
        train_ds =  CityscapesDataset(**dataset_cfg, split=train_txt)

        dataset_cfg["ann_dir"] = dataset_cfg["ann_dir"].replace("train", "val")
        if use_events: dataset_cfg["events_dir"] = dataset_cfg["events_dir"].replace("train", "val")
        if use_rgb: dataset_cfg["img_dir"] = dataset_cfg["img_dir"].replace("train", "val")

        test_ds = CityscapesDataset(**dataset_cfg, split=val_txt)
        return train_ds, test_ds
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
