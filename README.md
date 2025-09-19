# Cross modality framework

## Description
This project provides a flexible framework for cross-modality learning, enabling the integration and training of models across different data modalities (e.g., images and events). It supports unimodal and multimodal architectures, domain adaptation, and tasks such as detection and segmentation. The framework is designed for extensibility, allowing easy configuration, modular backbone and head selection. It is suitable for research and development in multi-domain and multi-task machine learning scenarios.



## Table of Contents
- [Dependencies Installation](#dependencies-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Model Output Format](#model-output-format)

## Dependencies Installation
 
You can set up the environment using the following commands:

```bash
conda create -n CMF python=3.13.2  pip # Create a new conda environment (other Python 3.x versions should work)
```

To install all required dependencies, use the provided script:

```bash
conda activate CMF
sh install_req.sh
```

The script will install PyTorch 2.7.0 with CUDA 12.6. Adjust versions as needed for your system.

## Usage

This guide provides instructions to set up and run the framework:

2. **Preparing the Data:**
    - Organize your datasets according to the required modalities (e.g., images, events). 
    - Update configuration files with dataset paths and parameters.

3. **Configuration:**
    Edit the configuration file (typically located in `config/`) to specify:
    - Modality-specific parameters.
    - Model details (e.g. backbone and head settings).
    - Training hyperparameters.

    Example snippet:
    ```yaml
    model:
        backbone: ResNet50
        head: DetectionHead
    training:
        batch_size: 32
        epochs: 50
    ```

    Use existing templates as a reference.

    ### ⚠️⚠️⚠️ Warning
    The config `.yaml` file must include all the parameters defined in the argparse for them to take effect. Otherwise, the argument parsing WILL NOT WORK.
4. **Preparing the Dataset**
    4. **Preparing the Dataset**

        The DSEC-Night and Cityscapes datasets are currently supported. To prepare them for training:

         1. Ensure the root directory of each dataset (or a symlink to it) is placed within the `data/` folder.
         2. Run the appropriate script to generate the train and validation split files.

         - **For Cityscapes:**
            ```bash
            python dataset/create_cs_txt.py
            ```

         - **For DSEC-Night:**
            ```bash
            python dataset/create_dataset_txt.py
            ```

        For DSEC-Night, the script `create_dataset_vg.py` is also available. This script will create caches for voxel grids inside your dataset folder. This was added due to the high computational cost of creating voxel grids at runtime.

5. **Training the Model:**
    - Run the training script with your configuration file:
      ```bash
      python train_from_config.py configs/your_config.yaml
      ```
    - To monitor the process, the framework is fully integrated with wandb.

6. **Evaluating the Model:**
    - Run the evaluation script:
      ```bash
      python detect_from_config.py --config config/your_config.yaml --checkpoint path/to/your/checkpoint.pth
      ```

## TODO

- [X] check cityscape output as DSEC
- [X] Fix coords -> cx,cy,h,w in custom to cx,cy,w,h
- [ ] Check objectness output value 
- [ ] Fix: errors in evaluator (summarize after)
- [ ] FIX: errors in multimodal training.
- [ ] FIX: errors in DSEC-det -> bbox should be aligned to images. 
- [ ] check for validity of VGs
 
- [ ] Add segmentation task (low prio)
- [ ] Add same build from config as mmcv.

- [ ] (per la proposta di metodo) considerare di fare la loss di contrastive solo sulla bbox e tutto il resto considerarlo come negative
- [ ] Refactor unimodal -> SingleModality. 


## Model Output Format

### YOLOXHead Input and Output Specifications

#### **Input Format (Training)**
During training, the YOLOXHead expects ground truth labels in the following format:
- **Labels tensor shape**: `[batch_size, max_objects, 5]`
- **Label format**: `[class_id, x_center, y_center, width, height]`
  - `class_id`: Integer class identifier (0-based indexing)
  - `x_center, y_center`: Center coordinates of the bounding box (absolute pixel coordinates)
  - `width, height`: Width and height of the bounding box (absolute pixel values)
- **Coordinate system**: Center-based format with absolute pixel coordinates
- **Padding**: Unused label slots should be filled with negative values (e.g., `-1`)

#### **Output Format (Inference)**
By default, the model outputs bounding boxes in the format: `[x_center, y_center, width, height, objectness_score, class_confidence_0, class_confidence_1, ...]`

**Detailed breakdown:**
- **Bounding box coordinates**: `[x_center, y_center, width, height]` (center coordinates with width/height)
- **Objectness score**: Confidence that the box contains an object
- **Class confidences**: Per-class confidence scores (one for each class)
- **Coordinate system**: Center-based format with absolute pixel coordinates
- **Output tensor shape**: `[batch_size, num_detections, 5 + num_classes]`


#### **Internal Processing**
The YOLOXHead uses center-based coordinates throughout its internal processing:
- Loss computation uses `[x_center, y_center, width, height]` format
- Assignment algorithms expect center-based ground truth
- IoU calculations support both coordinate formats via the `xyxy` parameter
- Multi-scale feature processing maintains center-based representation