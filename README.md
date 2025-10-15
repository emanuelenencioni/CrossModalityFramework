# Cross modality framework

## Description
This project wants to provide a flexible framework for cross-modality learning, enabling the integration and training of models across different data modalities (e.g., images and events). It supports unimodal and multimodal architectures, domain adaptation, and tasks such as detection and segmentation(WIP). The framework is designed for extensibility, allowing easy configuration, modular backbone and head selection. It is suitable for research and development in multi-domain and multi-task machine learning scenarios.



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

1. **Preparing the Data:**
    - Organize your datasets according to the required modalities (e.g., images, events). 
    - Update configuration files with dataset paths and parameters.

2. **Configuration:**
    Edit the configuration file (typically located in `config/`) to specify:
    - Modality-specific parameters.
    - Model details (e.g. backbone and head settings).
    - Training hyperparameters.

    Example snippet:
    ```yaml
    model:
    name: 'resnet50_yolox'
    head: 
        name: 'yolox_head' # redundant here, but useful for clarity
        num_classes: 8
        losses_weights: [5.0, 1.0, 1.0, 1.0]  # [iou, obj, cls, l1]
    backbone:
        name: '' # if name is empty -> stack of the two backbone will be used
        rgb_backbone: resnet50 # from timm
        pretrained: True # if not specified, it will be set to True
        pretrained_weights: #'../resnet50_backbone_from_detr.pth' # path to pretrained weights if needed
        embed_dim: 256
        input_size: 512
        output_indices: [3, 4] # indices of the output layers to be used
    ```

    Use existing templates as a reference.
    ### ⚠️⚠️⚠️ Warning ⚠️⚠️⚠️
    The config `.yaml` file must include all the parameters defined in the argparse for them to take effect. Otherwise, the argument parsing WILL NOT WORK.  You can think of the `.yaml` file as containing the default arguments for your specific training run.

3. **The losses**, two types:
    - **Unimodal Tasks:** Losses are computed within each head of the respective models, such as the "YoloXHead." A base class will be implemented soon to ensure a consistent interface across all heads. 
    - **Multimodal Tasks:** The loss needs to be specified in the `.yaml` configuration file. A factory method will handle the building process. If a loss is not yet implemented, add it to the builder.

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
    For DSEC-Night, the script `create_dataset_vg.py` is also available. This script will create caches for voxel grids inside your dataset folder. This was added due to the high computational cost of creating voxel grids at runtime. \
    (⚠️⚠️⚠️ Currently under mantainance ⚠️⚠️⚠️)

5. **Training the Model:**
    - Run the training script with your configuration file:
      ```bash
      python train_from_config.py configs/your_config.yaml
      ```
      You can add any arguments you want (at least, the ones specified in utils/argparser.py). Arguments are parsed as follows:
    - Use `_` to separate words in argument names (e.g., `--batch_size`).
    - Use `-` to specify a key within a sub-dictionary (e.g., `--logger-name`, where `name` is a key inside the `logger` sub-dictionary in the config).
    - To monitor the process, the framework is fully integrated with wandb.
    - Use the `DEBUG` environment variable to monitor the internal processes. Higher values (>=1) will increase the verbosity of the output, Most used are:          
        - `DEBUG=1`: Provides basic information such as real-time loss for each batch and setup details.
        - `DEBUG=3`: e.g. saves and allows inspection of ground truth bounding box images (just one, >4 for all of them).

6. **Evaluating the Model:**
    - Run the evaluation script:
      ```shell
      python detect_from_config.py --config config/your_config.yaml --checkpoint path/to/your/checkpoint.pth --input_image path_to_image
      ```
## How to Create a Custom Task Head

This guide explains how to implement a custom task head (e.g., classification, detection) that integrates seamlessly with the framework's training loop. 

### Forward Function Requirements

Your custom head's `forward()` method **need** to implement and calculate its own task specific loss. It also must follow this signature during training:

```python
def forward(self, x, targets=None):
    """
    Args:
        x: Input features from the backbone
        targets: Ground truth labels (required during training)
    
    Returns:
        During training:
            tuple: (outputs, total_loss, losses_dict)
                - outputs: Your model's predictions
                - total_loss: Single scalar tensor with the weighted sum of all losses
                - losses_dict: Dictionary with individual loss values
        
        During inference:
            tuple: (outputs, None) or just outputs
    """
    # Your implementation here
    pass
```


## TODO
- [ ] REFACTORING:
    - [X] modules on main (no assert there pls)
    - [X] clean and understandable
- [X] Add training events on cityscapes with the IC from CMDA
- [X] make unimodal agnostic to the losses defined in the model head, so make the model head agnostic too (use like tot_loss, dict, output), where dict contain all the specific losses. 
- [X] Fix DSECNight
- [X] check dsec-det no event on bbox
- [ ] Fix multimodal training 
        [X] refactoring -> sub class of unimodal.
- [ ] Make the logger uniform for all the framework (probably the one in dsec_evaluator)
- [ ] in custom, get the output frame dims in input (for now it only works with 512x512)

- [ ] check for validity of VGs
 
- [ ] Add segmentation task (low prio)
- [ ] Add same build from config as mmcv.

- [ ] (per la proposta di metodo) considerare di fare la loss di contrastive solo sulla bbox e tutto il resto considerarlo come negative


## Model Output Format

All models in this framework follow a standardized output format to ensure consistency and easy integration with the training pipeline.

### Standard Model Output Structure

Models must return a dictionary with the following structure:

```python
{
    'backbone_features': {
        'preflatten_feat': [...],    # Multi-scale features (list of tensors)
        'flattened_feat': tensor,    # Flattened features (optional)
        # ... other backbone-specific outputs
    },
    'head_outputs': tensor,          # Task-specific predictions
    'total_loss': scalar,            # Total weighted loss (training only)
    'losses': {                      # Individual loss components (training only)
        'iou_loss': scalar,
        'obj_loss': scalar,
        'cls_loss': scalar,
        'l1_loss': scalar,           # Optional, task-dependent
        # ... other task-specific losses
    }
}
```

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
