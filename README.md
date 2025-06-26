# Cross modality framework

## Description

TODO

## Table of Contents
    TODO
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

## TODO
- [X] Handling DSEC_Night object detection
- [X] Finish a simple baseline model
- [X] First attempt to train the model -> only backbones in contrastive way (CLIP or similar)
- [X] Finish yaml loading
- [X] finish train_from_config
- [X] wandb integration

- [ ] Add frame as voxel_grid alternative. (maybe a model that can use both? or any event representation? can it be adaptable?)


- [X] put config in the  loss function build
- [ ] one of the backbone = None -> unimodal training
- [ ] create framework for domain adaptation -> (fare codice per fare domain adaptation cioe in cui si traina il new model e con anche l'old model)
- [ ] implement argparsing for the hyperparams (override of .yaml)
- [ ] encoders must return a dict -> flatten_feat, projected_feat, preflatten_feat.
- [ ] Add detection head, remember NO flatten -> yolo latest version (with no transformers)
    - [ ] watch for the YoloV11 loss function

- [ ] (per la proposta di metodo) considerare di fare la loss di contrastive solo sulla bbox e tutto il resto considerarlo come negative

Priority:  

1. detection head
2. domain adaptation