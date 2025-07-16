# Cross modality framework

## Description

TODO

## Table of Contents
    TODO
- [Dependencies Installation](#dependencies-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dependencies Installation 
Use the following commands:
```bash
conda create -n CMF python=3.13.2 # Create a new conda environment
```
It doesn't need to work specifically with python 3.13.2, but is what i used for the development.
U can use `install_req.sh` to install all the dependencies:
```bash
conda activate CMF && sh install_req.sh
```
Pytorch 2.7.0 with cuda 12.6 is installed with the script.

## TODO
- [X] Handling DSEC_Night object detection
- [X] Finish a simple baseline model
- [X] First attempt to train the model -> only backbones in contrastive way (CLIP or similar)
- [X] Finish yaml loading
- [X] finish train_from_config
- [X] wandb integration

- [ ] Add frame as voxel_grid alternative. (maybe a model that can use both? or any event representation? can it be adaptable?)


- [X] put config in the  loss function build
- [X] one of the backbone = None -> unimodal training, a questo punto sarebbe ottimo che dual modality non è altro che il training di 2 unimodal backbone.
- [ ] create framework for domain adaptation -> (fare codice per fare domain adaptation cioe in cui si traina il new model e con anche l'old model)
- [ ] add train from saved model -> save and load config too.
- [ ] implement argparsing for the hyperparams (override of .yaml)
    - [X] implementation
    - [ ] FIX: given a parameter not present in the yaml file -> add to cfg.

- [ ] encoders must return a dict -> flatten_feat, projected_feat, preflatten_feat.
- [ ] Add detection head, remember NO flatten -> yolo latest version (with no transformers)
    - [/] watch for the YoloV11 loss function (i watched the YoloX instead)

- [ ] (per la proposta di metodo) considerare di fare la loss di contrastive solo sulla bbox e tutto il resto considerarlo come negative
- [ ] Refactor unimodal -> SingleModality. 
Priority:  

1. detection head
2. domain adaptation