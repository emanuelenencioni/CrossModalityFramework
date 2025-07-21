# Cross modality framework

## Description
This project provides a flexible framework for cross-modality learning, enabling the integration and training of models across different data modalities (e.g., images and events). It supports unimodal and multimodal architectures, domain adaptation, and tasks such as detection and segmentation. The framework is designed for extensibility, allowing easy configuration, modular backbone and head selection. It is suitable for research and development in multi-domain and multi-task machine learning scenarios.

## Table of Contents
    TODO
- [Dependencies Installation](#dependencies-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dependencies Installation

You can set up the environment using the following commands:

```bash
conda create -n CMF python=3.13.2  # Create a new conda environment (other Python 3.x versions should work)
```

To install all required dependencies, use the provided script:

```bash
conda activate CMF
sh install_req.sh
```

The script will install PyTorch 2.7.0 with CUDA 12.6. Adjust versions as needed for your system.

## TODO


- [X] put config in the  loss function build
- [X] one of the backbone = None -> unimodal training, a questo punto sarebbe ottimo che dual modality non è altro che il training di 2 unimodal backbone.
- [X] in yaml: backbone and head inside model
- [X] refactor sl.py e ssl.py in unimodal.py multimodal.py
- [ ] create framework for domain adaptation -> (fare codice per fare domain adaptation cioe in cui si traina il new model e con anche l'old model) -> this one deactivated for now
- [ ] add train from saved model -> save and load config too.
- [X] outputs return in the forward of YoloX but it should be the norm for all the heads. 
- [ ] Add segmentation task (low prio)


- [X] implement argparsing for the hyperparams (override of .yaml)
    - [X] implementation
    - [X] FIX: given a parameter not present in the yaml file -> ~~add to cfg~~. Instead if param not present in cfg, it will not be added.

- [X] encoders must return a dict -> flatten_feat, projected_feat, preflatten_feat.
- [X] Add detection head, remember NO flatten -> yolo latest version (with no transformers)
    - [/] watch for the YoloV11 loss function (i watched the YoloX instead)

- [ ] (per la proposta di metodo) considerare di fare la loss di contrastive solo sulla bbox e tutto il resto considerarlo come negative
- [ ] Refactor unimodal -> SingleModality. 
