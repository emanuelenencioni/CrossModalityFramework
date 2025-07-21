import yaml
import sys
import os
from training import optimizer,loss
from training.multimodal import TrainSSL
from training.unimodal import Trainer

from model.backbone import DualModalityBackbone, UnimodalBackbone
from model.detector import Detector
from model.yolox_head import YOLOXHead

from torch.utils.data import DataLoader
from dataset.dsec import DSECDataset, collate_ssl
from datetime import datetime
import torch
import wandb
import argparse
import random
import numpy as np
from helpers import DEBUG


def find_and_modify(d, tag, mod):
    if tag in d.keys():
        d[tag] = mod
    for k, v in d.items():
        if isinstance(v, dict):
            find_and_modify(v, tag, mod)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters.")

    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file")
    # Add arguments for all parameters in the YAML file
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default=None)
    parser.add_argument("--seed", type=int, help="Seed value", default=None)
    parser.add_argument("--device", type=str, help="Device to use (cuda or cpu)", default=None)

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, help="Dataset name", default=None)
    parser.add_argument("--data_dir", type=str, help="Data directory", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument("--num_workers", type=int, help="Number of workers", default=None)
    parser.add_argument("--train_type", type=str, help="Training type (ssl or sl)", default=None) # maybe useless

    # Backbone parameters
    parser.add_argument("--backbone_name", type=str, help="Backbone name", default=None)
    parser.add_argument("--rgb_backbone", type=str, help="RGB backbone name", default=None)
    parser.add_argument("--event_backbone", type=str, help="Event backbone name", default=None)
    parser.add_argument("--embed_dim", type=int, help="Embedding dimension", default=None)
    parser.add_argument("--input_size", type=int, help="Input size", default=None)
    #parser.add_argument("--num_classes", type=int, help="Number of classes")
    #parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--outputs", type=list, help="Dropout rate", default=None)

    # Optimizer parameters
    parser.add_argument("--optimizer_name", type=str, help="Optimizer name", default=None)
    parser.add_argument("--lr", type=float, help="Learning rate", default=None)
    parser.add_argument("--wd", type=float, help="Weight decay", default=None)

    # Loss function parameters
    parser.add_argument("--loss_name", type=str, help="Loss function name", default=None)

    # Training loop parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=None)
    parser.add_argument("--log_interval", type=int, help="Log interval (steps)", default=None)
    parser.add_argument("--val_interval", type=int, help="Validation interval (steps)", default=None)
    parser.add_argument("--checkpoint_interval", type=int, help="Checkpoint interval (epochs)", default=None)
    parser.add_argument("--save_folder", type=str,  help="Save folder", default=None)

    # Scheduler parameters
    parser.add_argument("--scheduler_name", type=str, help="Scheduler name", default=None)
    parser.add_argument("--factor", type=float, help="Scheduler factor", default=None)
    parser.add_argument("--patience", type=int, help="Scheduler patience", default=None)
    parser.add_argument("--monitor", type=str,  help="Scheduler monitor metric", default=None)
    parser.add_argument("--mode", type=str, help="Scheduler mode (min or max)", default=None)

    # Logging parameters
    parser.add_argument("--logger_name", type=str, help="Logger name", default=None)
    parser.add_argument("--project", type=str,  help="Wandb project name", default=None)
    parser.add_argument("--entity", type=str, help="Wandb entity name", default=None)

    args = parser.parse_args()

    cfg = None
    with open(vars(args)['config_path']) as file:
        cfg = yaml.safe_load(file)
    
    assert cfg
    # Update cfg with parsed arguments
    for k, v in vars(args).items():
        if k != "config_path" and v is not None:
            find_and_modify(cfg,k,v)
        else:
            # Ensure the value from cfg is used if the argument is not provided
            v_cfg = getattr(cfg, k, None)
            if v_cfg is not None:
                setattr(cfg, k, v_cfg)
    return cfg

def check_backbone_params(cfg):
    """
    Check if the backbone parameters are correctly specified in the configuration file.
    :param cfg: The configuration dictionary.
    :return: True if both event and rgb backbones are specified, False otherwise.  Also returns the specified backbone
    """
    assert 'backbone' in cfg.keys(), "Error - specify the backbone"
    cfg_b = cfg['backbone']

    assert 'backbone' in cfg.keys(), "Error - specify the backbone"
    cfg_b = cfg['backbone']
    assert 'embed_dim' in cfg_b.keys(), "Error - specify the embed_dim"
    assert 'input_size' in cfg_b.keys(), "Error - specify the input_size"
    ev_bb = cfg_b['event_backbone'] if 'event_backbone' in cfg['backbone'].keys() else None
    rgb_bb = cfg_b['rgb_backbone'] if 'rgb_backbone' in cfg['backbone'].keys() else None
    
    
    assert ev_bb != None or rgb_bb != None, "Error - specify at least one backbone: event or rgb"
    if ev_bb is None:
        return False, rgb_bb
    elif rgb_bb is None:
        return False, ev_bb

    return True, None

def print_cfg_params(cfg, indent=0):
    """
    Recursively prints all parameters specified in the cfg dictionary.
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_cfg_params(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")

if __name__ == "__main__":

    cfg = parse_arguments()
    if DEBUG>0:
        print("Configuration parameters:")
        print_cfg_params(cfg)

    # Setup #
    assert 'model' in cfg.keys(), "Error - specify the model architecture"
    dual_modality, backbone = check_backbone_params(cfg['model'])
    if 'head' in cfg['model'].keys():
        assert 'bb_num_classes' in cfg['dataset'], "Error - number of classes need to be specified in unimodal training"
        model = Detector(backbone,num_classes=cfg['dataset']['bb_num_classes'], img_size=int(cfg['model']['backbone']['input_size']))
    else:
        if dual_modality: #TODO: Fix this, not only backbones in there
            model = DualModalityBackbone(rgb_backbone=cfg['model']['backbone']['rgb_backbone'],
                        event_backbone=cfg['model']['backbone']['event_backbone'],
                        embed_dim=cfg['model']['backbone']['embed_dim'],
                        img_size=cfg['model']['backbone']['input_size']
            )
        else:
            model = UnimodalBackbone(backbone, embed_dim=cfg['model']['backbone']['embed_dim'],
                        img_size=cfg['model']['backbone']['input_size'])
    
    # Loss   
    assert 'loss' in cfg.keys(), "loss params list missing in yaml file"
    criterion, learnable = loss.build_from_config(cfg['loss'])

    if learnable:
        params = list(model.parameters()) + list(criterion.parameters())
        print(criterion.parameters())
    else:
        params = model.parameters()
    assert 'loss' in cfg.keys(), "'optimizer' params list missing in yaml file"
    opti = optimizer.build_from_config(params, cfg['optimizer'])
    
    if 'seed' in cfg.keys() and cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['seed'])
    
    # TODO: adjust for each dataset of interest
    # Dataloader (CMDA)
    events_bins_5_avg_1 = False
    if events_bins_5_avg_1:
        events_bins = 1
        events_clip_range = None  # (1.0, 1.0)
    else:
        events_bins = 1
        events_clip_range = None
    dir_path = os.path.dirname(os.path.realpath(__file__))

    
    outputs = {'events_vg', 'image'}
    if dual_modality:
        if 'outputs' in cfg['dataset'].keys():
            outputs = cfg['dataset']['outputs']
        else:
            print(" 'outputs' params list missing from config file, using default for ssl")
            
        
    else:
        assert 'outputs' in cfg['dataset'].keys(), "Error, missing mandatory outputs for the dataset in unimodal training"
        outputs = cfg['dataset']['outputs']

    dataset = DSECDataset(dataset_txt_path=dir_path+'/dataset/night_dataset.txt',
                           outputs=outputs,
                           events_bins=events_bins, events_clip_range=events_clip_range,
                           events_bins_5_avg_1=events_bins_5_avg_1)
    
    # Dataloader (CMDA)
    assert 'dataset' in cfg.keys(), " 'dataset' params list missing from config file"
    assert 'batch_size' in cfg['dataset'].keys(), " specify 'batch_size' dataset param"
    num_workers=2
    if 'num_workers' in cfg['dataset'].keys(): 
        num_workers = int(cfg['dataset']['num_workers'])
    dataloader = DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], num_workers=num_workers, shuffle=False, collate_fn=collate_ssl, pin_memory=True)
    
    wandb_log = False
    run_name = cfg['model']['backbone']['name'] if ('name' in cfg['model']['backbone'].keys() and cfg['model']['backbone']['name'] != '') else model.get_name()
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'logger' in cfg.keys() and 'name' in cfg['logger'].keys():
        if(cfg['logger']['name'] == 'wandb'):
            wandb_cfg = cfg['logger']
            assert 'project' in wandb_cfg.keys(), "specify 'project' wandb param"
            assert 'entity' in wandb_cfg.keys(), "specify 'entity' wandb param"
            wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], name=run_name, config=cfg,settings=wandb.Settings(init_timeout=600))
            # Log model architecture
            wandb.watch(model)
            wandb_log = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)    
    # Trainer
    assert 'trainer' in cfg.keys(), "'trainer' params list missing from config file "
    
    if dual_modality:
        trainer = TrainSSL(model, dataloader, opti, criterion, device, cfg, root_folder=dir_path, wandb_log=wandb_log)
    else:
        trainer = Trainer(model,dataloader, opti, criterion, device,  cfg, root_folder=dir_path, wandb_log=wandb_log)
    trainer.train()
