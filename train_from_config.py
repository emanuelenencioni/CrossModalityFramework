import yaml
import sys
import os
from training import optimizer,loss
from training.multimodal import DualModalityTrainer
from training.unimodal import Trainer
from training.scheduler import scheduler_builder

from model.backbone import DualModalityBackbone, UnimodalBackbone
from model.yolox_head import YOLOXHead
from model.builder import build_model_from_cfg

from torch.utils.data import DataLoader
from dataset.dsec import DSECDataset, collate_ssl
from datetime import datetime
import torch
import wandb
import argparse
import random
import numpy as np
from helpers import DEBUG
from evaluator.dsec_evaluator import DSECEvaluator
from evaluator.cityscapes_evaluator import CityscapesEvaluator
import dataset.dataset_builder as dataset_builder


def find_and_modify(d, tag, mod):
    if tag in d.keys():
        d[tag] = mod
    for k, v in d.items():
        if isinstance(v, dict):
            find_and_modify(v, tag, mod)

def deep_dict_compare(dict1, dict2, path=""):
    """
    Recursively compare two dictionaries for equality.
    Returns True if equal, False otherwise.
    Prints differences with their paths.
    """
    if type(dict1) != type(dict2):
        print(f"Type mismatch at {path}: {type(dict1)} vs {type(dict2)}")
        return False
    
    if isinstance(dict1, dict):
        # Get all keys from both dictionaries
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                print(f"Key missing in first dict at {current_path}")
                return False
            if key not in dict2:
                print(f"Key missing in second dict at {current_path}")
                return False
                
            # Recursively compare values
            if not deep_dict_compare(dict1[key], dict2[key], current_path):
                return False
        return True
    
    elif isinstance(dict1, list):
        if len(dict1) != len(dict2):
            print(f"List length mismatch at {path}: {len(dict1)} vs {len(dict2)}")
            return False
        
        for i, (item1, item2) in enumerate(zip(dict1, dict2)):
            if not deep_dict_compare(item1, item2, f"{path}[{i}]"):
                return False
        return True
    
    else:
        # Compare primitive values
        if dict1 != dict2:
            print(f"Value mismatch at {path}: {dict1} vs {dict2}")
            return False
        return True


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters.")

    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file")
    # Add arguments for all parameters in the YAML file
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default=None)
    parser.add_argument("--seed", type=int, help="Seed value", default=None)
    parser.add_argument("--device", type=str, help="Device to use (cuda or cpu)", default="cuda")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file", default=None)
    # Dataset parameters
    parser.add_argument("--dataset-name", type=str, help="Dataset name", default=None, dest="dataset-name")
    parser.add_argument("--data_dir", type=str, help="Data directory", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument("--num_workers", type=int, help="Number of workers", default=None)
    parser.add_argument("--train_type", type=str, help="Training type (ssl or sl)", default=None) # maybe useless
    parser.add_argument("--train_split", type=str, help="Training split file", default=None)
    parser.add_argument("--val_split", type=str, help="Validation split file", default=None)
    parser.add_argument("--losses_weights", type=list, help="Losses weights", default=None)

    # Backbone parameters
    parser.add_argument("--model-backbone-name", type=str, help="Backbone name", default=None, dest="model-backbone-name")
    parser.add_argument("--rgb_backbone", type=str, help="RGB backbone name", default=None)
    parser.add_argument("--event_backbone", type=str, help="Event backbone name", default=None)
    parser.add_argument("--embed_dim", type=int, help="Embedding dimension", default=None)
    parser.add_argument("--input_size", type=int, help="Input size", default=None)
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--outputs", type=list, help="Dropout rate", default=None)
    parser.add_argument("--model-backbone-pretrained_weights", type=str, help="Path to pretrained weights", default=None, dest="model-backbone-pretrained_weights")

    # Optimizer parameters
    parser.add_argument("--optimizer-name", type=str, help="Optimizer name", default=None, dest="optimizer-name")
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
    parser.add_argument("--checkpoint_interval_epochs", type=int, help="Checkpoint interval in epochs", default=None)
    # Scheduler parameters
    parser.add_argument("--scheduler-name", type=str, help="Scheduler name", default=None, dest="scheduler-name")
    parser.add_argument("--factor", type=float, help="Scheduler factor", default=None)
    parser.add_argument("--patience", type=int, help="Scheduler patience", default=None)
    parser.add_argument("--monitor", type=str,  help="Scheduler monitor metric", default=None)
    parser.add_argument("--mode", type=str, help="Scheduler mode (min or max)", default=None)

    # Logging parameters
    parser.add_argument("--logger-name", type=str, help="Logger name", default=None, dest="logger-name")
    parser.add_argument("--project", type=str,  help="Wandb project name", default=None)
    parser.add_argument("--entity", type=str, help="Wandb entity name", default=None)

    args = parser.parse_args()

    cfg = None
    with open(vars(args)['config_path']) as file:
        cfg = yaml.safe_load(file)

    pretrained_checkpoint = None
    # If a checkpoint path is provided, load the checkpoint

    if cfg.get("checkpoint_path", None) is not None:
        if args.checkpoint_path is not None:
            cfg["checkpoint_path"] = args.checkpoint_path
        #root_dir = os.path.dirname(os.path.realpath(__file__))
        
        checkpoint_path = cfg["checkpoint_path"]
        print(f"Loading pretrained model from {checkpoint_path}")
        try:
            pretrained_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # Load the saved config (if available) to verify architecture consistency
            saved_cfg = pretrained_checkpoint.get('config', {})
            # You can compare saved_cfg['model'] with cfg['model'] for consistency.
            # For example:
            if saved_cfg.get('model', None) and not deep_dict_compare(saved_cfg['model'], cfg['model']):
                print("Warning: The provided config differs from the saved model's config. Proceeding with loaded model parameters.") #TODO: maybe use an assert instead.
            elif DEBUG>=1: print(f"Loading model from:{pretrained_checkpoint}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            pretrained_checkpoint = None
            cfg["checkpoint_path"] = None

    
    assert cfg
    # Update cfg with parsed arguments
    for k, v in vars(args).items():
        
        if k != "config_path" and v is not None:
            if '-' in k:
                key = k.split('-')[0]
                subkey = k.split('-')[1]
                if len(k.split('-')) > 2:
                    subsubkey = k.split('-')[2]
                    if key in cfg and subkey in cfg[key] and subsubkey in cfg[key][subkey]:
                        cfg[key][subkey][subsubkey] = v
                elif key in cfg and subkey in cfg[key]:
                    cfg[key][subkey] = v
            else:
                find_and_modify(cfg,k,v)
        else:
            # Ensure the value from cfg is used if the argument is not provided
            v_cfg = getattr(cfg, k, None)
            if v_cfg is not None:
                setattr(cfg, k, v_cfg)
    return cfg, pretrained_checkpoint



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

    cfg, pretrained_checkpoint = parse_arguments()

    if DEBUG>0:
        print("Configuration parameters:")
        print_cfg_params(cfg)

    ### Model ###
    assert 'bb_num_classes' in cfg['dataset'], "Error - number of classes need to be specified in unimodal training"
    model = build_model_from_cfg(cfg)

    ### Loss ###

    criterion, learnable = loss.build_from_config(cfg)

    ### Optimizer ###
    
    opti = optimizer.build_from_config(model, criterion if learnable else None, cfg)
    
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
    
    train_ds, test_ds = dataset_builder.build_from_config(cfg['dataset'])
    # Dataloader (CMDA)
    assert 'dataset' in cfg.keys(), " 'dataset' params list missing from config file"
    assert 'batch_size' in cfg['dataset'].keys(), " specify 'batch_size' dataset param"
    num_workers=2
    if 'num_workers' in cfg['dataset'].keys(): 
        num_workers = int(cfg['dataset']['num_workers'])
        
    train_dl = DataLoader(train_ds, batch_size=cfg['dataset']['batch_size'], num_workers=num_workers, shuffle=True, collate_fn=collate_ssl, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg['dataset']['batch_size'], num_workers=num_workers, shuffle=False, collate_fn=collate_ssl, pin_memory=True) if test_ds is not None else None
    
    wandb_log = False
    aug = "aug" if cfg['dataset'].get('use_augmentations', False) else  "noaug"
    schedl_name = cfg['scheduler']['name'] if 'name' in cfg['scheduler'].keys() else ""
    run_name = cfg['model']['backbone']['name'] if ('name' in cfg['model']['backbone'].keys() and cfg['model']['backbone']['name'] != '') else model.get_name()
    run_name = f"{run_name}_{cfg['optimizer']['name']}_{schedl_name}_{aug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'logger' in cfg.keys() and 'name' in cfg['logger'].keys():
        if(cfg['logger']['name'] == 'wandb'):
            wandb_cfg = cfg['logger']
            assert 'project' in wandb_cfg.keys(), "specify 'project' wandb param"
            assert 'entity' in wandb_cfg.keys(), "specify 'entity' wandb param"
            wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], name=run_name, config=cfg,settings=wandb.Settings(init_timeout=600))
            # Log model architecture
            wandb.watch(model)
            wandb_log = True

    device = cfg['device'] if 'device' in cfg.keys() else "cuda"
    model.to(device)    

    # Place this right after model.to(device)
    if pretrained_checkpoint is not None:
        model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        print("Pre-trained model loaded successfully")

    # Scheduler
    schdlr = None
    if 'scheduler' in cfg.keys() and 'name' in cfg['scheduler'].keys() and cfg['scheduler']['name'] is not None:
        schdlr = scheduler_builder(opti, cfg['scheduler'])
        assert schdlr is not None, "Error - scheduler not correctly defined"
    
    # Trainer
    assert 'trainer' in cfg.keys(), "'trainer' params list missing from config file "

    if cfg['dual_modality']:
        trainer = DualModalityTrainer(model, train_dl, opti, criterion, device, cfg, root_folder=dir_path, wandb_log=wandb_log, pretrained_checkpoint=pretrained_checkpoint)
    else:
        trainer = Trainer(model,train_dl, opti, criterion, device,  cfg, root_folder=dir_path, wandb_log=wandb_log, pretrained_checkpoint=pretrained_checkpoint, scheduler=schdlr)
    in_size = cfg['model']['backbone']['input_size']
    evaluator = CityscapesEvaluator(test_dl, img_size=(in_size, in_size), confthre=0.3, nmsthre=0.6, num_classes=cfg['dataset']['bb_num_classes'], device=device)
    trainer.train(evaluator=evaluator)
