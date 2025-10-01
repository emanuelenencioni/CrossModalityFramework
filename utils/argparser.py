import argparse
from .helpers import DEBUG
import torch
import yaml

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
    parser.add_argument("--use_augmentations", type=bool, help="Use augmentations", default=None)

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

    
    if DEBUG>0:
        print("Configuration parameters:")
        print_cfg_params(cfg)

    return cfg, pretrained_checkpoint