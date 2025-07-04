import yaml
import sys
import os
from training import optimizer,loss
from training.ssl import TrainSSL
from model.backbone import DualModalityBackbone
from torch.utils.data import DataLoader
from dataset.dsec import DSECDataset, collate_ssl
from datetime import datetime
import torch
import wandb

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

if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Error - use python train_from_config.py path_to_yaml_file"

    with open(sys.argv[1]) as file:
        cfg = yaml.safe_load(file)
    assert file != None, "Error opening file - file not found"
    
    # Configuration
    modality, backbone = check_backbone_params(cfg)
    if modality:
        model = DualModalityBackbone(rgb_backbone=cfg['backbone']['rgb_backbone'],
                    event_backbone=cfg['backbone']['event_backbone'],
                    embed_dim=cfg['backbone']['embed_dim'],
                    img_size=cfg['backbone']['input_size']
        )
    else:
        model = unimodalBackbone(backbone, embed_dim=cfg['backbone']['embed_dim'],
                    img_size=cfg['backbone']['input_size'])

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
    # Dataloader (CMDA)
    events_bins_5_avg_1 = False
    if events_bins_5_avg_1:
        events_bins = 1
        events_clip_range = None  # (1.0, 1.0)
    else:
        events_bins = 1
        events_clip_range = None
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset = DSECDataset(dataset_txt_path=dir_path+'/dataset/night_dataset.txt',
                           outputs={'events_vg','image'},
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
    run_name = cfg['backbone']['name'] if ('name' in cfg['backbone'].keys() and cfg['backbone']['name'] != '') else model.get_model_name()
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
    
    trainer = TrainSSL(model, dataloader, opti, criterion, device, cfg['trainer'], root_folder=dir_path, wandb_log=wandb_log)
    trainer.train()
