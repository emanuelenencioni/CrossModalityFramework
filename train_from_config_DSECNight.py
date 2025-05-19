import yaml
import sys
import os
from training import optimizer,loss
from training.ssl import train_ssl
from model.backbone import DualModalityBackbone
from torch.utils.data import DataLoader
from dataset.dsec import DSECDataset, collate_ssl
import torch
import wandb



if __name__ == "__main__":
    if(len(sys.argv) < 2):sys.exit("Error - use python train_from_config.py path_to_yaml_file")

    with open(sys.argv[1]) as file:
        cfg = yaml.safe_load(file)
    assert file != None, "Error opening file - file not found"

    # Configuration
    model = DualModalityBackbone(
        rgb_backbone='resnet18',
        event_backbone='resnet18',  # Could use 'vit_tiny_patch16_224' for asymmetry
        embed_dim=128,
        img_size=512
    )

    # Loss
    criterion, learnable = loss.build_from_config(cfg['loss']['name'])
    if learnable:
        params = list(model.parameters()) + list(criterion.parameters())
    else:
        params = model.parameters()

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
                           outputs={'events_vg', 'img_metas','image'},
                           events_bins=events_bins, events_clip_range=events_clip_range,
                           events_bins_5_avg_1=events_bins_5_avg_1)
    
    # Dataloader (CMDA)
    dataloader = DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True, collate_fn=collate_ssl)
    
    wandb_log = False
    run = None
    if(cfg['logger']['name'] == 'wandb'):
        wandb_cfg = cfg['logger']
        wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfg,settings=wandb.Settings(init_timeout=600))
        # Log model architecture
        wandb.watch(model)
        wandb_log = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)    

    for epoch in range(1):
        loss_v = train_ssl(model, dataloader, opti, criterion, device, wandb_log=wandb_log)
        print(f"Epoch {epoch}, Loss: {loss_v:.4f}")
