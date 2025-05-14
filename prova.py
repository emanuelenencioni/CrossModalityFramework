import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import timm  # For flexible backbones like ResNet/ViT
from dataset.dsec import DSECDataset
from model.backbone import DualModalityBackbone
from training.ssl import train_ssl
from training.loss import CLIP_loss

from tqdm import tqdm


# TODO: Add modality-specific augmentations later
def get_augmentation(modality):
    """Placeholder for future augmentation pipeline"""
    return lambda x: x  # Identity for now



def collate_fn(batch):
    return batch


if __name__ == "__main__":
        # Configuration
    model = DualModalityBackbone(
        rgb_backbone='resnet18',
        event_backbone='resnet18',  # Could use 'vit_tiny_patch16_224' for asymmetry
        embed_dim=128
    )

    # Loss & Optimizer
    criterion = CLIP_loss()
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

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

    
    dataloader = DataLoader(dataset, batch_size=5, sampler=None,num_workers=8, collate_fn=collate_fn)

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)    
    
    for epoch in range(1):
        loss = train_ssl(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")