import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import timm  # For flexible backbones like ResNet/ViT
from dataset.dsec import DSECDataset

from training.loss import CLIP_loss

from tqdm import tqdm

class DualModalityBackbone(nn.Module):
    def __init__(self, 
                 rgb_backbone='resnet18', 
                 event_backbone='resnet18', 
                 pretrained=True,
                 embed_dim=256):
        """
        Args:
            rgb_backbone: Timm model name or custom module
            event_backbone: Timm model name or custom module
            embed_dim: Shared latent space dimension
        """
        super().__init__()
        # RGB Backbone
        if isinstance(rgb_backbone, str):
            self.rgb_backbone = timm.create_model(
                rgb_backbone, pretrained=pretrained, 
                in_chans=3, num_classes=0)  # Remove classifier
        else:
            self.rgb_backbone = rgb_backbone  # Custom module
        
        # Event Backbone (same architecture by default)
        if isinstance(event_backbone, str):
            self.event_backbone = timm.create_model(
                event_backbone, pretrained=pretrained,
                in_chans=3, num_classes=0)  # Assume 5-channel voxel grid
        else:
            self.event_backbone = event_backbone
        
        # Shared Projector (for contrastive loss)
        self.projector = nn.Sequential(
            nn.Linear(self._get_output_dim(), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def _get_output_dim(self):
        """Infer feature dimension from backbone"""
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 224, 224)
            dummy_event = torch.randn(1, 3, 224, 224)
            return self.rgb_backbone(dummy_rgb).shape[-1] + \
                   self.event_backbone(dummy_event).shape[-1]

    def forward(self, rgb, event):
        """
        Args:
            rgb: (B, 3, H, W)
            event: (B, 5, H, W) - Voxel grid or time-binned tensor
        Returns:
            (rgb_proj, event_proj): Projected features in shared space
        """
        rgb_feat = self.rgb_backbone(rgb)
        event_feat = self.event_backbone(event)
        
        # Concatenate features
        combined = torch.cat([rgb_feat, event_feat], dim=1)
        projected = self.projector(combined)
        
        split_idx = projected.shape[1] // 2
        return projected[:, :split_idx], projected[:, split_idx:]


# TODO: Add modality-specific augmentations later
def get_augmentation(modality):
    """Placeholder for future augmentation pipeline"""
    return lambda x: x  # Identity for now

# Training Loop
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader,desc="Training net"):
        #batch_t =  TODO
        rgbs = torch.stack([batch[i]["image"]for i in range(len(batch))]).to(device)
        events = torch.stack([ batch[i]["events_vg"] for i in range(len(batch))]).to(device)
        # Forward pass
        #print(events.size())
        rgb_proj, event_proj = model(rgbs, events)
        
        # Compute loss
        loss = criterion(rgb_proj, event_proj)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    
    return total_loss / len(dataloader)

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
    optimizer = torch.optim.Adam({model.parameters(),criterion.parameters()}, lr=1e-4)

    # Dataloader (CMDA)
    events_bins_5_avg_1 = False
    if events_bins_5_avg_1:
        events_bins = 1
        events_clip_range = None  # (1.0, 1.0)
    else:
        events_bins = 1
        events_clip_range = None
    dataset = DSECDataset(dataset_txt_path='/home/emanuele/Documenti/Codice/framework_VMR/dataset/night_dataset_warp.txt',
                           outputs={'events_vg', 'img_metas', 'BB','image'},
                           events_bins=events_bins, events_clip_range=events_clip_range,
                           events_bins_5_avg_1=events_bins_5_avg_1)

    
    dataloader = DataLoader(dataset, batch_size=5, sampler=None,num_workers=4, collate_fn=collate_fn)

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)    
    
    for epoch in range(100):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")