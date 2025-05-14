import torchvision.models as models
import torch
import torch.nn as nn
import timm


# TODO: gestire modelli esterni a torch -> dare linee guida per struttura basic.
class BackboneAdapter(nn.Module):
    """
    Initializes the Backbone model.
    Args:
        model (torchvision.models): A pre-trained torchvision model. (at least for now)
        pretrained (bool): If True, loads pre-trained weights for the model. Defaults to True.
        input_dim: squared dim for input frame
        input_ch: input channel
    """
    def __init__(self, model, input_ch, torch_pretrained=False, custom_weights=""):
        super().__init__()
        assert model != None, "Insert a valid model"
        if torch_pretrained:
            model_name = model.__class__.__name__.lower()
            print(model_name)
            try:
                
                self.backbone = getattr(models, model_name)(pretrained=True)
                
            except Exception as e:
                print(f"Error loading torch pretrained weights: {e}")
                self.backbone = model
        else:
            self.backbone = model

        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()

    def forward(self, x):
        return self.backbone(x)



class DualModalityBackbone(nn.Module):
    def __init__(self, 
                 rgb_backbone=None, 
                 event_backbone=None, 
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
            return self.rgb_backbone(dummy_rgb).shape[-1] + self.event_backbone(dummy_event).shape[-1]

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












if __name__ == "__main__":
    model = BackboneAdapter(models.resnet18(),224,3,True)
    print(model)
    x = torch.randn(1,3,224,224)
    print(model(x).size())
