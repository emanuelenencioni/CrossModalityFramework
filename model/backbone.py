import torchvision.models as models
import torch
import torch.nn as nn
import timm
from utils.helpers import DEBUG
import os

class UnimodalBackbone(nn.Module):
    def __init__(self, backbone=None, pretrained=True, pretrained_weights=None,
                 embed_dim=256, img_size=224, model_name='',outputs=["projector"], output_indices = None, old_projector=True):
        """
        Args:
            bacbkone: Timm model name or custom module
            embed_dim: Shared latent space dimension
        """
        super().__init__()
        if model_name == '':
            self.name = backbone
        else:
            self.name = model_name
        self.img_size = img_size
        use_multiple_features = True if output_indices is not None else False
        if DEBUG>=1: print (f"Using multiple features: {use_multiple_features}")
        self.out_indices = output_indices if use_multiple_features else None
        # TODO add in_chans param as input in cfg. 
        if pretrained_weights is not None and pretrained:
            pretrained = False
            if DEBUG >= 1: print(f"Pretrained weights will be loaded from {pretrained_weights}")
        
        if isinstance(backbone, str):
            if 'resnet' in backbone:
                self.backbone = timm.create_model( backbone, pretrained=pretrained,
                    in_chans=3, num_classes=0,features_only=use_multiple_features, out_indices=output_indices, cache_dir=".cache_dir")
            else:
                self.backbone = timm.create_model( backbone, img_size=img_size, pretrained=pretrained,
                    in_chans=3, num_classes=0,features_only=use_multiple_features, out_indices=output_indices, cache_dir=".cache_dir")  # Assume 5-channel voxel grid
        else:
            self.backbone = backbone
        if pretrained_weights is not None: self.load_pretrained_weights(pretrained_weights)

        if use_multiple_features:
            if DEBUG>=1: print(f"\033[93m"+"WARNING: Using multiple features from backbone. Make sure to set out_indices correctly."+"\033[0m")
            self.feature_info = self.backbone.feature_info
    
        self.projector = nn.Sequential(
            nn.Linear(self.get_feature_output_dim(old_projector), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.outputs = outputs


    def load_pretrained_weights(self, pretrained_weights):
        """
        Load pretrained weights into the backbone.
        Args:
            pretrained_weights (str): Path to the pretrained weights file.
        """
        root_dir = os.path.dirname(os.path.abspath(__file__)).replace('model', '')
        pretrained_weights = os.path.join(root_dir, pretrained_weights)
        if DEBUG >= 1: print(f"Loading pretrained weights from {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}, total: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}, total: {len(unexpected_keys)}")

    def get_feature_output_dim(self, old_projector=False):
        
        dummy_in  = torch.randn(1, 3, self.img_size, self.img_size)
        if self.out_indices is None:
            return self.backbone.forward_features(dummy_in).flatten(start_dim=1).shape[-1]
        else:
            if old_projector:
                return self.backbone(dummy_in)[-1].shape[-1]
            else:
                return self.backbone(dummy_in)[-1].flatten(start_dim=1).shape[-1]

    def _get_features(self, feat):
        """
        Processes the feature tensor and returns a dictionary of specified outputs.
        Args:
            feat (torch.Tensor): The input feature tensor.
        Returns:
            dict: A dictionary containing the specified features. Possible forms of features include:
                - "preflatten_feat": The feature tensor before flattening.
                - "flatten_feat": The feature tensor after flattening.
                - "projected_feat": The feature tensor after being passed through the projector.
        """
        out_dict = {}
        last_feat = feat[-1] if isinstance(feat, list) else feat
        if "preflatten_feat" in self.outputs:
            out_dict["preflatten_feat"] = feat.copy()
        flatten_feat = last_feat.flatten(start_dim=1).clone()
        if "flatten_feat" in self.outputs:
            out_dict['flatten_feat'] = flatten_feat
        # if "projected_feat" in self.outputs :
        #     if self.projector is not None:
        #         out_dict["projected_feat"] = self.projector(flatten_feat)
        #     else:
        #         print("\033[93m"+"WARNING: Projector not found"+"\033[0m")

        return out_dict

    def get_name(self): return self.name


    def forward(self, x):
        if self.out_indices is None:
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)
        return self._get_features(x)


class DualModalityBackbone(nn.Module):
    def __init__(self, rgb_backbone, event_backbone, pretrained=True,
                 embed_dim=256, img_size=224, model_name='',outputs=['preflatten_feat', 'flatten_feat', 'projected_feat']):
        """
        Args:
            rgb_backbone: Timm model name or custom module
            event_backbone: Timm model name or custom module
            embed_dim: Shared latent space dimension
        """
        super().__init__()
        if model_name == '':
            self.name = "rgb_"+ rgb_backbone +"_events_"+ event_backbone
        else:
            self.name = model_name
        self.img_size = img_size

        self.rgb_backbone = UnimodalBackbone(rgb_backbone, embed_dim=embed_dim, img_size=img_size, 
                             model_name=model_name, outputs=outputs)        
        self.event_backbone = UnimodalBackbone(event_backbone, embed_dim=embed_dim, img_size=img_size, 
                             model_name=model_name, outputs=outputs)

        self.rgb_projector = nn.Sequential(
            nn.Linear(self.rgb_backbone.get_feature_output_dim(), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.event_projector = nn.Sequential(
            nn.Linear(self.event_backbone.get_feature_output_dim(), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.outputs = outputs
    
    def forward(self, rgb, events):
        """
        Forward pass for the DualModalityBackbone.
        Args:
            rgb (torch.Tensor): Input tensor for the RGB backbone.
            events (torch.Tensor): Input tensor for the Event backbone.
        Returns:
            dict: Dictionary containing the features for the RGB backbone.
            dict: Dictionary containing the features for the Event backbone.
        """
        return self.rgb_backbone.forward(rgb), self.event_backbone(events)

    def get_grad_norm(self):
        rgb_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach())
                                    for p in self.rgb_backbone.parameters()
                                    if p.grad is not None]))
        event_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach())
                                    for p in self.event_backbone.parameters()
                                    if p.grad is not None]))
        return rgb_grad_norm, event_grad_norm


    def get_weights_norm(self):
        rgb_weights_norm = torch.norm(torch.stack([torch.norm(p.detach())
                                    for p in self.rgb_backbone.parameters()]))
        event_weights_norm = torch.norm(torch.stack([torch.norm(p.detach())
                                    for p in self.event_backbone.parameters()]))
        return rgb_weights_norm, event_weights_norm

    def get_name(self): return self.name


# if __name__ == "__main__":
#     model = BackboneAdapter(models.resnet18(),224,3,True)
#     print(model)
#     x = torch.randn(1,3,224,224)
#     print(model(x).size())
