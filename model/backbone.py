import torchvision.models as models
import torch
import torch.nn as nn
import timm


# TODO: handle extern model from torch or timm ->  Give guidelines to a basic compatible structure
class Backbone(nn.Module):
    """
    Initializes the Backbone model.
    Args:
        model (torchvision.models): A pre-trained torchvision model. (at least for now)
        pretrained (bool): If True, loads pre-trained weights for the model. Defaults to True.
        input_dim: squared dim for input frame
        input_ch: input channel
    """
    def __init__(self):
        super().__init__()
        self.outputs = None
        self.projector = None
    

    

class UnimodalBackbone(Backbone):
    def __init__(self, backbone=None, pretrained=True,
                 embed_dim=256, img_size=224, model_name='',outputs=["projector"]):
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
        
        if isinstance(backbone, str):
            if "vit" in backbone: #TODO: trovare metodo migliore per capire se è un vit.
                self.backbone = timm.create_model( backbone, img_size=img_size, pretrained=pretrained,
                    in_chans=3, num_classes=0 )  # Assume 5-channel voxel grid
            else:
                self.backbone = timm.create_model(backbone, pretrained=pretrained,
                    in_chans=3, num_classes=0 )  # Assume 5-channel voxel grid
        else:
            self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(self.get_feature_output_dim(), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.outputs = outputs

    def get_feature_output_dim(self):
        return self.backbone.forward_features(torch.randn(1, 3, self.img_size, self.img_size)).flatten().shape[-1]

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
        if "preflatten_feat" in self.outputs:
            out_dict["preflatten_feat"] = feat
        feat = feat.flatten(start_dim=1)
        if "flatten_feat" in self.outputs:
            out_dict['flatten_feat'] = feat
        if "projected_feat" in self.outputs :
            if self.projector is not None:
                out_dict["projected_feat"] = self.projector(feat)
            else:
                print("\033[93m"+"WARNING: Projector not found"+"\033[0m")

        return out_dict

    def get_model_name(self):
        return self.name


    def forward(self, x):
        x = self.backbone.forward_features(x)
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

    def get_model_name(self):
        return self.name


if __name__ == "__main__":
    model = BackboneAdapter(models.resnet18(),224,3,True)
    print(model)
    x = torch.randn(1,3,224,224)
    print(model(x).size())
