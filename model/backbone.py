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
    def _get_features(self, feat, projector=None):
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
            if  projector is not None:
                out_dict["projected_feat"] =  projector(feat)
            elif self.projector is not None:
                out_dict["projected_feat"] = self.projector(feat)
            else:
                print("\033[93m"+"WARNING: Projector not found"+"\033[0m")

        return out_dict

class unimodalBackbone(Backbone):
    def __init__(self, backbone=None, pretrained=True,
                 embed_dim=256, img_size=224, model_name='',outputs=["projector"]):
        """
        Args:
            bacbkone: Timm model name or custom module
            embed_dim: Shared latent space dimension
        """
        super().__init__()
        if model_name == '':
            self.name = event_backbone
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

        # Shared Projector (for contrastive loss) # TODO make the user choose it
        self.projector = nn.Sequential(
            nn.Linear(self._get_output_dim(), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        if event_backbone is None or rgb_backbone is None:
            self.unimodal_training = True
        else:
            self.unimodal_training = False

        self.outputs = outputs
    


class DualModalityBackbone(Backbone):
    def __init__(self, rgb_backbone=None, event_backbone=None, pretrained=True,
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
        # RGB Backbone
        if isinstance(rgb_backbone, str):
            self.rgb_backbone = timm.create_model(
                rgb_backbone, pretrained=pretrained, 
                in_chans=3, num_classes=0)  # Remove classifier
        else:
            self.rgb_backbone = rgb_backbone  # Custom module
        
        # Event Backbone (same architecture by default)
        if isinstance(event_backbone, str):
            if "vit" in event_backbone: #TODO: trovare metodo migliore per capire se è un vit.
                self.event_backbone = timm.create_model(
                    event_backbone, img_size=img_size, pretrained=pretrained,
                    in_chans=3, num_classes=0 )  # Assume 5-channel voxel grid
            else:
                self.event_backbone = timm.create_model(
                    event_backbone, pretrained=pretrained,
                    in_chans=3, num_classes=0 )  # Assume 5-channel voxel grid
        else:
            self.event_backbone = event_backbone

        for i, layer in enumerate(self.event_backbone.parameters()):
            if isinstance(layer, nn.Flatten):
                self.event_backbone.model[i] = nn.Identity()

        # Shared Projector (for contrastive loss) # TODO make the user choose it
        self.rgb_projector = nn.Sequential(
            nn.Linear(self.rgb_backbone.forward_features(torch.randn(1, 3, self.img_size, self.img_size)).flatten().shape[-1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.event_projector = nn.Sequential( #TODO check for validity of the dummy dim.
            nn.Linear(self.event_backbone.forward_features(torch.randn(1, 3, self.img_size, self.img_size)).flatten().shape[-1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.outputs = outputs
    
    def _get_output_dim(self):  
        """Infer feature dimension from backbone"""
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, self.img_size, self.img_size)
            dummy_event = torch.randn(1, 3, self.img_size, self.img_size)
            return self.rgb_backbone.forward_features(dummy_rgb).shape[-1]    + self.event_backbone(dummy_event).shape[-1] 
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
        rgb_feat = self.rgb_backbone.forward_features(rgb)
        event_feat = self.event_backbone.forward_features(events)
        return self._get_features(rgb_feat, self.rgb_projector), self._get_features(event_feat, self.event_projector)

    def get_grad_norm(self):
        """
        Returns the L2 norm of the gradients of the RGB and Event backbones.
        """
        rgb_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach())
                                     for p in self.rgb_backbone.parameters()
                                     if p.grad is not None]))
        event_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach())
                                     for p in self.event_backbone.parameters()
                                     if p.grad is not None]))
        return rgb_grad_norm, event_grad_norm


    def get_weights_norm(self):
        """
        Returns the L2 norm of the weights of the RGB and Event backbones.
        """
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
