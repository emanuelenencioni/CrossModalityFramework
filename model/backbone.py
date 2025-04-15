import torchvision
import torch






import torchvision
import torch
import torch.nn as nn


class BackboneWrapper(nn.Module):
    """
    Wraps a pre-trained torchvision model to be used as a backbone.

    Args:
        backbone_name (str): The name of the torchvision model to use (e.g., 'resnet18', 'vgg16').
        pretrained (bool): Whether to load pre-trained weights.
        return_layers (list, optional): List of layer names to return as features. If None, returns the output of the last layer.
                                        Defaults to None.  If specified, the output will be a dictionary of features, keyed by layer name.
        modify_backbone (callable, optional):  A function that takes the backbone model as input and modifies it.
                                                This is useful for removing or replacing layers. Defaults to None.
        trainable_layers (list or int, optional): Number of trainable layers from the end.  If None, all layers are trainable.
                                                If an int, the last 'trainable_layers' layers are trainable. If a list of strings,
                                                the layers with those names are trainable. Defaults to None (all trainable).

    Example:
        # Using ResNet18 as a backbone, returning features from layer4:
        backbone = BackboneWrapper(backbone_name='resnet18', pretrained=True, return_layers=['layer4'])

        # Freeze all layers except the last 2:
        backbone = BackboneWrapper(backbone_name='resnet50', pretrained=True, trainable_layers=2)

        # Modify the backbone to remove the avgpool and fc layers:
        def modify_resnet(model):
            model.avgpool = nn.Identity()
            model.fc = nn.Identity()
            return model

        backbone = BackboneWrapper(backbone_name='resnet50', pretrained=True, modify_backbone=modify_resnet)

        # Using VGG16 as a backbone, returning features from multiple layers:
        backbone = BackboneWrapper(backbone_name='vgg16', pretrained=True, return_layers=['features.23', 'features.30'])


    """
    def __init__(self, backbone_name, pretrained=True, return_layers=None, modify_backbone=None, trainable_layers=None):
        super().__init__()

        try:
            backbone = getattr(torchvision.models, backbone_name)(pretrained=pretrained)
        except AttributeError:
            raise ValueError(f"Invalid backbone name: {backbone_name}.  Must be a torchvision.models attribute.")

        if modify_backbone is not None:
            backbone = modify_backbone(backbone)

        self.backbone = backbone
        self.return_layers = return_layers

        self._set_trainable_layers(trainable_layers)


        if return_layers is not None:
            if isinstance(return_layers, list):
                self.returned_layers = return_layers
            else:
                self.returned_layers = [return_layers]

            self.backbone = self._feature_extraction_net(self.backbone, self.returned_layers)

    def _set_trainable_layers(self, trainable_layers):
        """
        Freezes or unfreezes layers based on the trainable_layers parameter.
        """

        if trainable_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif isinstance(trainable_layers, int):
            # Freeze all layers up to the specified number of layers from the end
            layer_count = 0
            for name, param in self.backbone.named_parameters():
                layer_count += 1

            trainable_start = layer_count - trainable_layers

            current_layer = 0
            for name, param in self.backbone.named_parameters():
                current_layer += 1
                if current_layer <= trainable_start:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif isinstance(trainable_layers, list):
            # Only train the specified layers
            for name, param in self.backbone.named_parameters():
                if name in trainable_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise ValueError("trainable_layers must be None, an integer, or a list of strings.")


    def _feature_extraction_net(self, model, return_layers):
        """
        Creates a new network that outputs features from the specified layers.
        """

        if not isinstance(return_layers, list):
            return_layers = [return_layers]

        new_model = nn.ModuleDict()
        modules = list(model.named_children())

        current_feature_idx = 0
        for name, module in modules:
            new_model[name] = module
            if name in return_layers:
                current_feature_idx +=1

        return new_model


    def forward(self, x):
        """
        Performs a forward pass through the backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor or dict: The output tensor or a dictionary of feature tensors, depending on whether return_layers is specified.
        """

        if self.return_layers is None:
            return self.backbone(x)
        else:
            features = {}
            x_ = x
            for name, module in self.backbone.named_children():
                x_ = module(x_)
                if name in self.returned_layers:
                    features[name] = x_
            return features
