from model.backbone import UnimodalBackbone
from model.yolox_head import YOLOXHead
import torch
import torch.nn as nn
from loguru import logger

class Resnet50_yolox(nn.Module):
    """
    A complete object detection model that combines a resnet50 backbone with a YOLOX head.

    Args:
        backbone (dict): Backbone configuration dictionary.
        name (str): Model name.
        head (dict): Head configuration dictionary.
        freeze (bool): If True, freeze backbone and head parameters.
    """
    def __init__(self,  backbone: dict, name: str="resnet50_yolox", head: dict={'name': 'yolox_head','num_classes': 8}, freeze: bool=False):
        super().__init__()
        assert 'output_indices' in backbone, "Error - output_indices must be specified in the backbone config"
        out_indices = backbone['output_indices']
        # 1. Init backbone to extract 3 feat lvls
        self.backbone = UnimodalBackbone(
            backbone=backbone.get('name', 'resnet50'),
            pretrained=backbone.get('pretrained', True),
            pretrained_weights=backbone.get("pretrained_weights", None),
            img_size=backbone.get('input_size'),
            outputs=["preflatten_feat", "projected_feat", "flatten_feat"],
            output_indices=backbone.get('output_indices')  # Default to last 3 layers if not specified
        )
        

        self.name = name
        
        feature_info = self.backbone.feature_info
        in_channels = [info['num_chs'] for info in feature_info if info['index'] in out_indices]
        strides = [info['reduction'] for info in feature_info if info['index'] in out_indices]
        
        logger.info(f"Backbone '{backbone['name']}' initialized.")
        logger.info(f"  - Extracted strides: {strides}")
        logger.info(f"  - Input channels for Head: {in_channels}")
        
        # 3. Initialize the YOLOXHead with correct parameters
        self.head = YOLOXHead(
            num_classes=head['num_classes'],
            in_channels=in_channels,
            strides=strides,
            losses_weights=head.get('losses_weights', [5.0, 1.0, 1.0, 1.0])
            
        )

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.warning("Backbone frozen. Its weights will not be updated during training.")
            for param in self.head.parameters():
                param.requires_grad = False
        elif backbone.get('freeze', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.warning("Backbone frozen. Its weights will not be updated during training.")
        elif head.get('freeze', False):
            for param in self.head.parameters():
                param.requires_grad = False
            logger.warning("Head frozen. Its weights will not be updated during training.")

    def get_name(self): return self.name

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Performs the complete forward pass.

        Args:
            x (torch.Tensor): Input image.
            targets (torch.Tensor, optional): Ground truth for training. If None, the model is in inference mode.

        Returns:
            dict: A dictionary containing:
                - 'backbone_features': Dict with all backbone features (e.g., 'preflatten_feat', 'flattened_feat', etc.)
                - 'head_outputs': Bounding box predictions/detections
                - 'total_loss': Total loss scalar (only during training)
                - 'losses': Dict of individual loss components (only during training)
                    - 'iou_loss': IoU loss
                    - 'obj_loss': Objectness loss
                    - 'cls_loss': Classification loss
                    - 'l1_loss': L1 loss (if applicable)
        """
        # Get backbone features (already a dict)
        backbone_features = self.backbone(x)
        
        # Get head outputs
        head_output = self.head(backbone_features["preflatten_feat"], labels=targets)
        
        # Build output dictionary
        output = {
            'backbone_features': backbone_features,  # Dict: {'preflatten_feat': [...], 'flattened_feat': [...], ...}
        }
        head_predictions, total_loss, losses_dict = head_output
        if targets is not None:
            # Training mode: head_output is (outputs, total_loss, losses_dict)
            
            output.update({
                'head_outputs': head_predictions,
                'total_loss': total_loss,
                'losses': losses_dict  # Dict: {'iou_loss': ..., 'obj_loss': ..., 'cls_loss': ..., 'l1_loss': ...}
            })
        else:
            # Inference mode: head_output is just the predictions
            output['head_outputs'] = head_predictions
        
        return output
