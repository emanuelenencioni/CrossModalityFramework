import torch
import torch.nn as nn
from model.rf_detr import LWDETR, SetCriterion
from model.rf_detr.models import build_backbone, build_transformer, build_matcher
from loguru import logger
from utils.helpers import DEBUG


class Rfdetrwrapper(nn.Module):
    """
    Wrapper for RF-DETR to work with CrossModalityFramework builder.py.
    
    Expected config structure:
    model:
      name: "rfdetrwrapper"
      backbone:
        type: "resnet50"      # Backbone type
        embed_dim: 256        # Hidden dimension
        input_size: 640       # Input image size
        num_feature_levels: 4
        pretrained_encoder: true
        freeze_encoder: false
        # ... see config file for all backbone parameters
      head:
        num_classes: 80
        num_queries: 300
        aux_loss: true
        group_detr: 1
        two_stage: false
      transformer:
        sa_nheads: 8          # Self-attention heads
        ca_nheads: 8          # Cross-attention heads
        dec_layers: 6
        dim_feedforward: 1024
        dropout: 0.1
        dec_n_points: 4
      matcher:
        set_cost_class: 1.0
        set_cost_bbox: 5.0
        set_cost_giou: 2.0
      loss:
        cls_loss_coef: 1.0
        bbox_loss_coef: 5.0
        giou_loss_coef: 2.0
        eos_coef: 0.1
    """
    
    def __init__(self, backbone, head, name="rfdetrwrapper", **kwargs):
        """
        Initialize RF-DETR wrapper compatible with builder.py.
        
        Args:
            backbone (dict): Backbone configuration with keys:
                - type: backbone type (resnet50, resnet101, dinov2, etc.)
                - embed_dim: hidden dimension
                - input_size: input image size
                - dilation: whether to use dilation
                - num_feature_levels: number of feature levels
            head (dict): Head configuration with keys:
                - num_classes: number of classes
                - num_queries: number of queries
                - aux_loss: whether to use auxiliary loss
                - group_detr: group detr parameter
                - two_stage: whether to use two-stage
            name (str): Model name
            **kwargs: Additional parameters including transformer, matcher, loss configs
        """
        super().__init__()
        
        # Extract configurations
        self.backbone_cfg = backbone
        self.head_cfg = head
        self.transformer_cfg = kwargs.get('transformer', {})
        self.matcher_cfg = kwargs.get('matcher', {})
        self.loss_cfg = kwargs.get('loss', {})
        
        self.num_classes = head['num_classes']
        self.num_queries = head.get('num_queries', 300)
        
        if DEBUG >= 1:
            logger.info(f"Initializing RF-DETR wrapper with {self.num_classes} classes")
        
        # Build components
        self.rf_backbone = self._build_backbone()
        self.transformer = self._build_transformer()
        
        # Build LWDETR model
        self.model = LWDETR(
            backbone=self.rf_backbone,
            transformer=self.transformer,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            aux_loss=head.get('aux_loss', True),
            group_detr=head.get('group_detr', 1),
            two_stage=head.get('two_stage', False),
            segmentation_head=head.get('segmentation_head', None)
        )
        
        # Build criterion
        class MatcherArgs:
            set_cost_class = self.matcher_cfg.get('set_cost_class', 1)
            set_cost_bbox = self.matcher_cfg.get('set_cost_bbox', 5)
            set_cost_giou = self.matcher_cfg.get('set_cost_giou', 2)
            focal_alpha = self.matcher_cfg.get('focal_alpha', 0.25)
            segmentation_head = head.get('segmentation_head', None)
            mask_ce_loss_coef = self.loss_cfg.get('mask_ce_loss_coef', 1.0)
            mask_dice_loss_coef = self.loss_cfg.get('mask_dice_loss_coef', 1.0)
            mask_point_sample_ratio = self.matcher_cfg.get('mask_point_sample_ratio', 0.75)
        
        matcher = build_matcher(MatcherArgs())
        
        weight_dict = {
            'loss_ce': self.loss_cfg.get('cls_loss_coef', 1),
            'loss_bbox': self.loss_cfg.get('bbox_loss_coef', 5),
            'loss_giou': self.loss_cfg.get('giou_loss_coef', 2)
        }
        
        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=self.loss_cfg.get('focal_alpha', 0.25),
            losses=['labels', 'boxes', 'cardinality'],
            group_detr=head.get('group_detr', 1),
            sum_group_losses=self.loss_cfg.get('sum_group_losses', False),
            use_varifocal_loss=self.loss_cfg.get('use_varifocal_loss', False),
            use_position_supervised_loss=self.loss_cfg.get('use_position_supervised_loss', False),
            ia_bce_loss=self.loss_cfg.get('ia_bce_loss', False),
            mask_point_sample_ratio=self.matcher_cfg.get('mask_point_sample_ratio', 16)
        )
        
        # For compatibility with your framework
        self.loss_keys = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_cardinality']
        
        if DEBUG >= 1:
            logger.success("RF-DETR wrapper initialized successfully")
    
    def _build_backbone(self):
        """Build backbone based on configuration."""
        from model.rf_detr.models.backbone import build_backbone
        
        backbone_type = self.backbone_cfg.get('type', 'resnet50')
        hidden_dim = self.backbone_cfg.get('embed_dim', 256)
        
        if DEBUG >= 1:
            logger.info(f"Building backbone: {backbone_type} with hidden_dim={hidden_dim}")
        
        # RF-DETR's build_backbone requires all these parameters
        return build_backbone(
            encoder=backbone_type,
            vit_encoder_num_layers=self.backbone_cfg.get('vit_encoder_num_layers', 12),
            pretrained_encoder=self.backbone_cfg.get('pretrained_encoder', True),
            window_block_indexes=self.backbone_cfg.get('window_block_indexes', []),
            drop_path=self.backbone_cfg.get('drop_path', 0.0),
            out_channels=self.backbone_cfg.get('out_channels', [256, 512, 1024, 2048]),
            out_feature_indexes=self.backbone_cfg.get('out_feature_indexes', [1, 2, 3, 4]),
            projector_scale=self.backbone_cfg.get('projector_scale', 1.0),
            use_cls_token=self.backbone_cfg.get('use_cls_token', False),
            hidden_dim=hidden_dim,
            position_embedding=self.backbone_cfg.get('position_embedding', 'sine'),
            freeze_encoder=self.backbone_cfg.get('freeze_encoder', False),
            layer_norm=self.backbone_cfg.get('layer_norm', True),
            target_shape=self.backbone_cfg.get('target_shape', None),
            rms_norm=self.backbone_cfg.get('rms_norm', False),
            backbone_lora=self.backbone_cfg.get('backbone_lora', None),
            force_no_pretrain=self.backbone_cfg.get('force_no_pretrain', False),
            gradient_checkpointing=self.backbone_cfg.get('gradient_checkpointing', False),
            load_dinov2_weights=self.backbone_cfg.get('load_dinov2_weights', None),
            patch_size=self.backbone_cfg.get('patch_size', 16),
            num_windows=self.backbone_cfg.get('num_windows', 0),
            positional_encoding_size=self.backbone_cfg.get('positional_encoding_size', None)
        )
    
    def _build_transformer(self):
        """Build transformer based on configuration."""
        from model.rf_detr.models import build_transformer
        
        embed_dim = self.backbone_cfg.get('embed_dim', 256)
        
        class Args:
            hidden_dim = embed_dim
            dropout = self.transformer_cfg.get('dropout', 0.1)
            sa_nheads = self.transformer_cfg.get('sa_nheads', self.transformer_cfg.get('nheads', 8))
            ca_nheads = self.transformer_cfg.get('ca_nheads', self.transformer_cfg.get('nheads', 8))
            num_queries = self.head_cfg.get('num_queries', 300)
            dim_feedforward = self.transformer_cfg.get('dim_feedforward', 1024)
            dec_layers = self.transformer_cfg.get('dec_layers', 6)
            group_detr = self.head_cfg.get('group_detr', 1)
            two_stage = self.head_cfg.get('two_stage', False)
            num_feature_levels = self.backbone_cfg.get('num_feature_levels', 4)
            dec_n_points = self.transformer_cfg.get('dec_n_points', 4)
            lite_refpoint_refine = self.transformer_cfg.get('lite_refpoint_refine', False)
            decoder_norm = self.transformer_cfg.get('decoder_norm', 'layer_norm')
            bbox_reparam = self.transformer_cfg.get('bbox_reparam', False)
        
        if DEBUG >= 1:
            logger.info(f"Building transformer: dec_layers={Args.dec_layers}, sa_nheads={Args.sa_nheads}, ca_nheads={Args.ca_nheads}")
        
        return build_transformer(Args())
    
    def forward(self, x, targets=None):
        """
        Forward pass compatible with CrossModalityFramework.
        
        Args:
            x: Input tensor [B, C, H, W]
            targets: Target tensor [B, N, 5] where N is max objects
                     Format: [class_id, x1, y1, x2, y2] (xyxy format)
        
        Returns:
            dict with keys:
                - 'backbone_features': dict of backbone features at different scales
                - 'total_loss': scalar tensor (if targets provided)
                - 'losses': dict of individual losses (if targets provided)
                - 'predictions': model predictions (if targets not provided)
        """
        # Forward through model
        outputs = self.model(x)
        
        # Get backbone features for multimodal compatibility
        result = {
            'backbone_features': {
                'preflatten_feat': self.model.backbone.body(x)[-1] if hasattr(self.model.backbone, 'body') else None
            }
        }
        
        if targets is not None:
            # Convert targets to DETR format
            targets_detr = self._convert_targets(targets, x.shape[-2:])
            
            # Compute losses
            loss_dict = self.criterion(outputs, targets_detr)
            
            # Calculate total loss
            total_loss = sum(loss_dict[k] * self.criterion.weight_dict.get(k, 1) 
                           for k in loss_dict.keys() if k in self.criterion.weight_dict)
            
            result['total_loss'] = total_loss
            result['losses'] = loss_dict
        else:
            result['predictions'] = outputs
        
        return result
    
    def _convert_targets(self, targets, img_size):
        """
        Convert targets from [B, N, 5] format to DETR format.
        
        Input format: [class_id, x1, y1, x2, y2] (xyxy in pixel coordinates)
        Output format: list of dicts with 'labels' and 'boxes' (cxcywh normalized [0,1])
        
        Args:
            targets: tensor of shape [B, N, 5]
            img_size: tuple (H, W) of image dimensions
        """
        batch_size = targets.shape[0]
        targets_detr = []
        h, w = img_size
        
        for i in range(batch_size):
            # Filter out padding (assuming class_id < 0 means no object)
            valid_mask = targets[i, :, 0] >= 0
            valid_targets = targets[i][valid_mask]
            
            if len(valid_targets) == 0:
                targets_detr.append({
                    'labels': torch.tensor([], dtype=torch.long, device=targets.device),
                    'boxes': torch.tensor([], dtype=torch.float, device=targets.device).reshape(0, 4)
                })
                continue
            
            labels = valid_targets[:, 0].long()
            boxes_xyxy = valid_targets[:, 1:]
            
            # Normalize boxes to [0, 1]
            boxes_xyxy_norm = boxes_xyxy.clone()
            boxes_xyxy_norm[:, [0, 2]] /= w
            boxes_xyxy_norm[:, [1, 3]] /= h
            
            # Convert xyxy to cxcywh
            boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy_norm)
            
            targets_detr.append({
                'labels': labels,
                'boxes': boxes_cxcywh
            })
        
        return targets_detr
    
    def _xyxy_to_cxcywh(self, boxes):
        """Convert boxes from xyxy to cxcywh format."""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    
    def get_name(self):
        """Return model name for logging."""
        return "RF-DETR"