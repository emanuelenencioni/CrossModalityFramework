import torch
import torch.nn as nn
from model.rf_detr import LWDETR, SetCriterion
from model.rf_detr.models import build_backbone, build_transformer, build_matcher
from loguru import logger
from utils.helpers import DEBUG


class Rfdetrwrapper(nn.Module):
    """
    Wrapper for RF-DETR to work with CrossModalityFramework builder.py.
    """

    
    def __init__(self, backbone, head, name="rfdetrwrapper", **kwargs):
        """
        Initialize RF-DETR wrapper compatible with builder.py.
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
        
        # Add auxiliary losses if aux_loss is enabled
        if head.get('aux_loss', True):
            aux_weight_dict = {}
            for i in range(self.transformer_cfg.get('dec_layers', 6) - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
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
        
        if DEBUG >= 1:
            logger.success("RF-DETR wrapper initialized successfully")
    
    def _build_backbone(self):
        """Build backbone based on configuration."""
        from model.rf_detr.models.backbone import build_backbone
        
        backbone_type = self.backbone_cfg.get('name', 'dinov2_small')
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
        """
        # Get image dimensions
        B, C, W, H = x.shape
        
        # Forward through model
        outputs = self.model(x)
        
        # Get multi-scale backbone features
        if hasattr(self.model.backbone, 'body'):
            backbone_feats = self.model.backbone.body(x)
        else:
            backbone_feats = []
        
        # Extract predictions
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes']     # [B, num_queries, 4] normalized
        
        # DEBUG: Check raw logits distribution
        if DEBUG >= 2:
            logger.info(f"Raw logits stats: min={pred_logits.min():.4f}, max={pred_logits.max():.4f}, mean={pred_logits.mean():.4f}, std={pred_logits.std():.4f}")
        
        # Convert boxes to absolute coordinates
        pred_boxes_abs = pred_boxes.clone()
        pred_boxes_abs[..., 0] = pred_boxes[..., 0] * W  # cx
        pred_boxes_abs[..., 1] = pred_boxes[..., 1] * H  # cy
        pred_boxes_abs[..., 2] = pred_boxes[..., 2] * W  # width
        pred_boxes_abs[..., 3] = pred_boxes[..., 3] * H  # height
        
        # Get class probabilities (sigmoid for focal loss compatibility)
        class_probs = pred_logits.sigmoid()
        
        # DEBUG: Check sigmoid output
        if DEBUG >= 2:
            logger.info(f"Class probs after sigmoid: min={class_probs.min():.4f}, max={class_probs.max():.4f}, mean={class_probs.mean():.4f}")
            # Show distribution
            high_conf = (class_probs > 0.5).sum().item()
            mid_conf = ((class_probs > 0.1) & (class_probs <= 0.5)).sum().item()
            low_conf = (class_probs <= 0.1).sum().item()
            total = class_probs.numel()
            logger.info(f"Confidence distribution: high(>0.5)={high_conf}/{total} ({100*high_conf/total:.2f}%), "
                       f"mid(0.1-0.5)={mid_conf}/{total} ({100*mid_conf/total:.2f}%), "
                       f"low(<0.1)={low_conf}/{total} ({100*low_conf/total:.2f}%)")
        
        # Use max class probability as "objectness"
        objectness, pred_class_idx = class_probs.max(dim=-1, keepdim=True)  # [B, num_queries, 1]
        
        # YOLOX format: [cx, cy, w, h, objectness, class_conf_0, ...]
        head_outputs = torch.cat([
            pred_boxes_abs,  # [B, num_queries, 4]
            objectness,      # [B, num_queries, 1]
            class_probs      # [B, num_queries, num_classes]
        ], dim=-1)
        
        # Build result dict
        result = {
            'backbone_features': {
                'preflatten_feat': backbone_feats if isinstance(backbone_feats, list) else [backbone_feats]
            },
            'head_outputs': head_outputs
        }
        
        if targets is not None:
            targets_detr = self._convert_targets(targets, (H, W))
            loss_dict = self.criterion(outputs, targets_detr)
            
            # DEBUG: Show loss values and check if model is matching GTs
            if DEBUG >= 2:
                logger.info(f"Loss breakdown:")
                for k, v in loss_dict.items():
                    if k in self.criterion.weight_dict:
                        weighted_loss = v * self.criterion.weight_dict[k]
                        logger.info(f"  {k}: {v:.4f} × {self.criterion.weight_dict[k]} = {weighted_loss:.4f}")
                
                # Check predictions vs ground truth
                for i, target in enumerate(targets_detr):
                    if len(target['labels']) > 0:
                        img_objectness = objectness[i].squeeze()
                        img_pred_classes = pred_class_idx[i].squeeze()
                        img_class_probs = class_probs[i]  # [num_queries, num_classes]
                        
                        # Top predictions
                        top_k = min(5, len(img_objectness))
                        top_conf, top_idx = img_objectness.topk(top_k)
                        top_classes = img_pred_classes[top_idx]
                        
                        # Check if any predictions match GT classes
                        gt_classes = target['labels']
                        matches = sum(1 for c in top_classes if c in gt_classes)
                        
                        logger.info(f"Image {i}: GT classes={gt_classes.tolist()}, num_GT={len(gt_classes)}")
                        logger.info(f"  Top-{top_k} preds: classes={top_classes.tolist()}, conf={top_conf.tolist()}")
                        logger.info(f"  Matches with GT: {matches}/{top_k}")
                        
                        # Check the actual predictions for GT classes
                        for gt_class in gt_classes[:3]:  # Show first 3 GT classes
                            gt_class_probs_all = img_class_probs[:, gt_class]
                            max_prob_for_gt = gt_class_probs_all.max()
                            logger.info(f"  GT class {gt_class.item()}: max prob across all queries = {max_prob_for_gt:.4f}")
            
            total_loss = sum(loss_dict[k] * self.criterion.weight_dict.get(k, 1) 
                           for k in loss_dict.keys() if k in self.criterion.weight_dict)
            
            result['total_loss'] = total_loss
            result['losses'] = loss_dict
        
        if DEBUG >= 1:
            logger.info(f"Objectness: min={objectness.min():.4f}, max={objectness.max():.4f}, mean={objectness.mean():.4f}")
            logger.info(f"Predictions >0.1: {(objectness > 0.1).sum().item()}, >0.3: {(objectness > 0.3).sum().item()}, >0.5: {(objectness > 0.5).sum().item()}")
        
        return result
    
    def _convert_targets(self, targets, img_size):
        """
        Convert targets from [B, N, 5] format to DETR format.
        
        Input format: [class_id, x_center, y_center, width, height] (absolute pixels)
        Output format: list of dicts with 'labels' and 'boxes' (cxcywh normalized [0,1])
        
        Args:
            targets: tensor of shape [B, N, 5] with absolute pixel coordinates
            img_size: tuple (H, W) of image dimensions for normalization
        """
        batch_size = targets.shape[0]
        H, W = img_size
        targets_detr = []
        
        for i in range(batch_size):
            # Filter out padding (class_id < 0 means no object)
            valid_mask = targets[i, :, 0] >= 0
            valid_targets = targets[i][valid_mask]
            
            if len(valid_targets) == 0:
                targets_detr.append({
                    'labels': torch.tensor([], dtype=torch.long, device=targets.device),
                    'boxes': torch.tensor([], dtype=torch.float, device=targets.device).reshape(0, 4)
                })
                continue
            
            labels = valid_targets[:, 0].long()
            boxes_abs = valid_targets[:, 1:]  # [x_center, y_center, width, height] absolute
            
            # Normalize boxes to [0, 1] for DETR
            boxes_norm = boxes_abs.clone()
            boxes_norm[:, 0] = boxes_abs[:, 0] / W  # cx
            boxes_norm[:, 1] = boxes_abs[:, 1] / H  # cy
            boxes_norm[:, 2] = boxes_abs[:, 2] / W  # width
            boxes_norm[:, 3] = boxes_abs[:, 3] / H  # height
            
            targets_detr.append({
                'labels': labels,
                'boxes': boxes_norm
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