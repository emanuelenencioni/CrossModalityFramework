from logging import DEBUG
import torch.nn as nn
import torch.nn.functional as F
import torch


class CLIP(nn.Module):
    """
        Implementation of the CLIP loss function (https://arxiv.org/pdf/2103.00020)
    """
    def __init__(self):
        super().__init__()
        #learnable temperature
        self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        z1_emb = F.normalize(z1, p=2, dim=-1)
        z2_emb = F.normalize(z2, p=2, dim=-1)
        # Cosine similarity logits
        logits = self.temp.exp() * torch.matmul(z1_emb, z2_emb.T)

        labels = torch.arange(batch_size, device=z1.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

class BarlowTwinsLoss(nn.Module):
    """
    Implementation of the Barlow Twins loss function (https://arxiv.org/abs/2103.03230).
    """
    def __init__(self, lambda_coeff=5e-3):
        """
        Initializes the BarlowTwinsLoss module.
        
        Args:
            lambda_coeff (float): The weight for the redundancy reduction term.
                                  The paper suggests a value of 5e-3.
        """
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Computes the Barlow Twins loss (https://arxiv.org/abs/2103.03230).
        Args:
            z_a (torch.Tensor): The first batch of embeddings, of shape [batch_size, embedding_dim].
            z_b (torch.Tensor): The second batch of embeddings, of shape [batch_size, embedding_dim].
        Returns:
            torch.Tensor: A scalar loss value.
        """
        batch_size, embedding_dim = z_a.shape
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-5)
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-5)
        cross_corr_matrix = (z_a_norm.T @ z_b_norm) / batch_size
        on_diag = torch.diagonal(cross_corr_matrix)
        invariance_loss = ((on_diag - 1)**2).sum()

        off_diag = cross_corr_matrix.fill_diagonal_(0)
        redundancy_loss = (off_diag**2).sum()

        total_loss = invariance_loss + self.lambda_coeff * redundancy_loss
        
        return total_loss

class IOUloss(nn.Module):
    def __init__(self, reduction="none", eps=1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Calculate IoU loss for bounding boxes in [cx, cy, w, h] format
        Args:
            pred: [N, 4] predicted boxes [center_x, center_y, width, height]
            target: [N, 4] target boxes [center_x, center_y, width, height]
        """
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        pred_x1 = pred[..., 0] - pred[..., 2] / 2
        pred_y1 = pred[..., 1] - pred[..., 3] / 2
        pred_x2 = pred[..., 0] + pred[..., 2] / 2
        pred_y2 = pred[..., 1] + pred[..., 3] / 2
        
        target_x1 = target[..., 0] - target[..., 2] / 2
        target_y1 = target[..., 1] - target[..., 3] / 2
        target_x2 = target[..., 0] + target[..., 2] / 2
        target_y2 = target[..., 1] + target[..., 3] / 2
        
        # Calculate intersection coordinates
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        # Calculate intersection area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Calculate areas of both boxes
        pred_area = pred[..., 2] * pred[..., 3]  # width * height
        target_area = target[..., 2] * target[..., 3]  # width * height
        
        # Calculate union area
        union_area = pred_area + target_area - inter_area + self.eps
        
        # Calculate IoU
        iou = inter_area / union_area
        
        # IoU loss = 1 - IoU
        iou_loss = 1 - iou
        
        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss


def build_from_config(cfg):
    """
    Factory method. Return the loss criterion, based on the configuration file
    Args:
        cfg: the sub vocabulary of the configuration file, containing the loss criterion parameters
    Returns:
        The loss criterion, and a boolean value indicating whether the loss criterion have learnable parameters
    """
    if cfg.get('dual_modality', True):
        assert 'multi_modality_loss' in cfg.keys(), "multi_modality_loss params list missing in yaml file"
    else:
        if 'multi_modality_loss' not in cfg.keys():
            print("\033[93mWarning - multi_modality_loss params list missing in yaml file, remember: this param is mandatory in DualModality\033[0m")
            return None, False
    loss_cfg = cfg['multi_modality_loss']

    assert 'name' in loss_cfg.keys(), "specify 'name' loss param"
    criterion = loss_cfg['name'].lower()

    if criterion in ["clip", "clip_loss"]:
        cfg['learnable_loss'] = True
        return CLIP()
    elif criterion in ["barlow_twins", "barlowtwins", "barlow_twins_loss", "barlow_twin", "barlowtwin"]:
        assert "lambda" in loss_cfg.keys(), "Missing lambda parameter in the configuration file"
        assert isinstance(loss_cfg["lambda"], float), "lambda_coeff must be a float"
        lambda_coeff = float(loss_cfg["lambda"])
        assert lambda_coeff > 0 and lambda_coeff < 1, "lambda must be in the range (0, 1)"

        return BarlowTwinsLoss(lambda_coeff=lambda_coeff), False
    elif criterion in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.MSELoss(), False

    elif criterion in ["cross_entropy", "crossentropyloss", "cross_entropy_loss", "ce"]:
        return torch.nn.CrossEntropyLoss(), False

    elif criterion in ["bce", "bce_loss", "binary_cross_entropy", "binarycrossentropy"]:
        return torch.nn.BCEWithLogitsLoss(), False
    elif criterion in ["IOULoss", "iouloss", "IOU", "iou"]:
        return IOUloss()

    else:
        raise ValueError("Criterion name mispelled or missing implementation")