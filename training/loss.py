import torch.nn as nn
import torch.nn.functional as F
import torch


class CLIP_loss(nn.Module):
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



def build_from_config(cfg):
    """
    Factory method. Return the loss criterion, based on the configuration file
    Args:
        cfg: the sub vocabulary of the configuration file, containing the loss criterion parameters
    Returns:
        The loss criterion, and a boolean value indicating whether the loss criterion have learnable parameters
    """
    assert 'name' in cfg.keys(), "specify 'name' loss param"
    criterion = cfg['name'].lower()

    if criterion in ["clip", "clip_loss"]:
        return CLIP_loss(), True
    elif criterion in ["barlow_twins", "barlowtwins", "barlow_twins_loss", "barlow_twin", "barlowtwin"]:
        assert "lambda" in cfg.keys(), "Missing lambda parameter in the configuration file"
        assert isinstance(cfg["lambda"], float), "lambda_coeff must be a float"
        lambda_coeff = float(cfg["lambda"])
        assert lambda_coeff > 0 and lambda_coeff < 1, "lambda must be in the range (0, 1)"

        return BarlowTwinsLoss(lambda_coeff=lambda_coeff), False
    else:
        raise ValueError("Criterion name mispelled or missing implementation")
