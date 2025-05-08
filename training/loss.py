import torch.nn as nn
import torch.nn.functional as F
import torch


class CLIP_loss(nn.Module):
    """
        CLIP-style (https://arxiv.org/pdf/2103.00020) symmetric cross-entropy loss between two embeddings.
        z1, z2: [batch_size, dim]
    """
    def __init__(self):
        super().__init__()
        #learnable temperature
        self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, z1, z2, temperature=0.07):
        batch_size = z1.size(0)

        z1_emb = F.normalize(z1, p=2, dim=-1)
        z2_emb = F.normalize(z2, p=2, dim=-1)
        # Cosine similarity logits
        logits = self.temp.exp() * torch.matmul(z1_emb, z2_emb.T)

        labels = torch.arange(batch_size, device=z1.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2