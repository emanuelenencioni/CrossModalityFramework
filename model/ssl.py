import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models



def clip_contrastive_loss(z1, z2, temperature=0.07):
    """
    CLIP-style symmetric cross-entropy loss between two embeddings.
    z1, z2: [batch_size, dim]
    """
    batch_size = z1.size(0)
    # Cosine similarity logits
    logits = torch.matmul(z1, z2.t()) / temperature

    labels = torch.arange(batch_size, device=z1.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2