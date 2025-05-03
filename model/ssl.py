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

def train_contrastive(model1, model2, dataloader, optimizer, device, temperature=0.07):
    model1.train()
    model2.train()
    total_loss = 0.0
    for x1, x2 in dataloader:
        x1, x2 = x1.to(device), x2.to(device)

        optimizer.zero_grad()

        z1 = model1(x1)
        z2 = model2(x2)

        loss = clip_contrastive_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)