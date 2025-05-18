import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm
from model.first_model import model
from helpers import DEBUG, Timing
import time




class ContrastiveImageDataset(Dataset):
    """
    Custom dumb dataset for SSL returning two augmented views per sample. 
    Just to try some stuff before actually using the
    articulated mumbo-jumbo stuff from CMDA
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    path = os.path.join(class_dir, fname)
                    self.samples.append(path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert('RGB')
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2

# Training Loop
def train_ssl(model, dataloader, optimizer, criterion, device, epochs=1):
    """
    # Training Loop
    # Args:
    #   model: The model to train.
    #   dataloader: The dataloader to use. voc
    #   optimizer: The optimizer to use.
    #   criterion: The loss function to use.
    #   device: The device to train on.
    #   epochs: The number of epochs to train for.
    #
    # Returns:
    #   The average loss over all batches.
    """
    model.train()
    total_loss = 0
    loss = 0
    for i in range(epochs):
        pbar = tqdm(total=len(dataloader),desc=f"Training net, loss:{loss}")
        start_tm = time.perf_counter()
        for rgbs, events in dataloader:
            if(DEBUG>1): print(f"batch loading: {((time.perf_counter()-start_tm)*1000).__round__(3)} ms")

            # Forward pass
            #print(events.size())
            if(DEBUG>1): start_tm = time.perf_counter()# Timing
            rgb_proj, event_proj = model(rgbs, events)
            if(DEBUG>1): print(f"inference time: {((time.perf_counter()-start_tm)*1000).__round__(3)} ms")
            # Compute loss
            if(DEBUG>1): start_tm = time.perf_counter()
            loss = criterion(rgb_proj, event_proj)
            if(DEBUG>1): print(f"calculating loss: {((time.perf_counter()-start_tm)*1000).__round__(3)} ms")
            # Backward
            optimizer.zero_grad()
            if(DEBUG>1): start_tm = time.perf_counter()# Timing
            loss.backward()
            if(DEBUG>1): print(f"bacprop time: {((time.perf_counter()-start_tm)*1000).__round__(3)} ms")
            optimizer.step()

            pbar.set_description(f"Training net, loss:{loss.item()}")
            total_loss += loss.item()

            pbar.update(1)
            if(DEBUG>1): start_tm = time.perf_counter()# Timing
    
    return total_loss / len(dataloader)
