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
import wandb
def train_ssl(model, dataloader, optimizer, criterion, device, epochs=1, wandb_log=False):
    """
    # Training Loop
    # Args:
    #   model: The model to train.
    #   dataloader: The dataloader to use. voc
    #   optimizer: The optimizer to use.
    #   criterion: The loss function to use.
    #   device: The device to train on.
    #   epochs: The number of epochs to train for.
    #   wandb_log: If True, log metrics to wandb.
    # Returns:
    #   The average loss over all batches.
    """
    if wandb_log: assert wandb.run is not None, "Wandb run must be initialized before setting wandb_log to True"

    model.train()
    total_loss = 0
    loss = 0
    for i in range(epochs):
        pbar = tqdm(total=len(dataloader),desc=f"Training net, loss:{loss}")
        start_tm = time.perf_counter()
        for batch in dataloader:
            if(DEBUG>1):
                end_tm = time.perf_counter()-start_tm
                print(f"batch loading: {((end_tm)*1000).__round__(3)} ms")
                if(wandb_log): wandb.log({"batch_loading_time":(end_tm*1000).__round__(3)})
            
            if(DEBUG>1): start_tm = time.perf_counter()# Timing
            rgbs = torch.stack([item["image"] for item in batch]).to(device)
            events = torch.stack([item["events_vg"] for item in batch]).to(device)
            if(DEBUG>1): 
                end_tm = time.perf_counter()-start_tm
                print(f"frame extraction: {(end_tm*1000).__round__(3)} ms")
                if(wandb_log): wandb.log({"frame_extraction_time":(end_tm*1000).__round__(3)})


            if(DEBUG>1): start_tm = time.perf_counter()# Timing
            rgb_proj, event_proj = model(rgbs, events)
            if(DEBUG>1): 
                end_tm = time.perf_counter()-start_tm
                print(f"inference time: {((end_tm)*1000).__round__(3)} ms")
                if(wandb_log): wandb.log({"inference_time":(end_tm*1000).__round__(3)})

            # Compute loss
            if(DEBUG>1): start_tm = time.perf_counter()
            loss = criterion(rgb_proj, event_proj)
            if(DEBUG>1): 
                end_tm = time.perf_counter()-start_tm
                print(f"calculating loss: {((end_tm)*1000).__round__(3)} ms")
                if(wandb_log): wandb.log({"loss_time":(end_tm*1000).__round__(3)})
            # Backward
            optimizer.zero_grad()
            if(DEBUG>1): start_tm = time.perf_counter()# Timing
            loss.backward()
            if(DEBUG>1): 
                end_tm = time.perf_counter()-start_tm
                print(f"backprop time: {((end_tm)*1000).__round__(3)} ms")
                if(wandb_log): wandb.log({"backprop_time":(end_tm*1000).__round__(3)})
            optimizer.step()

            pbar.set_description(f"Training net, loss:{loss.item()}")
            total_loss += loss.item()

            if wandb_log:
                wandb.log({"batch_loss": loss.item()})

                rgb_n, event_n = model.get_grad_norm()
                wandb.log({"rgb_grad_norm": rgb_n})
                wandb.log({"event_grad_norm": event_n})

                rgb_n, event_n = model.get_weights_norm()
                wandb.log({"rgb_weight_norm": rgb_n})
                wandb.log({"event_weight_norm": event_n})

            pbar.update(1)
            if(DEBUG>1): start_tm = time.perf_counter()# Timing
    
    avg_loss = total_loss / len(dataloader)

    if wandb_log:
        wandb.log({"average_loss": avg_loss})

    return avg_loss
