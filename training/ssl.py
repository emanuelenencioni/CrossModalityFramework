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
import sys

# Training Loop
import wandb



class TrainSSL:
    def __init__(self, model, dataloader, optimizer, criterion, device, epochs=1, wandb_log=False, scheduler = None, patience=sys.maxsize):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.wandb_log = wandb_log
        if self.wandb_log: assert wandb.run is not None, "Wandb run must be initialized before setting wandb_log to True"
        self.total_loss = 0
        self.loss = 0
        self.scheduler = scheduler 
        # patience
        self.patience=patience
        self.best_loss = float('inf')
        self.counter = 0
        
    def train(self):
        self.model.train()
        for i in range(self.epochs):    
            self.total_loss = 0
            pbar = tqdm(total=len(self.dataloader),desc=f"Training net, loss:{self.loss}")
            start_tm = time.perf_counter()
            for batch in self.dataloader:
                if(DEBUG>1):
                    end_tm = time.perf_counter()-start_tm
                    print(f"batch loading: {((end_tm)*1000).__round__(3)} ms")
                    if(self.wandb_log): wandb.log({"batch_loading_time":(end_tm*1000).__round__(3)})
                
                if(DEBUG>1): start_tm = time.perf_counter()# Timing
                
                rgbs = torch.stack([item["image"] for item in batch]).to(self.device)
                events = torch.stack([item["events_vg"] for item in batch]).to(self.device)
                if(DEBUG>1): 
                    end_tm = time.perf_counter()-start_tm
                    print(f"frame extraction: {(end_tm*1000).__round__(3)} ms")
                    if(self.wandb_log): wandb.log({"frame_extraction_time":(end_tm*1000).__round__(3)})


                if(DEBUG>1): start_tm = time.perf_counter()# Timing
                
                rgb_proj, event_proj = self.model(rgbs, events)
                if(DEBUG>1): 
                    end_tm = time.perf_counter()-start_tm
                    print(f"inference time: {((end_tm)*1000).__round__(3)} ms")
                    if(self.wandb_log): wandb.log({"inference_time":(end_tm*1000).__round__(3)})

                # Compute loss
                if(DEBUG>1): start_tm = time.perf_counter()
                
                self.loss = self.criterion(rgb_proj, event_proj)
                if(DEBUG>1): 
                    end_tm = time.perf_counter()-start_tm
                    print(f"calculating loss: {((end_tm)*1000).__round__(3)} ms")
                    if(self.wandb_log): wandb.log({"loss_time":(end_tm*1000).__round__(3)})
                # Backward
                    self.optimizer.zero_grad()
                if(DEBUG>1): start_tm = time.perf_counter()# Timing
                
                self.loss.backward()
                if(DEBUG>1): 
                    end_tm = time.perf_counter()-start_tm
                    print(f"backprop time: {((end_tm)*1000).__round__(3)} ms")
                    if(self.wandb_log): wandb.log({"backprop_time":(end_tm*1000).__round__(3)})
                self.optimizer.step()
                
                pbar.set_description(f"Training net, loss:{self.loss.item()}")
                self.total_loss += self.loss.item()

                if self.wandb_log:
                    wandb.log({"batch_loss": self.loss.item()})

                    rgb_n, event_n = self.model.get_grad_norm()
                    wandb.log({"rgb_grad_norm": rgb_n})
                    wandb.log({"event_grad_norm": event_n})

                    rgb_n, event_n = self.model.get_weights_norm()
                    wandb.log({"rgb_weight_norm": rgb_n})
                    wandb.log({"event_weight_norm": event_n})


                pbar.update(1)
                if(DEBUG>1): start_tm = time.perf_counter()# Timing
    
            epoch_loss = self.total_loss / len(self.dataloader)
            if epoch_loss < self.best_loss: #TODO: save model weights, opti, cfg, scheduler...
                self.best_loss = epoch_loss
                self.counter = 0
            else: self.counter+=1
            if self.counter >= self.patience: #If the counter exceeds the patience value
                    print("Early stopping triggered")
                    return #Stop the training loop

            if self.wandb_log:
                wandb.log({"average_loss": epoch_loss})

            if self.scheduler is not None: self.scheduler.step()

        wandb.finish()
        print("training finished")
