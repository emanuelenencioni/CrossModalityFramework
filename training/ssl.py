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
        self.current_step = 0

    def _train_step(self, batch):
            rgbs = torch.stack([item["image"] for item in batch]).to(self.device)
            events = torch.stack([item["events_vg"] for item in batch]).to(self.device)
            rgb_proj, event_proj = self.model(rgbs, events)
            self.loss = self.criterion(rgb_proj, event_proj)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def _train_step_debug(self, batch):   
        if(DEBUG>1): start_tm = time.perf_counter()# Timing
        
        rgbs = torch.stack([item["image"] for item in batch]).to(self.device)
        events = torch.stack([item["events_vg"] for item in batch]).to(self.device)
        if(DEBUG>1): 
            end_tm = ((time.perf_counter()-start_tm)*1000).__round__(3)
            print(f"frame extraction: {end_tm} ms")
            if(self.wandb_log): wandb.log({"frame_extraction_time":end_tm},step=self.current_step)

        if(DEBUG>1): start_tm = time.perf_counter()# Timing
        rgb_proj, event_proj = self.model(rgbs, events)
        if(DEBUG>1): 
            end_tm = time.perf_counter()-start_tm
            print(f"inference time: {((end_tm)*1000).__round__(3)} ms")
            if(self.wandb_log): wandb.log({"inference_time":(end_tm*1000).__round__(3)}, step=self.current_step)

        # Compute loss
        if(DEBUG>1): start_tm = time.perf_counter()
        
        self.loss = self.criterion(rgb_proj, event_proj)
        if(DEBUG>1): 
            end_tm = time.perf_counter()-start_tm
            print(f"calculating loss: {((end_tm)*1000).__round__(3)} ms")
            if(self.wandb_log): wandb.log({"loss_time":(end_tm*1000).__round__(3)}, step=self.current_step)
        # Backward
        self.optimizer.zero_grad()
        if(DEBUG>1): start_tm = time.perf_counter()# Timing
        
        self.loss.backward()
        if(DEBUG>1): 
            end_tm = time.perf_counter()-start_tm
            print(f"backprop time: {((end_tm)*1000).__round__(3)} ms")
            if(self.wandb_log): wandb.log({"backprop_time":(end_tm*1000).__round__(3)}, step=self.current_step)
        self.optimizer.step()        
        pass

    def _train_epoch(self, pbar=None):
        start_tm = time.perf_counter()

        for batch in self.dataloader:
            if(DEBUG>1):
                end_tm = time.perf_counter()-start_tm
                print(f"batch loading: {((end_tm)*1000).__round__(3)} ms")
                if(self.wandb_log): wandb.log({"batch_loading_time":(end_tm*1000).__round__(3)}, step=self.current_step)

            if DEBUG>0: self._train_step_debug(batch)
            else: self._train_step(batch)
            
            if pbar is not None: pbar.set_description(f"Training backbones, loss:{self.loss.item()}")
            self.total_loss += self.loss.item()

            if self.wandb_log:
                wandb.log({"batch_loss": self.loss.item()})

                rgb_n, event_n = self.model.get_grad_norm()
                wandb.log({"rgb_grad_norm": rgb_n}, step=self.current_step)
                wandb.log({"event_grad_norm": event_n}, step=self.current_step)

                rgb_n, event_n = self.model.get_weights_norm()
                wandb.log({"rgb_weight_norm": rgb_n}, step=self.current_step)
                wandb.log({"event_weight_norm": event_n}, step=self.current_step)
            
            self.current_step += 1

            if pbar is not None: pbar.update(1)
            if(DEBUG>1): start_tm = time.perf_counter()# Timing

        
    def train(self):
        self.current_step = 0
        self.model.train()
        if DEBUG>2:
            self._train_debug()
        else:
            for i in range(self.epochs):  
                self.total_loss = 0
                pbar = tqdm(total=len(self.dataloader),desc=f"Training backbones, loss:{self.loss}")
                self._train_epoch(pbar)
                epoch_loss = self.total_loss / len(self.dataloader)
                if epoch_loss < self.best_loss: #TODO: save model weights, opti, cfg, scheduler...
                    self.best_loss = epoch_loss
                    self.counter = 0
                else: self.counter+=1
                if self.counter >= self.patience: #If the counter exceeds the patience value
                    print("Early stopping triggered")
                    return #Stop the training loop

                if self.wandb_log:
                    wandb.log({"average_loss": epoch_loss}, step=self.current_step)

                if self.scheduler is not None: self.scheduler.step()

            wandb.finish()
            print("training finished")

    def _train_debug(self):
        from torch.profiler import profile, record_function, ProfilerActivity
        activties = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if self.wandb_log: sys.exit("ERROR - wandb logger not yet available in debug mode, please deactivate it")
        with profile(activities=activties) as prof:
            if len(self.dataloader) < 25:
                self._train_epoch()
            else:
                for i, batch in enumerate(self.dataloader):
                    self._train_step_debug(batch)
                    self.current_step += 1
                    if i >= 25:
                        break
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))

