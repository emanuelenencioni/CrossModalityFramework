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
from datetime import datetime
import sys
import os
# Training Loop
import wandb



class TrainSSL:
    def __init__(self, model, dataloader, optimizer, criterion, device, cfg, root_folder, wandb_log=False, scheduler = None, patience=sys.maxsize):
        

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.epochs = int(cfg['epochs'])
        assert 'epochs' in cfg.keys(), " specify 'epochs' trainer param" # The cfg Must be the sub dict relative to 'trainer'
        #assert 'log_interval' in cfg.keys(), "specify 'log_interval' trainer param"
        #assert 'val_interval' in cfg.keys(), "specify 'val_interval' trainer param"

        self.checkpoint_interval = int(cfg['checkpoint_interval']) if 'checkpoint_interval' in cfg.keys() else 0

        
        self.wandb_log = wandb_log
        if self.wandb_log: assert wandb.run is not None, "Wandb run must be initialized before setting wandb_log to True"
        self.save_folder = root_folder +"/"+ cfg['save_folder'] if 'save_folder' in cfg.keys() and cfg['save_folder'] is not None else None
        
        self.save_name = None
        self.save_best_dir = None
        if self.save_folder is not None:
            if self.save_folder[-1] != '/':
                self.save_folder = self.save_folder + '/'
            self.save_best_dir = f"{self.save_folder}best/"
            if not os.path.isdir(self.save_best_dir): os.makedirs(self.save_best_dir)

            self.save_name = wandb.run.name if self.wandb_log else f"{self.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            print("\033[93m"+"WARNING: the model will not be saved"+"\033[0m")

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
            if self.current_step % self.checkpoint_interval == 0 and self.current_step > 0 and self.save_folder is not None:
                checkpoint_path = f"{self.save_folder}{self.save_name}_checkpoint_step_{self.current_step}.pth"
                torch.save({
                    'step': self.current_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                    # Optionally save scheduler state too
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                }, checkpoint_path)
                print(f"Checkpoint saved at step {self.current_step} to {checkpoint_path}")
            if self.wandb_log:
                wandb.log({"batch_loss": self.loss.item()}, step=self.current_step)

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
                    if self.save_folder is not None: self.save_best()
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

    def save_best(self):
            save_path = f"{self.save_best_dir}{self.save_name}_best.pth"
            torch.save({
                    'epoch': self.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                    # Optionally save scheduler state too
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                }, save_path)
            print(f"saved best model to {save_path}")
