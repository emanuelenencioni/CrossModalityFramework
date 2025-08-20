import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from tqdm import tqdm
import numpy as np
import wandb
import os
import sys
from helpers import DEBUG

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device, cfg, root_folder,wandb_log=False, scheduler=None, patience=sys.maxsize, pretrained_checkpoint=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = 1
        self.cfg = cfg
        self.trainer_cfg = cfg['trainer']
        assert 'epochs' in self.trainer_cfg.keys(), " specify 'epochs' trainer param"
        self.total_epochs = int(self.trainer_cfg['epochs'])
        self.input_type = 'events_vg' if 'events_vg' in dataloader.dataset[0] else 'image'
        if 'events_vg' in dataloader.dataset[0] and 'image' in dataloader.dataset[0]:
            print("\033[93m"+"WARNING: the dataloader contains both events_vg and image, using events_vg as input type"+"\033[0m")
        if DEBUG >= 1: print(f"Input type: {self.input_type}")
        if pretrained_checkpoint is not None:
            if 'model_state_dict' in pretrained_checkpoint:
                self.model.load_state_dict(pretrained_checkpoint['model_state_dict'])
                if DEBUG >= 1: print("Pre-trained model loaded successfully")
            if 'optimizer_state_dict' in pretrained_checkpoint:
                self.optimizer.load_state_dict(pretrained_checkpoint['optimizer_state_dict'])
                if DEBUG >= 1: print("Pre-trained optimizer state loaded successfully")
            if 'scheduler_state_dict' in pretrained_checkpoint and cfg['trainer'].get('resume_scheduler', False):
                scheduler_state = pretrained_checkpoint['scheduler_state_dict']
                if scheduler_state is not None and self.scheduler is not None:
                    self.scheduler.load_state_dict(scheduler_state)
                    if DEBUG >= 1: print("Pre-trained scheduler state loaded successfully")
            if 'epoch' in pretrained_checkpoint:
                self.epoch = pretrained_checkpoint['epoch'] + 1
                self.total_epochs = int(self.trainer_cfg['epochs']) + self.epoch - 1
                if DEBUG >= 1: print(f"Resuming training from epoch {self.epoch}")

        self.device = device
        
        self.scheduler = scheduler
        self.wandb_log = True if wandb.run is not None else False
        if self.wandb_log: assert wandb.run is not None, "Wandb run must be initialized before setting wandb_log to True"
        self.save_folder = root_folder + "/" + self.trainer_cfg['save_folder'] if 'save_folder' in self.trainer_cfg.keys() and self.trainer_cfg['save_folder'] is not None else None
        self.save_best_dir = None
        if self.save_folder is not None:
            if self.save_folder[-1] != '/':
                self.save_folder += '/'
            self.save_best_dir = f"{self.save_folder}best/"
            os.makedirs(self.save_best_dir, exist_ok=True)
            self.save_name = wandb.run.name if self.wandb_log else f"{self.model.__class__.__name__}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            print("\033[93m"+"WARNING: the model will not be saved"+"\033[0m")
            self.save_name = None
        self.patience = patience
        self.best_accuracy = 0
        self.best_epoch = 0
        self.counter = 0
        self.loss = 0
        self.total_loss = 0
        self.best_loss = float('inf')
        self.accuracies = []
        self.best_params = self.model.state_dict()
        self.best_optimizer = self.optimizer.state_dict()
        self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
        self.best_ap50_95 = 0
        self.saving_stride = cfg['trainer']['log_interval'] if 'log_interval' in cfg['trainer'].keys() else 500
        self.step = 0

    def _train_step(self, batch):
        input_frame = torch.stack([item[self.input_type] for item in batch]).to(self.device)
        targets = torch.stack([item["BB"] for item in batch]).to(self.device) #For now considering only object detection tasks
        self.optimizer.zero_grad()
        _, losses = self.model(input_frame, targets)
        losses[0].backward()
        if DEBUG >= 1: 
            print(f"loss_obj: {losses[1].item():.4f}, loss_cls: {losses[2].item():.4f}, loss_l1: {losses[3].item():.4f}")
        if wandb.run is not None:
            wandb.log({"loss_obj": losses[1].item(), "loss_cls": losses[2].item(), "loss_l1": losses[3].item()}, step=self.step)
        self.optimizer.step()
        self.loss = losses[0].item()
        return self.loss

    def _train_epoch(self, pbar=None):
        self.model.train()
        self.total_loss = 0
        for batch in self.dataloader:
            batch_loss = self._train_step(batch)
            self.total_loss += batch_loss
            if pbar is not None:
                pbar.set_description(f"Training model {self.model.get_name()}, loss:{batch_loss:.4f}")
                pbar.update(1)
            if self.wandb_log:
                wandb.log({"batch_loss": batch_loss})
            self.step += 1
        avg_loss = self.total_loss / len(self.dataloader)
        if DEBUG == 1: print(f"Epoch loss: {avg_loss:.4f}", step=self.step)
        if self.wandb_log:
            wandb.log({"epoch_loss": avg_loss},step=self.step)
        return avg_loss

    def train(self, evaluator=None, eval_loss=False):
        for epoch in range(self.total_epochs):
            start_time = time.time()
            with tqdm(total=len(self.dataloader), desc=f"Epoch {self.epoch}/{self.total_epochs}") as pbar:
                avg_loss = self._train_epoch(pbar)
            epoch_time = time.time() - start_time
            if self.scheduler is not None:
                self.scheduler.step()
            if (epoch + 1) % self.saving_stride == 0:
                if self.save_folder is not None:
                    self._save_checkpoint(epoch)
                else:
                    print("\033[93m"+"WARNING: the model will not be saved - saving folder need to be specified"+"\033[0m")
            if DEBUG == 1:
                print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            if self.wandb_log:
                wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=self.step)

            if evaluator is not None:
                ap50_95, ap50, _ = evaluator.evaluate(self.model)
                if DEBUG >= 1: print(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")

                if self.wandb_log:
                    wandb.log({"ap50_95": ap50_95, "ap50": ap50}, step=self.step)
            elif hasattr(self.dataloader.dataset, 'evaluate'):
                # Use dataset's evaluate method
                ap50_95, ap50, _ = self.dataloader.dataset.evaluate(self.model)
                if DEBUG >= 1: print(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")
                
                if self.wandb_log:
                    wandb.log({"ap50_95": ap50_95, "ap50": ap50}, step=self.step)
            if avg_loss < self.best_loss: #TODO: should be on the loss
                if DEBUG >= 1: print(f"New best loss: {avg_loss:.4f} at epoch {self.epoch}")
                self.best_ap50_95 = ap50_95 if evaluator is not None else None
                self.best_ap50 = ap50 if evaluator is not None else None
                self.best_epoch = self.epoch
                self.best_params = self.model.state_dict()
                self.best_optimizer = self.optimizer.state_dict()
                self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
                if self.save_folder is not None:
                    self._save_best()
            self.epoch += 1

        print("Training finished.")

    def _save_best(self):
        save_path = f"{self.save_best_dir}{self.save_name}_best.pth"
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.best_params,
            'optimizer_state_dict': self.best_optimizer,
            'scheduler_state_dict': self.best_sch_params,
            'config': self.cfg
        }, save_path)
        print(f"Saved best model to {save_path}")

    def _save_checkpoint(self, epoch):
        checkpoint_path = f"{self.save_folder}{self.save_name}_checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'config': self.cfg
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def load_model_state(self, file_path):
        print("Loading model...")
        checkpoint = torch.load(file_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


