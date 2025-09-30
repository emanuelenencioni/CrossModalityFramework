import inspect
import datetime
import time
from tqdm import tqdm
import numpy as np
import wandb
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils.helpers import DEBUG, deep_dict_equal


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
        self.input_type = 'events' if 'events' in dataloader.dataset[0] else 'image'
        if 'events' in dataloader.dataset[0] and 'image' in dataloader.dataset[0]:
            print("\033[93m"+"WARNING: the dataloader contains both events_vg and image, using events_vg as input type"+"\033[0m")
        if DEBUG >= 1: print(f"Input type: {self.input_type}")
        if pretrained_checkpoint is not None:
            if 'model_state_dict' in pretrained_checkpoint:
                self.model.load_state_dict(pretrained_checkpoint['model_state_dict'])
                if DEBUG >= 1: print("Pre-trained model loaded successfully (CrossModalityFramework)")
            
            if 'optimizer_state_dict' in pretrained_checkpoint:
                self.optimizer.load_state_dict(pretrained_checkpoint['optimizer_state_dict'])
                if DEBUG >= 1: print("Optimizer state loaded successfully")

            if 'scheduler_state_dict' in pretrained_checkpoint and cfg['trainer'].get('resume_scheduler', False):
                scheduler_state = pretrained_checkpoint['scheduler_state_dict']
                if scheduler_state is not None and self.scheduler is not None and \
                deep_dict_equal(cfg['scheduler'], pretrained_checkpoint['config'].get('scheduler', None)):
                    self.scheduler.load_state_dict(scheduler_state)
                    if DEBUG >= 1: print("Scheduler state loaded successfully")

            if 'epoch' in pretrained_checkpoint:
                self.epoch = pretrained_checkpoint['epoch'] + 1
                self.total_epochs = int(self.trainer_cfg['epochs']) + self.epoch - 1
                if DEBUG >= 1: print(f"Resuming training from epoch {self.epoch}")
        

        self.checkpoint_interval_epochs = cfg['trainer'].get('checkpoint_interval_epochs', 0)
        
        if self.checkpoint_interval_epochs > 0:
            self.checkpoint_interval = 0
        else: 
            self.checkpoint_interval = cfg['trainer'].get('checkpoint_interval', 0)

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
        self.total_loss = 0
        self.best_loss = float('inf')
        self.accuracies = []
        self.best_params = self.model.state_dict()
        self.best_optimizer = self.optimizer.state_dict()
        self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
        self.best_ap50_95 = 0
        self.step = 0

    def _train_step(self, batch):
        input_frame = torch.stack([item[self.input_type] for item in batch]).to(self.device)
        targets = torch.stack([item["BB"] for item in batch]).to(self.device) #For now considering only object detection tasks
        self.optimizer.zero_grad()
        _, losses = self.model(input_frame, targets)
        losses[0].backward()
        l1_loss = losses[4] if isinstance(losses[4], float) else losses[4].item()
        if DEBUG >= 1: 
            print(f"weighted_iou_loss: {losses[1].item():.4f}, loss_obj: {losses[2].item():.4f}, loss_cls: {losses[3].item():.4f}, loss_l1: {l1_loss:.4f}")
        if self.wandb_log: # TODO make it work with not knowing losses length
            wandb.log({"loss/weighted_iou": losses[1].item(), "loss/obj": losses[2].item(), "loss/cls": losses[3].item(), "loss/l1": l1_loss, "loss/batch(sum):": losses[0].item(), "step": self.step})
        self.optimizer.step()
        return losses

    def _train_epoch(self, pbar=None):
        self.model.train()
        self.total_loss = 0
        total_losses = []
        for batch in self.dataloader:
            losses = self._train_step(batch)
            self.total_loss += losses[0].item()
            total_losses.append([loss.item() if not isinstance(loss, float) else loss for loss in losses])
            if pbar is not None:
                pbar.set_description(f"Training model {self.model.get_name()}, loss:{losses[0].item():.4f}")
                pbar.update(1)
            self.step += 1
        total_losses = np.array(total_losses)
        avg_losses = np.mean(total_losses, axis=0)  #
        if DEBUG == 1: print(f"Epoch loss: {avg_losses[0]:.4f}")
        if self.wandb_log:
            wandb.log({"loss/epoch": avg_losses[0],"loss/epoch_iou": avg_losses[1],"loss/epoch_obj": avg_losses[2],"loss/epoch_cls": avg_losses[3],"loss/epoch_l1": avg_losses[4],"epoch": self.epoch})
        return avg_losses[0]

    def train(self, evaluator=None, eval_loss=False):
        for epoch in range(self.total_epochs):
            self.model.train()
            start_time = time.time()
            with tqdm(total=len(self.dataloader), desc=f"Epoch {self.epoch}/{self.total_epochs}") as pbar:
                avg_loss = self._train_epoch(pbar)
            epoch_time = time.time() - start_time
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau requires a metric value
                    self.scheduler.step(avg_loss)
                else:
                    # Standard schedulers that step based on epochs
                    self.scheduler.step()

            if (self.checkpoint_interval > 0 and (self.step % self.checkpoint_interval == 0)) or (self.checkpoint_interval_epochs > 0 and (epoch + 1) % self.checkpoint_interval_epochs == 0):
                if self.save_folder is not None:
                    self._save_checkpoint(epoch)
                else:
                    print("\033[93m"+"WARNING: the model will not be saved - saving folder need to be specifiedR"+"\033[0m")
            if DEBUG == 1:
                print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            if self.wandb_log:
                wandb.log({"lr": self.optimizer.param_groups[0]['lr'], "epoch": self.epoch})

            if evaluator is not None:
                # stats is a numpy array of 12 elements
                stats = evaluator.evaluate(self.model)

                if DEBUG >= 1: 
                    # Primary COCO metrics1
                    ap50_95 = stats[0]
                    ap50 = stats[1]
                    print(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")

                if self.wandb_log:
                    wandb.log({
                        "AP/AP(50_95)": stats[0],
                        "AP/AP50": stats[1],
                        "AP/AP75": stats[2],
                        "AP/APs": stats[3],
                        "AP/APm": stats[4],
                        "AP/APl": stats[5],
                        "AR/AR1": stats[6],
                        "AR/AR10": stats[7],
                        "AR/AR100": stats[8],
                        "AR/ARs": stats[9],
                        "AR/ARm": stats[10],
                        "AR/ARl": stats[11],
                        "epoch": self.epoch
                    })

            elif hasattr(self.dataloader.dataset, 'evaluate'):
                # Use dataset's evaluate method
                ap50_95, ap50, _ = self.dataloader.dataset.evaluate(self.model)
                if DEBUG >= 1: print(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")

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
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
        checkpoint_path = f"{self.save_folder}{self.save_name}_checkpoint_epoch_{epoch}_{timestamp}.pth"
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


