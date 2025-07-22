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
    def __init__(self, model, dataloader, optimizer, criterion, device, cfg, root_folder, wandb_log=False, scheduler=None, patience=sys.maxsize, pretrained_checkpoint=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
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

        self.device = device
        self.cfg = cfg
        self.trainer_cfg = cfg['trainer']
        assert 'epochs' in self.trainer_cfg.keys(), " specify 'epochs' trainer param"
        self.total_epochs = int(self.trainer_cfg['epochs'])
        self.scheduler = scheduler
        self.wandb_log = wandb_log
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
        self.accuracies = []
        self.best_params = self.model.state_dict()
        self.best_optimizer = self.optimizer.state_dict()
        self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
        self.start_epoch = 0
        self.saving_stride = 100

    def _train_step(self, batch):
        input_frame = torch.stack([item["events_vg"] for item in batch]).to(self.device)
        targets = torch.stack([item["BB"] for item in batch]).to(self.device)
        self.optimizer.zero_grad()
        outputs, losses = self.model(input_frame, targets)
        sum(losses).backward()
        self.optimizer.step()
        self.loss = losses[0].item()
        return self.loss

    def _train_epoch(self, pbar=None):
        self.model.train()
        self.total_loss = 0
        for batch in tqdm(self.dataloader, desc=f"Training model - {self.model.__class__.__name__}"):
            batch_loss = self._train_step(batch)
            self.total_loss += batch_loss
            if pbar is not None:
                pbar.set_description(f"Training, loss:{batch_loss:.4f}")
                pbar.update(1)
            if self.wandb_log:
                wandb.log({"batch_loss": batch_loss})
        avg_loss = self.total_loss / len(self.dataloader)
        if DEBUG == 1: print(f"Epoch loss: {avg_loss:.4f}")
        if self.wandb_log:
            wandb.log({"train_loss": avg_loss})
        return avg_loss

    def evaluate_model(self, val_set, eval_loss=False):
        self.model.eval()
        correct = 0
        total = 0
        losses = []
        avg_loss = None
        with torch.no_grad():
            for xs, ys in tqdm(val_set, desc="Evaluating"):
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                output = self.model(xs)
                _, preds = torch.max(output, 1)
                total += ys.size(0)
                correct += (preds == ys).sum().item()
                if eval_loss:
                    losses.append(F.nll_loss(F.log_softmax(output, dim=1), ys).cpu().item())
        if eval_loss:
            avg_loss = np.asarray(losses).mean()
        return correct / total, avg_loss

    def train(self, val_data=None):
        self.best_accuracy = 0
        self.counter = 0
        for epoch in range(self.total_epochs):
            start_time = time.time()
            avg_loss = self._train_epoch()
            epoch_time = time.time() - start_time
            if self.scheduler is not None:
                self.scheduler.step()
            if (epoch + 1) % self.saving_stride == 0 and self.save_folder is not None:
                self._save_checkpoint(epoch)
            if DEBUG == 1:
                print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            if self.wandb_log:
                wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=epoch)
            if val_data is not None:
                accuracy, val_loss = self.evaluate_model(val_data, eval_loss=True)
                self.accuracies.append(accuracy)
                if DEBUG == 1:
                    print(f"Validation accuracy at epoch {epoch+1}: {accuracy:.4f}")
                if self.wandb_log:
                    wandb.log({"val_accuracy": accuracy, "val_loss": val_loss}, step=epoch)
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_epoch = epoch
                    self.best_params = self.model.state_dict()
                    self.best_optimizer = self.optimizer.state_dict()
                    self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
                    self.counter = 0
                    if self.save_folder is not None:
                        self._save_best()
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print("Early stopping triggered. Saving best parameters...")
                        if self.save_folder is not None:
                            self._save_best()
                        return
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
        self.start_epoch = checkpoint['epoch']
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


