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
    def __init__(self, model, dataloader, optimizer, criterion, device, cfg, root_folder, wandb_log=False, scheduler = None, patience=sys.maxsize):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.accuracies = []
        self.criterion = criterion
        self.device = device
        self.losses = []
        self.trainer_cfg = cfg['trainer']
        assert 'epochs' in self.trainer_cfg.keys(), " specify 'epochs' trainer param"
        self.total_epochs = int(self.trainer_cfg['epochs'])
        self.scheduler = scheduler
        if wandb_log: assert wandb.run is not None, "Wandb run must be initialized before setting wandb_log to True"
        self.use_wandb = wandb_log
        self.save_folder = root_folder +"/"+ cfg['save_folder'] if 'save_folder' in cfg.keys() and cfg['save_folder'] is not None else None
        self.counter = 0
        self.best_params = self.model.state_dict()
        self.best_epoch = 0
        self.best_opti_params = optimizer.state_dict()
        self.best_sch_params = scheduler.state_dict() if scheduler is not None else None
        self.start_epoch = 0
        self.save_id = id
        #self.use_logit = use_logit
        self.patience = patience
        self.saving_stride = 100

    def __run_batch(self, batch):
        input_frame = torch.stack([item["events_vg"] for item in batch]).to(self.device)
        targets = torch.stack([item["BB"] for item in batch])
        self.optimizer.zero_grad()
        losses = self.model(input_frame, targets)
        
        #loss = self.criterion(output, targets)
        #saved_loss = loss.detach().cpu()
        losses[0].backward()
        self.optimizer.step()
        return losses[0] #dummy return for now

    def __run_epoch(self, epoch):
        self.model.train()
        losses = []
        for batch in  self.dataloader:
            losses.append(self.__run_batch(batch))
        avg_loss = np.asarray(losses).mean()
        if DEBUG == 1: print(f"epoch loss: {avg_loss}")
        if self.use_wandb:
            wandb.log({"train_loss": avg_loss}, step=epoch)
        self.losses.append(avg_loss)


    def evaluate_model(self, val_set, eval_loss=False):
        '''Function that evaluate the model accuracy on a validation set.
        '''
        self.model.eval()
        predictions = []
        gts = []
        correct = 0
        total = 0
        losses = []
        avg_loss = None
        for xs, ys in tqdm(val_set):
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            output = self.model(xs)
            _, preds = torch.max(output, 1)
            total += ys.size(0)
            correct += (preds == ys).sum().item()
            predictions.append(preds.detach().cpu().numpy())
            if(eval_loss):
                losses.append(F.nll_loss(F.log_softmax(output, dim=1), ys).cpu().item() if self.use_logit else F.nll_loss(output, ys).cpu().item())
        if eval_loss: avg_loss = np.asarray(losses).mean()
        return correct / total, avg_loss

    def train(self, val_data=None):
        best_accuracy = 0
        best_epoch = 0
        platou_counter = 0
        for epoch in range(self.total_epochs):
            real_epoch = epoch + self.start_epoch
            start_time = time.time()
            self.__run_epoch(epoch)
            epoch_time = time.time() - start_time

            if self.scheduler is not None: self.scheduler.step()

            if (real_epoch+1) % self.saving_stride == 0:
                self.save_model_state(real_epoch, self.save_id)

            if DEBUG == 1: print(f"Epoch {real_epoch+1} completed in {epoch_time:.2f} seconds")
            
            if(self.use_wandb):
                wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=real_epoch)
                norms = []
                for p in self.model.parameters():
                    if p.grad is not None: norms.append(torch.norm(p.grad.detach()))
                wandb.log({"grad_norm": torch.norm(torch.stack(norms))}, step=real_epoch)
                wandb.log({"weight_norm": torch.norm(torch.stack([torch.norm(p.detach()) for p in self.model.parameters()]))}, step=real_epoch)
                wandb.log({"train_accuracy": self.evaluate_model(self.train_data)[0]}, step=real_epoch)
            if val_data is not None:
                accuracy, val_loss = self.evaluate_model(val_data, eval_loss=True)
                self.accuracies.append(accuracy)
                if DEBUG == 1: print(f"Validation accuracy at epoch {real_epoch+1}: {accuracy:.4f}")
                if(self.use_wandb):
                    wandb.log({"acc": accuracy}, step=real_epoch)
                    wandb.log({"val_loss": val_loss}, step=real_epoch)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.best_epoch = real_epoch
                    self.best_params = self.model.state_dict()
                    self.best_optimizer = self.optimizer.state_dict()
                    self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
                    platou_counter = 0
                else:
                    platou_counter += 1
                    if(self.patience > 0 and platou_counter > self.patience):
                        print("model encountered platou. Saving best parameters...")
                        self.save_best_model_state(self.save_id)
                        return

    def save_best_model_state(self, id):
        print("saving best model...")
        os.makedirs("model_weights", exist_ok=True)
        torch.save({
            'model_state_dict': self.best_params,
            'optimizer_state_dict': self.best_opti_params,
            'epoch': self.best_epoch,
            'scheduler_state_dict': self.best_sch_params,
        }, f'model_weights/{id}_best_model_weights_epoch{self.best_epoch}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

    def save_model_state(self, epoch, id=""):
        if DEBUG == 1: print("saving model...")
        os.makedirs("model_weights", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
        }, f'model_weights/{id}_model_weights_epoch{epoch}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

    def load_model_state(self, file_path):
        print("loading model...")
        checkpoint = torch.load(file_path, map_location=torch.device(self.gpu_id))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        if self.scheduler is not None: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


