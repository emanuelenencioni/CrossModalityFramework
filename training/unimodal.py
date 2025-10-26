

import inspect
import datetime
import time
from tqdm import tqdm
import numpy as np
import wandb
import os
import sys
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils.helpers import DEBUG, DEBUG_EVAL, deep_dict_equal



class Trainer:
    def __init__(self, model, dataloader, optimizer, device, cfg, root_folder,wandb_log=False, scheduler=None, patience=-1, pretrained_checkpoint=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epoch = 1
        self.cfg = cfg
        self.trainer_cfg = cfg['trainer']

        assert 'epochs' in self.trainer_cfg.keys(), " specify 'epochs' trainer param"

        self.total_epochs = int(self.trainer_cfg['epochs'])
        self.input_type = 'events' if 'events' in dataloader.dataset[0] else 'image'
        if 'events' in dataloader.dataset[0] and 'image' in dataloader.dataset[0]:
            logger.warning("The dataloader contains both events_vg and image, using events_vg as input type in unimodal training mode")
        if DEBUG >= 1: logger.info(f"Input type: {self.input_type}")

        if pretrained_checkpoint is not None: # Load checkpoint infos
            if 'model_state_dict' in pretrained_checkpoint:
                self.model.load_state_dict(pretrained_checkpoint['model_state_dict'])
                if DEBUG >= 1: logger.success("Pre-trained model loaded successfully (CrossModalityFramework)")
            
            if 'optimizer_state_dict' in pretrained_checkpoint:
                self.optimizer.load_state_dict(pretrained_checkpoint['optimizer_state_dict'])
                if DEBUG >= 1: logger.success("Optimizer state loaded successfully")

            if 'scheduler_state_dict' in pretrained_checkpoint and cfg['trainer'].get('resume_scheduler', False):
                scheduler_state = pretrained_checkpoint['scheduler_state_dict']
                if scheduler_state is not None and self.scheduler is not None and \
                deep_dict_equal(cfg['scheduler'], pretrained_checkpoint['config'].get('scheduler', None)):
                    self.scheduler.load_state_dict(scheduler_state)
                    if DEBUG >= 1: logger.success("Scheduler state loaded successfully")

            if 'epoch' in pretrained_checkpoint:
                self.epoch = pretrained_checkpoint['epoch'] + 1
                self.total_epochs = int(self.trainer_cfg['epochs']) + self.epoch - 1
                if DEBUG >= 1: logger.info(f"Resuming training from epoch {self.epoch}")
        
        self.device = device
        
        # Get loss keys from model
        self._get_loss_keys()
        
        # deciding checkpoint interval based on epochs or steps
        self.checkpoint_interval_epochs = cfg['trainer'].get('checkpoint_interval_epochs', 0)
        if self.checkpoint_interval_epochs > 0:
            self.checkpoint_interval = 0
        else: 
            self.checkpoint_interval = cfg['trainer'].get('checkpoint_interval', 0)

        self.scheduler = scheduler
        self.wandb_log = True if wandb.run is not None else False
        if self.wandb_log: assert wandb.run is not None, "Wandb run must be initialized before setting wandb_log to True"

        ### Saving related ###
        self.save_folder = root_folder + "/" + self.trainer_cfg['save_folder'] if 'save_folder' in self.trainer_cfg.keys() and self.trainer_cfg['save_folder'] is not None else None
        self.save_best_dir = None
        if self.save_folder is not None:
            if self.save_folder[-1] != '/':
                self.save_folder += '/'
            self.save_best_dir = f"{self.save_folder}best/"
            os.makedirs(self.save_best_dir, exist_ok=True)
            self.save_name = wandb.run.name if self.wandb_log else f"{self.model.__class__.__name__}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            logger.warning("The model will not be saved")
            self.save_name = None
        
        # Early stopping / patience
        self.patience = patience
        self.patience_counter = 0  # Counter for epochs without improvement
        
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

        if DEBUG >= 1:
            if self.patience < sys.maxsize and self.patience > 0:
                logger.info(f"Early stopping enabled with patience={self.patience}")
            else:
                logger.info("Early stopping disabled")


    def _train_step(self, batch):
        input_frame = torch.stack([item[self.input_type] for item in batch]).to(self.device)
        targets = torch.stack([item["BB"] for item in batch]).to(self.device) #For now considering only object detection tasks
        self.model.train()
        self.optimizer.zero_grad()
        out_dict = self.model(input_frame, targets) # cause now we have: features, (head output)
        tot_loss = out_dict['total_loss']
        losses = out_dict['losses']

        tot_loss.backward()
        self.optimizer.step()

        if DEBUG >= 1: 
            logger.info(f"loss values: {[f'{k}: {v}' for k, v in losses.items()]}")
        
        step_dict = {key: v for key, v in zip(["batch(sum)"]+self.losses_keys, [tot_loss]+list(losses.values()))} # python 3.7+ maintains dict order
        step_dict["step"] = self.step
        self._log(step_dict, "loss")
        
        return [tot_loss]+list(losses.values()) # python 3.7+ maintains dict order

    def _train_epoch(self, pbar=None):
        total_losses = []
        for batch in self.dataloader:
            losses = self._train_step(batch)
            total_losses.append([loss.item() if not isinstance(loss, float) else loss for loss in losses])
            if pbar is not None:
                pbar.set_description(f"Training model {self.model.get_name()}, loss:{losses[0].item():.4f}")
                pbar.update(1)
            self.step += 1
        
        total_losses = np.array(total_losses)
        avg_losses = np.mean(total_losses, axis=0)  #
        if DEBUG >= 1: logger.info(f"Epoch loss: {avg_losses[0]:.4f}")
        epoch_dict = {key: avg_loss for key, avg_loss in zip(["epoch_l"]+self.losses_keys, avg_losses)}
        epoch_dict["epoch"] = self.epoch
        self._log(epoch_dict, "epoch")
        
        return avg_losses[0]

    def train(self, evaluator=None, eval_loss=False):
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.total_epochs):
            if DEBUG_EVAL == 0:
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
                        logger.warning("The model will not be saved - saving folder need to be specified")
                if DEBUG >= 1:
                    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
                
                self._log({"lr": self.optimizer.param_groups[0]['lr'], "epoch": self.epoch})

            if evaluator is not None:
                # stats is a numpy array of 12 elements
                stats, classAP, classAR = evaluator.evaluate(self.model)

                if DEBUG >= 1: 
                    ap50_95, ap50 = stats[0], stats[1]
                    logger.info(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")

                self._log_coco_metrics(stats, classAP, classAR)

            elif hasattr(self.dataloader.dataset, 'evaluate'):
                # Use dataset's evaluate method
                ap50_95, ap50, _ = self.dataloader.dataset.evaluate(self.model)
                if DEBUG >= 1: logger.info(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")

            # Check for improvement and update patience counter
            if avg_loss < self.best_loss:
                if DEBUG >= 1: logger.success(f"New best loss: {avg_loss:.4f} at epoch {self.epoch}")
                self.best_loss = avg_loss
                self.best_epoch = self.epoch
                self.best_params = self.model.state_dict()
                self.best_optimizer = self.optimizer.state_dict()
                self.best_sch_params = self.scheduler.state_dict() if self.scheduler is not None else None
                self.patience_counter = 0  # Reset counter on improvement
                
                if self.save_folder is not None:
                    self._save_best()
            else:
                self.patience_counter += 1
                if DEBUG >= 1:
                    logger.info(f"No improvement in loss. Patience: {self.patience_counter}/{self.patience}")
                
                # Check if patience exceeded
                if self.patience > 0 and self.patience_counter >= self.patience:
                    logger.warning(f"Early stopping triggered! No improvement for {self.patience} epochs.")
                    logger.info(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
                    break
            
            self.epoch += 1

        logger.success("Training finished.")

    def _save_best(self):
        save_path = f"{self.save_best_dir}{self.save_name}_best.pth"
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.best_params,
            'optimizer_state_dict': self.best_optimizer,
            'scheduler_state_dict': self.best_sch_params,
            'config': self.cfg
        }, save_path)
        logger.success(f"Saved best model to {save_path}")

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
        logger.success(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def load_model_state(self, file_path):
        logger.info("Loading model...")
        checkpoint = torch.load(file_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])



    def _log_coco_metrics(self, stats, classAP=None, classAR=None):
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
            if classAP is not None:
                classAP["epoch"] = self.epoch
                wandb.log(classAP)
            if classAR is not None:
                classAR["epoch"] = self.epoch 
                wandb.log(classAR)


    def _log(self, message, prefix= ""):
        if self.wandb_log:
            if prefix != "":
                prefixed_message = {}
                for k, v in message.items():
                        prefixed_message[f"{prefix}/{k}"] = v
                wandb.log(prefixed_message)
            else:
                wandb.log(message)

    def _get_loss_keys(self):
        self.losses_keys = get_loss_keys_model(self.model)


def get_loss_keys_model(model):
    if hasattr(model, 'loss_keys'):
            if DEBUG >= 1: logger.info(f"Loss keys from model attribute: {model.loss_keys}")
            return model.loss_keys
    else:
        model.train()
        with torch.no_grad():
            try:
                device = next(model.parameters()).device
                dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust shape as needed
                dummy_targets = torch.zeros(1, 10, 5).to(device)
                # Set class IDs to valid integers (e.g., 0)
                dummy_targets[:, :, 0] = 0  # All class IDs = 0
                # Set some random bbox coordinates
                dummy_targets[:, :, 1:] = torch.randn(1, 10, 4).to(device).abs() * 50  # Random positive bbox coords
                
                dummy_dict = model(dummy_input, dummy_targets)

                assert isinstance(dummy_dict['losses'], dict), "Model forward pass did not return a dict of losses"
                losses_keys = list(dummy_dict['losses'].keys()) 
                
                if DEBUG >= 1: logger.info(f"Loss keys extracted: {losses_keys}")
            except Exception as e:
                logger.warning(f"Could not extract loss keys from dummy forward pass: {e}")
                losses_keys = []
        return losses_keys