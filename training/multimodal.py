import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm
from loguru import logger
import numpy as np

from utils.helpers import DEBUG, Timing, deep_dict_equal
import time
from datetime import datetime
import sys
import os
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .unimodal import Trainer, get_loss_keys_model


class DualModalityTrainer(Trainer):
    """
    Trainer for dual modality models (e.g., RGB + Events).
    Inherits from Trainer and overrides only the dual-modality specific methods.
    """
    
    def __init__(self, model, dataloader, optimizer, criterion, device, cfg, root_folder, wandb_log=False, scheduler=None, patience=-1, pretrained_checkpoint=None):
        """
        Initialize DualModalityTrainer.
        Args:
            model: Either a tuple/list of (model1, model2) for separate models,
                    or a single dual-modality model
            dataloader: Training dataloader
            optimizer: Optimizer instance
            criterion: Contrastive loss function for aligning modalities
            device: Device to train on
            cfg: Configuration dictionary
            root_folder: Root folder for saving checkpoints
            wandb_log: Whether to log to wandb
            scheduler: Learning rate scheduler (optional)
            patience: Early stopping patience (optional)
            pretrained_checkpoint: Checkpoint to resume from (optional)
        """
        # Handle dual model architecture
        if isinstance(model, (tuple, list)):
            self.model1, self.model2 = model
            # For compatibility with parent class, set self.model to model1
            model = self.model1
        else:
            # Single dual-modality model
            self.model1 = model
            self.model2 = None

        self.criterion = criterion

        self._get_loss_keys()
        # Store criterion for contrastive loss between modalities
        self.feature = cfg['multi_modality_loss'].get('features', 'preflatten_feat')
        if self.feature == 'preflatten_feat':
            logger.warning(f"Using feature '{self.feature}' for contrastive loss between modalities, ensure same dimensions in both models feature outputs.")

        # Call parent constructor
        super().__init__(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            root_folder=root_folder,
            wandb_log=wandb_log,
            scheduler=scheduler,
            patience=patience,
            pretrained_checkpoint=pretrained_checkpoint
        )
        
        # Override input type for dual modality
        self.input_type = 'dual'  # Both RGB and events
        
        # Update model name for saving
        if self.save_name is not None and self.model2 is not None:
            model_name = f"{self.model1.__class__.__name__}_{self.model2.__class__.__name__}"
            self.save_name = wandb.run.name if self.wandb_log else f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store best parameters for both models
        if self.model2 is not None:
            self.best_params2 = self.model2.state_dict()
        
        if DEBUG >= 1:
            logger.info(f"DualModalityTrainer initialized with {'two separate models' if self.model2 else 'single dual-modality model'}") # TODO Can be

    def _train_step(self, batch):
        """
        Single training step for dual modality.
        Extracts both RGB and event data, forward through models, and computes contrastive loss.
        
        Args:
            batch: Batch of data containing 'image' and 'events'
            
        Returns:
            List of losses [contrastive_loss]
        """
        # Extract modalities from batch
        rgbs = torch.stack([item["image"] for item in batch]).to(self.device)
        events = torch.stack([item["events"] for item in batch]).to(self.device)
        targets = torch.stack([item["BB"] for item in batch]).to(self.device)
        self.optimizer.zero_grad()
        if self.model2 is not None:
            # Two separate models approach
            out_dict1 = self.model1(rgbs, targets)
            out_dict2 = self.model2(events, targets)
            
        else:
            # Single dual-modality model
            out_dict1 = self.model1(rgbs, events)
            out_dict2 = None
        
        tot_loss1 = out_dict1['total_loss']
        tot_loss2 = out_dict2['total_loss'] if out_dict2 is not None else None
        losses_1 = out_dict1['losses']
        losses_2 = out_dict2['losses'] if out_dict2 is not None else None
        bb_out = out_dict1['backbone_features'][self.feature], out_dict2['backbone_features'][self.feature] if out_dict2 is not None else None
        # Compute loss between modalities
        bb_loss = self.criterion(*bb_out)
        tot_loss = bb_loss + tot_loss1 + (tot_loss2 if tot_loss2 is not None else 0)
        tot_loss.backward()
        self.optimizer.step()

        # Log step
        if DEBUG >= 1:
            logger.info(f"Contrastive loss: {bb_loss.item():.4f}")
        step_dict = {key: v for key, v in zip(["batch(sum)"]+self.losses_keys, [tot_loss]+[bb_loss]+list(losses_1.values()))} # THIS dict order total, multimodal, model1 losses, model2 losses
        step_dict["step"] = self.step
        self._log(step_dict, 'loss')

        return [tot_loss] + [bb_loss] + list(losses_1.values()) + list(losses_2.values()) if losses_2 is not None else []

    def train(self, evaluator=None, eval_loss=False):
        """
        Main training loop for dual modality.
        Can optionally evaluate both models separately.
        """
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, self.total_epochs):
            self.model1.train()
            if self.model2 is not None:
                self.model2.train()
            
            start_time = time.time()
            
            with tqdm(total=len(self.dataloader), desc=f"Epoch {self.epoch}/{self.total_epochs}") as pbar:
                self.model1.train()
                if self.model2 is not None:
                    self.model2.train()
                avg_loss = self._train_epoch(pbar)
            
            epoch_time = time.time() - start_time
            
            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint at epoch intervals
            if (self.checkpoint_interval_epochs > 0 and (epoch + 1) % self.checkpoint_interval_epochs == 0):
                if self.save_folder is not None:
                    self._save_checkpoint(epoch)
                else:
                    logger.warning("The model will not be saved - saving folder need to be specified")
            
            if DEBUG >= 1:
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            
            # Log learning rate
            self._log({"lr": self.optimizer.param_groups[0]['lr'], "epoch": self.epoch})
            
            # Evaluation
            if evaluator is not None:
                if self.model2 is not None:
                    # Evaluate both models separately
                    stats1, classAP1, classAR1 = evaluator.evaluate(self.model1)
                    stats2, classAP2, classAR2 = evaluator.evaluate(self.model2)
                    
                    if DEBUG >= 1:
                        logger.info(f"Model1 (RGB) - AP50-95: {stats1[0]:.4f}, AP50: {stats1[1]:.4f}")
                        logger.info(f"Model2 (Event) - AP50-95: {stats2[0]:.4f}, AP50: {stats2[1]:.4f}")
                    
                    # Log both models' metrics
                    self._log_coco_metrics(stats1, classAP1, classAR1, prefix="model1_rgb")
                    self._log_coco_metrics(stats2, classAP2, classAR2, prefix="model2_event")
                else:
                    # Evaluate single dual-modality model
                    stats, classAP, classAR = evaluator.evaluate(self.model1)
                    
                    if DEBUG >= 1:
                        ap50_95, ap50 = stats[0], stats[1]
                        logger.info(f"AP50-95: {ap50_95:.4f}, AP50: {ap50:.4f}")
                    
                    self._log_coco_metrics(stats, classAP, classAR)
            
            # Check for improvement and update patience counter
            if avg_loss < self.best_loss:
                if DEBUG >= 1:
                    logger.success(f"New best loss: {avg_loss:.4f} at epoch {self.epoch}")
                
                self.best_loss = avg_loss
                self.best_epoch = self.epoch
                self.best_params = self.model1.state_dict()
                if self.model2 is not None:
                    self.best_params2 = self.model2.state_dict()
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
        """Save the best model checkpoint (overrides parent to handle two models)."""
        save_path = f"{self.save_best_dir}{self.save_name}_best.pth"
        
        model_state = {
            'model1': self.best_params
        }
        if self.model2 is not None:
            model_state['model2'] = self.best_params2
        
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.best_optimizer,
            'scheduler_state_dict': self.best_sch_params,
            'config': self.cfg
        }, save_path)
        logger.success(f"Saved best model to {save_path}")
    
    def _save_checkpoint(self, epoch):
        """Save a training checkpoint (overrides parent to handle two models)."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        checkpoint_path = f"{self.save_folder}{self.save_name}_checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        model_state = {
            'model1': self.model1.state_dict()
        }
        if self.model2 is not None:
            model_state['model2'] = self.model2.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion': self.criterion.params if hasattr(self.criterion, 'params') else None,
            'config': self.cfg
        }, checkpoint_path)
        logger.success(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def load_model_state(self, file_path):
        """Load model state from checkpoint (overrides parent to handle two models)."""
        logger.info("Loading model...")
        checkpoint = torch.load(file_path, map_location=torch.device(self.device))
        
        model_state = checkpoint['model_state_dict']
        
        if isinstance(model_state, dict) and 'model1' in model_state:
            # Load both models
            self.model1.load_state_dict(model_state['model1'])
            if self.model2 is not None and 'model2' in model_state:
                self.model2.load_state_dict(model_state['model2'])
        else:
            # Backward compatibility - single model state
            self.model1.load_state_dict(model_state)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def _log_coco_metrics(self, stats, classAP=None, classAR=None, prefix=""):
        """Log COCO evaluation metrics with optional prefix for dual models."""
        if self.wandb_log:
            prefix_str = f"{prefix}/" if prefix else ""
            wandb.log({
                f"{prefix_str}AP/AP(50_95)": stats[0],
                f"{prefix_str}AP/AP50": stats[1],
                f"{prefix_str}AP/AP75": stats[2],
                f"{prefix_str}AP/APs": stats[3],
                f"{prefix_str}AP/APm": stats[4],
                f"{prefix_str}AP/APl": stats[5],
                f"{prefix_str}AR/AR1": stats[6],
                f"{prefix_str}AR/AR10": stats[7],
                f"{prefix_str}AR/AR100": stats[8],
                f"{prefix_str}AR/ARs": stats[9],
                f"{prefix_str}AR/ARm": stats[10],
                f"{prefix_str}AR/ARl": stats[11],
                "epoch": self.epoch
            })
            if classAP is not None:
                classAP_prefixed = {f"{prefix_str}{k}": v for k, v in classAP.items()}
                classAP_prefixed["epoch"] = self.epoch
                wandb.log(classAP_prefixed)
            if classAR is not None:
                classAR_prefixed = {f"{prefix_str}{k}": v for k, v in classAR.items()}
                classAR_prefixed["epoch"] = self.epoch
                wandb.log(classAR_prefixed)


    def _get_loss_keys(self):
        lmodel1 = get_loss_keys_model(self.model1)
        lmodel2 = get_loss_keys_model(self.model2) if self.model2 is not None else []
        for i, k in enumerate(lmodel1):
            lmodel1[i] = f"model1/{k}"
        for i, k in enumerate(lmodel2):
            lmodel2[i] = f"model2/{k}"
        mm_loss_name = str(self.criterion.__class__.__name__).lower()
        self.losses_keys = ['multimodal_'+mm_loss_name] + lmodel1 + lmodel2