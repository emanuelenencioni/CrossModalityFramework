import torch
from loguru import logger
from utils.helpers import DEBUG
from torch.optim.lr_scheduler import (
    StepLR, 
    MultiStepLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR
)


def scheduler_builder(optimizer, cfg):
    """Build learning rate scheduler from config."""
    if cfg.get('scheduler', None) is not None:
        scheduler_cfg = cfg.get('scheduler')
    else:
        scheduler_cfg = cfg
    scheduler_name = scheduler_cfg.get('name', 'StepLR')
    
    if scheduler_name == 'MultiStepLRWithWarmup':
        # Get parameters
        milestones = scheduler_cfg.get('milestones', [30, 60, 90])
        gamma = scheduler_cfg.get('gamma', 0.1)
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
        warmup_factor = scheduler_cfg.get('warmup_factor', 0.1)
        
        if warmup_epochs > 0:
            # Create warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            # Create main scheduler (adjust milestones to account for warmup)
            main_scheduler = MultiStepLR(
                optimizer,
                milestones=[m - warmup_epochs for m in milestones if m > warmup_epochs],
                gamma=gamma
            )
            
            # Combine them
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            
            if DEBUG >= 1:
                logger.info(f"Created MultiStepLR with {warmup_epochs} epoch warmup")
                logger.info(f"Milestones (after warmup): {milestones}")
        else:
            # No warmup, just use MultiStepLR
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            if DEBUG >= 1:
                logger.info(f"Created MultiStepLR without warmup")
        
        return scheduler
    
    elif scheduler_name == 'MultiStepLR':
        # Standard MultiStepLR without warmup
        milestones = scheduler_cfg.get('milestones', [30, 60, 90])
        gamma = scheduler_cfg.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        if DEBUG >= 1:
            logger.info(f"Created MultiStepLR with milestones={milestones}, gamma={gamma}")
        
        return scheduler
    
    elif scheduler_name == 'StepLR':
        step_size = scheduler_cfg.get('step_size', 30)
        gamma = scheduler_cfg.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        if DEBUG >= 1:
            logger.info(f"Created StepLR with step_size={step_size}, gamma={gamma}")
        
        return scheduler
    
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = scheduler_cfg.get('T_max', cfg['trainer']['epochs'])
        eta_min = scheduler_cfg.get('eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        if DEBUG >= 1:
            logger.info(f"Created CosineAnnealingLR with T_max={T_max}")
        
        return scheduler
    
    elif scheduler_name == 'ReduceLROnPlateau':
        mode = scheduler_cfg.get('mode', 'min')
        factor = scheduler_cfg.get('factor', 0.1)
        patience = scheduler_cfg.get('patience', 10)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        
        if DEBUG >= 1:
            logger.info(f"Created ReduceLROnPlateau with patience={patience}")
        
        return scheduler
    
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}. Using StepLR as default.")
        return StepLR(optimizer, step_size=30, gamma=0.1)