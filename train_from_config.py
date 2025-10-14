import yaml
import sys
import os

import torch

from datetime import datetime

import wandb
from loguru import logger

from model.builder import build_model_from_cfg

from training import optimizer,loss
from training.multimodal import DualModalityTrainer
from training.unimodal import Trainer
from training.scheduler import scheduler_builder

import dataset.dataset_builder as dataset_builder
from dataset.dsec import DSECDataset, collate_ssl
from dataset import dataloader_builder as dl_builder
from utils.helpers import DEBUG, set_seed
from utils import argparser as argp
from evaluator import eval_builder



def init_wandb(cfg):
    wandb_log = False
    aug = "aug" if cfg['dataset'].get('use_augmentations', False) else  "noaug"
    schedl_name = cfg['scheduler']['name'] if 'name' in cfg['scheduler'].keys() else ""
    if cfg.get('dual_modality', True):
        run_name = cfg['model1']['backbone']['name'] if ('name' in cfg['model1']['backbone'].keys() and cfg['model1']['backbone']['name'] != '') else "model1"
        run_name += f"_{cfg['model2']['backbone']['name']}" if ('name' in cfg['model2']['backbone'].keys() and cfg['model2']['backbone']['name'] != '') else "_model2"
    else:
        run_name = cfg['model']['backbone']['name'] if ('name' in cfg['model']['backbone'].keys() and cfg['model']['backbone']['name'] != '') else model.get_name()
    type_ = "uni"
    if cfg.get('dual_modality', True):
        type_ = "dual" if cfg['dual_modality'] else "uni"
    else:
        type_ += "_event" if 'events' in cfg['dataset']['outputs'] else "_rgb"
    run_name = f"{run_name}_{type_}_{cfg['optimizer']['name']}_{schedl_name}_{aug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if 'logger' in cfg.keys() and 'name' in cfg['logger'].keys():
        if(cfg['logger']['name'] == 'wandb'):
            wandb_cfg = cfg['logger']
            assert 'project' in wandb_cfg.keys(), "specify 'project' wandb param"
            assert 'entity' in wandb_cfg.keys(), "specify 'entity' wandb param"
            wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], name=run_name, config=cfg,settings=wandb.Settings(init_timeout=600))

            wandb_log = True

    cfg['run_name'] = run_name
    return wandb_log


if __name__ == "__main__":
    # Config loading && argument parser #
    cfg, pretrained_checkpoint = argp.parse_arguments()
    CFG = cfg.copy() # keep read-only copy of cfg
    set_seed(cfg)

    model = build_model_from_cfg(cfg)
    
    # MULTI MODALITY LOSS
    criterion = loss.build_from_config(cfg)
    if isinstance(model, tuple):
        params = list(model[0].parameters()) + list(model[1].parameters())
    opti = optimizer.build_from_config(model, criterion, cfg)

    train_ds, test_ds = dataset_builder.build_from_config(cfg)
    # Dataloader (CMDA)
    train_dl, test_dl = dl_builder.build_from_config(train_ds, test_ds, cfg)

    wandb_log = init_wandb(cfg)

    # Device
    if torch.cuda.is_available():
        device = cfg['device'] if 'device' in cfg.keys() else "cuda"
    else:
        device = "cpu"
    if DEBUG >=1: logger.info(f"Using device: {device}")

    if cfg.get('dual_modality', True):
        model[0].to(device)
        model[1].to(device)
    else:   
        model.to(device)    

    # Scheduler
    schdlr = None
    if 'scheduler' in cfg.keys() and 'name' in cfg['scheduler'].keys() and cfg['scheduler']['name'] is not None:
        schdlr = scheduler_builder(opti, cfg['scheduler'])
        assert schdlr is not None, "Error - scheduler not correctly defined"
    
    # Trainer
    assert 'trainer' in cfg.keys(), "'trainer' params list missing from config file "
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if cfg['dual_modality']:
        trainer = DualModalityTrainer(model, train_dl, opti, criterion, device, CFG, root_folder=dir_path, wandb_log=wandb_log, pretrained_checkpoint=pretrained_checkpoint)
    else:
        trainer = Trainer(model,train_dl, opti, device,  CFG, root_folder=dir_path, wandb_log=wandb_log, pretrained_checkpoint=pretrained_checkpoint, scheduler=schdlr)

    evaluator = eval_builder.build_from_config(test_dl, cfg)
    trainer.train(evaluator=evaluator)
