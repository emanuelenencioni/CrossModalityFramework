import yaml
import sys
import os

import torch
from torch.utils.data import DataLoader

from datetime import datetime

import wandb
import argparse
import random
import numpy as np

from model.backbone import DualModalityBackbone, UnimodalBackbone
from model.yolox_head import YOLOXHead
from model.builder import build_model_from_cfg

from training import optimizer,loss
from training.multimodal import DualModalityTrainer
from training.unimodal import Trainer
from training.scheduler import scheduler_builder

from evaluator.dsec_evaluator import DSECEvaluator
from evaluator.cityscapes_evaluator import CityscapesEvaluator
import dataset.dataset_builder as dataset_builder
from dataset.dsec import DSECDataset, collate_ssl
from utils.helpers import DEBUG, set_seed
from utils import argparser as argp


if __name__ == "__main__":
    # Config loading && argument parser #
    cfg, pretrained_checkpoint = argp.parse_arguments()

    set_seed(cfg)

    model = build_model_from_cfg(cfg)

    criterion, learnable = loss.build_from_config(cfg)

    opti = optimizer.build_from_config(model, criterion if learnable else None, cfg)
    
    train_ds, test_ds = dataset_builder.build_from_config(cfg['dataset'])
    # Dataloader (CMDA)
    assert 'dataset' in cfg.keys(), " 'dataset' params list missing from config file"
    assert 'batch_size' in cfg['dataset'].keys(), " specify 'batch_size' dataset param"
    num_workers = cfg['dataset'].get('num_workers', 2)
        
    train_dl = DataLoader(train_ds, batch_size=cfg['dataset']['batch_size'], num_workers=num_workers, shuffle=True, collate_fn=collate_ssl, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg['dataset']['batch_size'], num_workers=num_workers, shuffle=False, collate_fn=collate_ssl, pin_memory=True) if test_ds is not None else None
    
    wandb_log = False
    aug = "aug" if cfg['dataset'].get('use_augmentations', False) else  "noaug"
    schedl_name = cfg['scheduler']['name'] if 'name' in cfg['scheduler'].keys() else ""
    run_name = cfg['model']['backbone']['name'] if ('name' in cfg['model']['backbone'].keys() and cfg['model']['backbone']['name'] != '') else model.get_name()
    run_name = f"{run_name}_{cfg['optimizer']['name']}_{schedl_name}_{aug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'logger' in cfg.keys() and 'name' in cfg['logger'].keys():
        if(cfg['logger']['name'] == 'wandb'):
            wandb_cfg = cfg['logger']
            assert 'project' in wandb_cfg.keys(), "specify 'project' wandb param"
            assert 'entity' in wandb_cfg.keys(), "specify 'entity' wandb param"
            wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], name=run_name, config=cfg,settings=wandb.Settings(init_timeout=600))
            # Log model architecture
            wandb.watch(model)
            wandb_log = True

    device = cfg['device'] if 'device' in cfg.keys() else "cuda"
    model.to(device)    

    # Place this right after model.to(device)
    if pretrained_checkpoint is not None:
        model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        print("Pre-trained model loaded successfully")

    # Scheduler
    schdlr = None
    if 'scheduler' in cfg.keys() and 'name' in cfg['scheduler'].keys() and cfg['scheduler']['name'] is not None:
        schdlr = scheduler_builder(opti, cfg['scheduler'])
        assert schdlr is not None, "Error - scheduler not correctly defined"
    
    # Trainer
    assert 'trainer' in cfg.keys(), "'trainer' params list missing from config file "
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if cfg['dual_modality']:
        trainer = DualModalityTrainer(model, train_dl, opti, criterion, device, cfg, root_folder=dir_path, wandb_log=wandb_log, pretrained_checkpoint=pretrained_checkpoint)
    else:
        trainer = Trainer(model,train_dl, opti, criterion, device,  cfg, root_folder=dir_path, wandb_log=wandb_log, pretrained_checkpoint=pretrained_checkpoint, scheduler=schdlr)
    in_size = cfg['model']['backbone']['input_size']
    evaluator = CityscapesEvaluator(test_dl, img_size=(in_size, in_size), confthre=0.3, nmsthre=0.6, num_classes=cfg['dataset']['bb_num_classes'], device=device)
    trainer.train(evaluator=evaluator)
