# Took from CMDA - https://github.com/CMDA/CMDA

import os.path as osp
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch

#from .builder import DATASETS
from .cityscapes import CityscapesDataset
from utils.helpers import DEBUG

import os
import tqdm



class CityscapesEventDataset(CityscapesDataset):
    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs):
        # Extract bounding box parameters if provided
        super(CityscapesEventDataset, self).__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            **kwargs)