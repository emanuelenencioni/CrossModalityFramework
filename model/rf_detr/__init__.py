# ------------------------------------------------------------------------
# RF-DETR Integration for CrossModalityFramework
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
RF-DETR model integration for CrossModalityFramework.
This package contains the LWDETR architecture and related components.
"""

from .models.lwdetr import LWDETR, SetCriterion

__all__ = ['LWDETR', 'SetCriterion']
