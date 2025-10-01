import cv2
import numpy as np
import torch

def tensor_to_cv2_image(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Converts a PyTorch tensor or numpy array to a BGR numpy array for OpenCV."""
    
    if isinstance(image_tensor, torch.Tensor):
        img_tensor = image_tensor.cpu()
        # Handle different tensor shapes
        if img_tensor.dim() == 3:  # (C, H, W)
            img_np = img_tensor.permute(1, 2, 0).numpy()
        elif img_tensor.dim() == 4:  # (B, C, H, W)
            img_np = img_tensor[0].permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.numpy()
        
        # Normalize to 0-255 range if needed
        if img_np.max() <= 1.0 and img_np.min() >= 0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = ((img_np*std) + mean)*255  # Unnormalize
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = image_tensor.numpy().asarray().astype(np.uint8)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return cv2.UMat(img_np)