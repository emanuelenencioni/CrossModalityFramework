import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
import os
import glob
from pathlib import Path
import cv2
import random

# Import your framework components
from model.builder import build_model_from_cfg
from model.backbone import DualModalityBackbone, UnimodalBackbone
from model.yolox_head import YOLOXHead
import dataset.dataset_builder as dataset_builder
from dataset.cityscapes import CityscapesDataset
from helpers import DEBUG
from torchvision.ops import nms, batched_nms


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def get_args_parser():
    parser = argparse.ArgumentParser('Detection script using trained models', add_help=False)
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--input_image', type=str, help='Path to single input image')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./detection_results', help='Output directory for results')
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to process (if using dataset)')
    parser.add_argument('--use_dataset', action='store_true', help='Use dataset instead of input images')
    parser.add_argument('--dataset_split', type=str, default='val', help='Dataset split to use (train/val)')
    return parser

def load_model_from_config(cfg, checkpoint_path, device):
    """Load model from config and checkpoint."""
    print(f"Loading model from config and checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']
    # Build model from config
    if 'head' in cfg['model'].keys():
        model = build_model_from_cfg(cfg['model'])
    else:
        # Backbone only model
        dual_modality = 'rgb_backbone' in cfg['model']['backbone'] and 'event_backbone' in cfg['model']['backbone']
        pretrained_weights = cfg['model']['backbone'].get('pretrained_weights', None)
        pretrained = cfg['model']['backbone'].get('pretrained', True)
        
        if dual_modality:
            model = DualModalityBackbone(
                rgb_backbone=cfg['model']['backbone']['rgb_backbone'],
                event_backbone=cfg['model']['backbone']['event_backbone'],
                embed_dim=cfg['model']['backbone']['embed_dim'],
                img_size=cfg['model']['backbone']['input_size'],
                pretrained=pretrained
            )
        else:
            backbone_name = cfg['model']['backbone'].get('rgb_backbone') or cfg['model']['backbone'].get('event_backbone')
            model = UnimodalBackbone(
                backbone_name,
                pretrained_weights=pretrained_weights,
                embed_dim=cfg['model']['backbone']['embed_dim'],
                img_size=cfg['model']['backbone']['input_size']
            )
    
    # Load checkpoint
    #checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")
    return model

def get_transform(input_size):
    """Get image transformation pipeline."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def detect_image(model, image, transform, device, confidence_threshold=0.5):
    """Run detection on a single image and apply post processing."""
    
    original_size = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    outputs, _ = model(img_tensor)
    outputs = postprocess(outputs, num_classes=8, conf_thre=confidence_threshold) # After this, boxes are in (x1, y1, x2, y2) format
    
    return outputs

def rescale_boxes(boxes, original_size):
    """Convert normalized boxes to image coordinates."""
    img_w, img_h = original_size
    
    # Convert from center format to corner format if needed
    # Assume cx, cy, w, h format and divide by 512

    #calculate scale factors
    x_scale = img_w / 512
    y_scale = img_h / 512

    boxes = boxes * torch.tensor([x_scale, y_scale, x_scale, y_scale], dtype=torch.float32)

    return boxes

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    ####### WARNING : Convert from (cx, cy, h, w) to (x1, y1, x2, y2) format -> TODO: (cx, cy, h, w) -> (cx, cy, w, h)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 3] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 2] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 3] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 2] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def visual(self,img, output, orig_size, cls_conf=0.35, classes=None):
        
        if output is None:
            return img
        output = output.cpu()

        bboxes = rescale_boxes(output[:, :4], orig_size)
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, classes)
        return vis_res

def main(args):
    """Main detection function."""
    print("Detection Script using Trained Framework Models")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    args.device = "cpu"
    # Load model
    model = load_model_from_config(cfg, args.checkpoint_path, args.device)
    
    # Get transform
    input_size = cfg['model']['backbone']['input_size']
    transform = get_transform(input_size)
    
    # Define classes (adapt based on your dataset)
    classes = [
        'person', 'rider', 'car', 'bus', 'truck', 
        'bicycle', 'motorcycle', 'train'
    ]
    
    if args.input_dir:
        # Process directory of images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        
        image_paths = sort(image_paths)
        print(f"Found {len(image_paths)} images in {args.input_dir}")
        
        for img_path in image_paths:
            print(f"\nProcessing: {img_path}")
            
            try:
                image = Image.open(img_path).convert('RGB')
                boxes, scores = detect_image(model, image, transform, args.device, args.confidence_threshold)
                
                # Save result
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(args.output_dir, f"{base_name}_detection.png")
                
                detection_count = plot_detections(image, boxes, scores, classes, output_path, args.confidence_threshold)
                print(f"Found {detection_count} detections")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    elif args.input_image:
        # Process single image
        print(f"Processing single image: {args.input_image}")
        image = Image.open(args.input_image).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output = detect_image(model, image, transform, args.device, args.confidence_threshold)
        output = output[0]

        visual_res = visual(model,image_cv, output, image.size, args.confidence_threshold, classes)
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_detection.png")
        cv2.imwrite(output_path, visual_res)
        
        #creating new image that contains original and detection side by side
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        combined_image = np.concatenate((image_cv, visual_res), axis=1)
        combined_output_path = os.path.join(args.output_dir, f"{base_name}_combined.png")
        cv2.imwrite(combined_output_path, combined_image)

        detection_count = len(output[0]) if output[0] is not None else 0
        #detection_count = plot_detections(image, boxes, scores, classes, output_path, args.confidence_threshold)
        print(f"Found {detection_count} detections")
    
    else:
        print("Please specify either --input_image, --input_dir, or --use_dataset")
        return
    
    print(f"\n✓ Detection complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Detection script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)




