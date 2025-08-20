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
from .custom import CustomDataset
from helpers import DEBUG

import os
import tqdm



class CityscapesDataset(CustomDataset):
    """Cityscapes dataset with enhanced path support."""

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')
    DSEC_DET_CLASSES = { # map to DSEC detection classes, order of labels is important, correspond to the same as dsec. so it needed to be the same index.
        11: "pedestrian", #the person class
        12: "rider", 
        13: "car",
        15: "bus",
        14: "truck",
        18: "bicycle",
        17: "motorcycle",
        16: "train",
        "pedestrian": 0,
        "rider": 1,
        "car": 2,
        "bus": 3,
        "truck": 4,
        "bicycle": 5,
        "motorcycle": 6,
        "train": 7
    }
    
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 split=None,  # New parameter for list of paths
                 **kwargs):
        # Store path list if provided
        self.path_list = []
        with open(split) as f:
            self.path_list = [line.strip() for line in f.readlines()]
        if len(self.path_list) == 0:
            raise ValueError(f"No paths found in the provided split file: {split}")
        
        # Extract bounding box parameters if provided
        bbox_ann_suffix = kwargs.get('bbox_ann_suffix', '.json')
        load_bboxes = kwargs.get('load_bboxes', False)
        kwargs["DETECTION_CLASSES"] = self.DSEC_DET_CLASSES if kwargs.get('custom_classes', False) else None
        
        super(CityscapesDataset, self).__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        os.makedirs(imgfile_prefix, exist_ok=True)
        result_files = []
        prog_bar = tqdm.tqdm(total=len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            if to_label_id:
                for label_id, label in CSLabels.id2label.items():
                    palette[label_id] = label.color
            else:
                palette = np.array(self.PALETTE, dtype=np.uint8)

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def visualize_segmentation(self, image, segmentation_mask, bboxes=None, 
                              save_path=None, show_labels=True, alpha=0.7,
                              bbox_color='red', bbox_thickness=2):
        """Visualize segmentation mask with optional bounding boxes.
        
        Args:
            image (np.ndarray or PIL.Image): Input image
            segmentation_mask (np.ndarray): Segmentation mask with class IDs
            bboxes (list, optional): List of bounding boxes in format [x1, y1, x2, y2, class_id]
            save_path (str, optional): Path to save the visualization
            show_labels (bool): Whether to show class labels
            alpha (float): Transparency for segmentation overlay
            bbox_color (str): Color for bounding boxes
            bbox_thickness (int): Thickness of bounding box lines
            
        Returns:
            PIL.Image: Visualized image
        """
        # Convert image to PIL if it's a numpy array
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Create segmentation overlaywalk
        seg_overlay = self._create_segmentation_overlay(segmentation_mask, alpha)
        
        # Blend the segmentation overlay with the original image
        vis_image = Image.blend(vis_image.convert('RGBA'), seg_overlay, alpha)
        vis_image = vis_image.convert('RGB')
        
        # Add bounding boxes if provided
        if bboxes is not None:
            vis_image = self._draw_bounding_boxes(vis_image, bboxes, bbox_color, 
                                                bbox_thickness, show_labels)
        
        # Save if path is provided
        if save_path:
            vis_image.save(save_path)
            print(f"Visualization saved to: {save_path}")
            
        return vis_image
    
    def _create_segmentation_overlay(self, segmentation_mask, alpha=0.7):
        """Create colored segmentation overlay."""
        colored_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
        
        # Apply colors based on class palette
        for class_id in range(len(self.CLASSES)):
            if class_id < len(self.PALETTE):
                mask = segmentation_mask == class_id
                colored_mask[mask] = self.PALETTE[class_id]
        
        # Convert to PIL and add alpha channel
        overlay = Image.fromarray(colored_mask).convert('RGBA')
        # Make overlay semi-transparent
        overlay_data = np.array(overlay)
        overlay_data[:, :, 3] = int(255 * alpha)  # Set alpha channel
        overlay = Image.fromarray(overlay_data)
        
        return overlay
    
    def _draw_bounding_boxes(self, image, bboxes, color='red', thickness=2, show_labels=True):
        """Draw bounding boxes on image."""
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        for bbox in bboxes:
            if len(bbox) >= 4:  # At least x1, y1, x2, y2
                x1, y1, x2, y2 = bbox[:4]
                class_id = bbox[4] if len(bbox) > 4 else None
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
                
                # Draw class label if available and requested
                if show_labels and class_id is not None and class_id < len(self.CLASSES):
                    label = self.CLASSES[class_id]
                    
                    # Calculate text size and position
                    if font:
                        bbox_font = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox_font[2] - bbox_font[0]
                        text_height = bbox_font[3] - bbox_font[1]
                    else:
                        text_width = len(label) * 8  # Approximate
                        text_height = 12
                    
                    # Draw background rectangle for text
                    text_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
                    draw.rectangle(text_bg, fill=color)
                    
                    # Draw text
                    draw.text((x1 + 2, y1 - text_height - 2), label, 
                             fill='white', font=font)
        
        return image
    
    def visualize_with_matplotlib(self, image, segmentation_mask, bboxes=None, 
                                save_path=None, figsize=(12, 8)):
        """Visualize using matplotlib for more advanced plotting."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        colored_mask = self._create_segmentation_overlay(segmentation_mask, alpha=1.0)
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Combined visualization
        overlay = self.visualize_segmentation(image, segmentation_mask, bboxes, alpha=0.6)
        axes[2].imshow(overlay)
        axes[2].set_title('Combined Visualization')
        axes[2].axis('off')
        
        # Add bounding boxes to the combined view if provided
        if bboxes is not None:
            for bbox in bboxes:
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    class_id = bbox[4] if len(bbox) > 4 else None
                    
                    # Create rectangle patch
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    axes[2].add_patch(rect)
                    
                    # Add label
                    if class_id is not None and class_id < len(self.CLASSES):
                        axes[2].text(x1, y1-5, self.CLASSES[class_id], 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                                   fontsize=10, color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matplotlib visualization saved to: {save_path}")
        
        plt.show()
        return fig
    
    def create_legend(self, save_path=None):
        """Create a legend showing class colors."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create color patches for each class
        patches_list = []
        for i, (class_name, color) in enumerate(zip(self.CLASSES, self.PALETTE)):
            color_normalized = [c/255.0 for c in color]  # Normalize to 0-1 range
            patch = patches.Patch(color=color_normalized, label=class_name)
            patches_list.append(patch)
        
        # Create legend
        ax.legend(handles=patches_list, loc='center', ncol=3, 
                 frameon=False, fontsize=10)
        ax.axis('off')
        
        plt.title('Cityscapes Dataset Class Legend', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Legend saved to: {save_path}")
            
        plt.show()
        return fig

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None,
                 efficient_test=False):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(
                super(CityscapesDataset,
                      self).evaluate(results, metrics, logger, efficient_test))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')

        if DEBUG>=1: print('Evaluating in Cityscapes style')

        result_files, tmp_dir = self.format_results(results, imgfile_prefix)

        if tmp_dir is None:
            result_dir = imgfile_prefix
        else:
            result_dir = tmp_dir.name

        eval_results = dict()
        if DEBUG>=1: print(f'Evaluating results under {result_dir} ...')

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for root, dirs, files in os.walk(self.ann_dir):
            for file in files:
                if file.endswith('gtFine_labelIds.png'):
                    seg_map = osp.relpath(osp.join(root, file), self.ann_dir)
                    seg_map_list.append(osp.join(self.ann_dir, seg_map))
                    pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Enhanced annotation loading with support for path lists and flexible text files."""
        img_infos = []
        
        # Priority 1: Use path_list if provided
        if self.path_list is not None:
            return self._load_from_path_list(img_dir, img_suffix, ann_dir, seg_map_suffix)
        
        # Priority 2: Use split file if provided
        elif split is not None:
            return self._load_from_split_file(img_dir, img_suffix, ann_dir, seg_map_suffix, split)
        
        # Priority 3: Load all files from directory
        else:
            return self._load_from_directory(img_dir, img_suffix, ann_dir, seg_map_suffix)

    def _load_from_path_list(self, img_dir, img_suffix, ann_dir, seg_map_suffix):
        """Load annotations from a list of paths."""
        img_infos = []
        
        for path_item in self.path_list:
            if isinstance(path_item, str):
                # Single path string
                img_path = path_item
            elif isinstance(path_item, dict):
                # Dictionary with 'img' key and optional 'ann' key
                img_path = path_item['img']
            else:
                continue
            
            # Determine if it's a full path or relative name
            if osp.isabs(img_path):
                # Full absolute path
                filename = osp.basename(img_path)
                img_info = dict(filename=filename, full_path=img_path)
            else:
                # Relative path/name
                filename = img_path if img_path.endswith(img_suffix) else img_path + img_suffix
                img_info = dict(filename=filename)
            
            # Handle annotation
            if ann_dir is not None:
                if isinstance(path_item, dict) and 'ann' in path_item:
                    # Custom annotation path provided
                    ann_path = path_item['ann']
                    if osp.isabs(ann_path):
                        seg_map = osp.basename(ann_path)
                        img_info['ann'] = dict(seg_map=seg_map, full_path=ann_path)
                    else:
                        img_info['ann'] = dict(seg_map=ann_path)
                else:
                    # Generate annotation path from image path
                    base_name = osp.splitext(filename)[0]
                    if img_suffix in base_name:
                        base_name = base_name.replace(img_suffix.replace('.', ''), '')
                    seg_map = base_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
            
            img_infos.append(img_info)
        
        if DEBUG >= 1:
            print(f'Loaded {len(img_infos)} images from path list')
        return img_infos

    def _load_from_split_file(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Enhanced split file loading with support for full paths."""
        img_infos = []
        
        with open(split) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                # Check if line contains full path or just filename
                if osp.isabs(line):
                    # Full path provided
                    img_path = line
                    filename = osp.basename(img_path)
                    img_info = dict(filename=filename, full_path=img_path)
                    
                    # Generate annotation path
                    if ann_dir is not None:
                        base_name = osp.splitext(filename)[0]
                        if img_suffix.replace('.', '') in base_name:
                            base_name = base_name.replace(img_suffix.replace('.', ''), '')
                        seg_map = base_name + seg_map_suffix
                        
                        # Try to find annotation in same directory as image or use ann_dir
                        img_dir_path = osp.dirname(img_path)
                        ann_full_path = osp.join(img_dir_path.replace('images', 'annotations'), seg_map)
                        if osp.exists(ann_full_path):
                            img_info['ann'] = dict(seg_map=seg_map, full_path=ann_full_path)
                        else:
                            img_info['ann'] = dict(seg_map=seg_map)
                else:
                    # Relative name (original behavior)
                    img_name = line
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                
                img_infos.append(img_info)
        
        if DEBUG >= 1:
            print(f'Loaded {len(img_infos)} images from split file: {split}')
        return img_infos

    def _load_from_directory(self, img_dir, img_suffix, ann_dir, seg_map_suffix):
        """Original directory loading method."""
        img_infos = []
        
        for entry in os.scandir(img_dir):
            if entry.is_file() and entry.name.endswith(img_suffix):
                img_info = dict(filename=entry.name)
                if ann_dir is not None:
                    seg_map = entry.name.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        
        if DEBUG >= 1:
            print(f'Loaded {len(img_infos)} images from directory: {img_dir}')
        return img_infos

    def prepare_train_img(self, idx):
        """Enhanced image preparation with full path support."""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx) if not self.test_mode else None
        results = dict(img_info=img_info, ann_info=ann_info, idx=idx)
        
        # Override img_prefix if full_path is provided
        if 'full_path' in img_info:
            results['img_prefix'] = osp.dirname(img_info['full_path'])
        
        # Override seg_prefix if annotation has full_path
        if ann_info and 'full_path' in ann_info:
            results['seg_prefix'] = osp.dirname(ann_info['full_path'])
        
        self.pre_pipeline(results)
        return results

    def prepare_test_img(self, idx):
        """Enhanced test image preparation with full path support."""
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info, idx=idx)
        
        # Override img_prefix if full_path is provided
        if 'full_path' in img_info:
            results['img_prefix'] = osp.dirname(img_info['full_path'])
        
        self.pre_pipeline(results)
        return results
