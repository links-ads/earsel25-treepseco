import numpy as np
import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from pathlib import Path
from torchmetrics import detection
from .components.custom_rcnn import CustomRcnn
from src.utils import Visualizer, configurable, unnormalize_image

class ClsHeadLitModule(pl.LightningModule):
    """
    This lit module can be used to train the mlpcls head of the tree pseco model
    or to perform full pipline inference.
    Use another SAMPointDecoderPretrainLitModule to train the point decoder.
    """
    @configurable
    def __init__(
        self,
        custom_rcnn: CustomRcnn,
        num_roi_head_classifier_train_anchors: int,
        roi_head_classifier_train_pos_ratio: float,
        learning_rate: float,
        weight_decay: float,
        check_val_every_n_epoch: int,
        scheduler_patience: int,
        viz_train_images_every_n_epochs = 5,
        viz_n_train_images = 10,
        viz_val_images_every_n_epochs = 5,
        viz_n_val_images = 10,
        visualize_random_images = True,
        freeze_backbone_body: bool = False,
        ):
        super().__init__()
        
        self.custom_rcnn = custom_rcnn
        
        # --- Freeze Backbone Body ---
        if freeze_backbone_body:
            print("Freezing backbone body (ResNet)...")
            # Access the backbone's body submodule
            if hasattr(self.custom_rcnn.backbone, 'body'):
                for name, param in self.custom_rcnn.backbone.body.named_parameters():
                    param.requires_grad = False
                
                if hasattr(self.custom_rcnn.backbone, 'fpn'):
                    print("Verifying FPN parameters remain trainable...")
                    fpn_trainable = False
                    for name, param in self.custom_rcnn.backbone.fpn.named_parameters():
                        if param.requires_grad:
                            fpn_trainable = True
                    if not fpn_trainable:
                        print("WARNING: No trainable parameters found in FPN!")
                    else:
                        print("  FPN appears trainable.")
                else:
                    print("WARNING: Backbone does not have an 'fpn' attribute to verify.")
                
            else:
                print("WARNING: CustomRCNN's backbone does not have a 'body' attribute. Freezing skipped.")
        else:
            print("Backbone body remains trainable.")
        
        # cls_head
        self.num_roi_head_classifier_train_anchors = num_roi_head_classifier_train_anchors
        self.roi_head_classifier_train_pos_ratio = roi_head_classifier_train_pos_ratio
        
        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        
        # Validation
        self.check_val_every_n_epoch = check_val_every_n_epoch
        
        # Metrics
        self.val_losses = []
        mAP_device = 'cuda'
        self.mAP = detection.mean_ap.MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            max_detection_thresholds=[50, 100, 200],
            ).to(mAP_device)
        self.mAP.warn_on_many_detections = False
        
        # Visualization
        self.viz_train_images_every_n_epochs = viz_train_images_every_n_epochs
        self.viz_n_train_images = viz_n_train_images
        self.viz_val_images_every_n_epochs = viz_val_images_every_n_epochs
        self.viz_n_val_images = viz_n_val_images
        
        self.train_visualizer = Visualizer(self.viz_train_images_every_n_epochs, self.viz_n_train_images,
                                        random_images=visualize_random_images)
        self.val_visualizer = Visualizer(self.viz_val_images_every_n_epochs, self.viz_n_val_images,
                                        random_images=visualize_random_images)
    
    @classmethod
    def from_config(cls, cfg):
        return {'num_roi_head_classifier_train_anchors': cfg.training.num_roi_head_classifier_train_anchors,
                'roi_head_classifier_train_pos_ratio': cfg.training.roi_head_classifier_train_pos_ratio,
                'learning_rate': cfg.training.learning_rate,
                'weight_decay': cfg.training.weight_decay,
                'scheduler_patience': cfg.training.scheduler_patience,
                'viz_train_images_every_n_epochs': cfg.training.viz_train_images_every_n_epochs,
                'viz_n_train_images': cfg.training.viz_n_train_images,
                'viz_val_images_every_n_epochs': cfg.training.viz_val_images_every_n_epochs,
                'viz_n_val_images': cfg.training.viz_n_val_images,
                'visualize_random_images': cfg.training.visualize_random_images,
                'check_val_every_n_epoch': cfg.training.check_val_every_n_epoch,
                'freeze_backbone_body': cfg.model.freeze_rn50_body,
                }
    
    def on_train_epoch_start(self):
        self.train_visualizer.on_epoch_start_setup(self.logger, self.trainer.num_training_batches)
    
    def training_step(self, batch, batch_idx):
        images, gt_bboxes, proposals, num_boxes, image_paths = batch.values()
        
        targets = [{"boxes": gt_bboxes[idx],
                    "labels": torch.ones((gt_bboxes[idx].shape[0]), dtype=torch.int64, device=gt_bboxes[idx].device)}
                    for idx in range(len(gt_bboxes))]
        losses = self.custom_rcnn(images, proposals, targets)
        loss = losses['loss_classifier'] + losses['loss_box_reg']
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        
        return loss
    
    def on_validation_epoch_start(self):
        self.mAP.reset()
        self.val_visualizer.on_epoch_start_setup(self.logger, self.trainer.num_val_batches[0])
    
    def validation_step(self, batch, batch_idx):
        images, gt_bboxes, proposals, num_boxes, image_paths = batch.values()
        
        detections = self.custom_rcnn(images, proposals) # list
        
        predictions = []
        targets = []
        for idx in range(len(detections)):
            detection = detections[idx]
            tree_idxes = detection['labels'] == 1
            pred_tree_boxes = detection['boxes'][tree_idxes]
            pred_tree_scores = detection['scores'][tree_idxes]
            
            prediction = {
                    'boxes': pred_tree_boxes,
                    'scores': pred_tree_scores,
                    'labels': torch.zeros_like(pred_tree_scores, dtype=torch.int32),  # Assuming single class
                    }
            
            target = {
                    'boxes': gt_bboxes[idx],
                    'labels': torch.zeros(gt_bboxes[idx].shape[0], dtype=torch.int32, device=gt_bboxes[idx].device)
                    }
            
            predictions.append(prediction)
            targets.append(target)
            
        self.mAP.update(predictions, targets)
        
        # Log visualizations
        if self.val_visualizer.visualize_batch_flag(batch_idx):
            first_image_detections = detections[0]
            
            tree_idxes = first_image_detections['labels'] == 1
            pred_tree_boxes = first_image_detections['boxes'][tree_idxes]
            pred_tree_scores = first_image_detections['scores'][tree_idxes]
            
            self._log_complete_img(image=images[0],
                                proposals=proposals[0].cpu().numpy(),
                                tree_boxes=pred_tree_boxes.cpu().numpy(),
                                gt_boxes=gt_bboxes[0].cpu().numpy(),
                                tree_boxes_scores=pred_tree_scores.cpu().numpy(),
                                prefix='val',
                                batch_idx=batch_idx)
    
    def on_validation_epoch_end(self):
        map_metrics = self.mAP.compute()
        self.log('val_mAP', map_metrics['map'], on_epoch=True, prog_bar=True)
        for key, value in map_metrics.items():
            self.log(f'val/{key}', value.item() if isinstance(value, torch.Tensor) else value)
    
    def configure_optimizers(self):
        params_to_optimize = filter(lambda p: p.requires_grad, self.custom_rcnn.parameters())
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=self.scheduler_patience,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mAP",
                "frequency": self.check_val_every_n_epoch, #! should be a multiple of check_val_every_n_epoch
            }
        }
    
    def _log_complete_img(self, image: torch.Tensor, proposals:np.ndarray, tree_boxes:np.ndarray, gt_boxes:np.ndarray, tree_boxes_scores:np.ndarray,
                        prefix: str, batch_idx=None):
        
        image_np = np.transpose(np.clip(unnormalize_image(image.detach().cpu()), 0, 1), (1, 2, 0))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        
        # 1. Plot proposals (e.g., in blue)
        self.plot_boxes_on_axis(axes[0], image_np, proposals, color='blue', title='Proposals')
        
        # 2. Plot tree_boxes (e.g., in green) with alpha based on scores
        tree_alphas = None
        if tree_boxes_scores is not None and len(tree_boxes_scores) > 0:
            tree_alphas = np.clip(tree_boxes_scores, 0.1, 1.0)
        
        self.plot_boxes_on_axis(axes[1], image_np, tree_boxes, color='red', title='Tree Boxes (alpha=score)', alphas=tree_alphas)
        
        # 3. Plot ground truth boxes (e.g., in red)
        self.plot_boxes_on_axis(axes[2], image_np, gt_boxes, color='green', title='Ground Truth Boxes', linewidth=1) # Make GT boxes thicker
        
        plt.tight_layout()
        
        batch_idx_str = f'_{batch_idx}' if batch_idx is not None else ''
        # Log to tensorboard
        self.logger.experiment.add_figure(f'{prefix}/complete{batch_idx_str}', fig, self.global_step)
        plt.close(fig)
    
    def plot_boxes_on_axis(self, ax, img_np, boxes, color, title, alphas=None, linewidth=1):
        """Helper function to display an image and plot bounding boxes on a matplotlib Axes."""
        ax.imshow(img_np)
        ax.set_title(title)
        ax.axis('off')
        
        if boxes is None or len(boxes) == 0:
            return # No boxes to plot
        
        if alphas is None:
            alphas = np.ones(len(boxes))
        elif len(alphas) != len(boxes):
            print(f"Warning: Mismatch between number of boxes ({len(boxes)}) and alpha values ({len(alphas)}). Using alpha=1.")
            alphas = np.ones(len(boxes))
        else:
            alphas = np.clip(alphas, 0.0, 1.0)
        
        for i, box in enumerate(boxes):
            # Assuming box format: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none',
                alpha=alphas[i]
            )
            
            ax.add_patch(rect)
