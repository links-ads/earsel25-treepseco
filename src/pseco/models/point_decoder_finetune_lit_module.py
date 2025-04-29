import torch
import numpy as np
from pytorch_msssim import MS_SSIM
from typing import Tuple
import math
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import detection
from .ops.ops import gaussian_radius
from pathlib import Path
import torchvision.ops as vision_ops

from src.utils import Visualizer, configurable, unnormalize_image
from .components import SAMPointDecoder, HeatmapBoxDetector
from ..utils.viz import viz_img_heat_seg_box

class PointDecoderFinetuneLitModule(pl.LightningModule):
    @configurable
    def __init__(
        self, 
        sam_model,
        learning_rate: float,
        weight_decay: float,
        scheduler_patience: int,
        check_val_every_n_epoch: int,
        viz_train_images_every_n_epochs = 5,
        viz_n_train_images = 10,
        viz_val_images_every_n_epochs = 5,
        viz_n_val_images = 10,
        visualize_random_images = True,
        use_sam_pretrained = True,
        point_decoder_state_dict_path = None,
        prefix_to_rmv_in_state_dict = None,
        val_oracle_point_decoder = False,
        mask_upscaling_type = 'default',
        loss_type = 'mse',
        save_boxes = False,
        save_root: str = "", 
        ):
        super().__init__()
        
        self.save_boxes = save_boxes
        self.save_root = Path(save_root)
        # Initialize models
        self.point_decoder = SAMPointDecoder(sam=sam_model, mask_upscaling_type=mask_upscaling_type, use_sam_pretrained=use_sam_pretrained)
        self.heatmap_box_detector = HeatmapBoxDetector(self.cfg, sam=sam_model)
        
        # Start from pretrained model
        if point_decoder_state_dict_path is not None:
            print(f"\n### Loading state dict from {point_decoder_state_dict_path} ###\n")
            #check if is a checkpoint
            if point_decoder_state_dict_path.endswith('.ckpt'):
                state_dict = torch.load(point_decoder_state_dict_path, map_location='cpu', weights_only=True)['state_dict']
            else:
                state_dict = torch.load(point_decoder_state_dict_path, map_location='cpu', weights_only=True)
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(prefix_to_rmv_in_state_dict):
                    new_key = key[len(prefix_to_rmv_in_state_dict):]
                    new_state_dict[new_key] = value
                else:
                    # If a key doesn't start with the prefix, keep it as is
                    new_state_dict[key] = value
            self.point_decoder.load_state_dict(new_state_dict, strict=False) 
        else:
            print(f"\n### Starting from fresh weights (NO pretrained loaded) ###\n")
        
        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.loss_type = loss_type
        if loss_type == 'ssim':
            self.ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        
        # Metrics
        self.val_losses = []
        
        # Visualization
        self.viz_train_images_every_n_epochs = viz_train_images_every_n_epochs
        self.viz_n_train_images = viz_n_train_images
        self.viz_val_images_every_n_epochs = viz_val_images_every_n_epochs
        self.viz_n_val_images = viz_n_val_images
        
        self.train_visualizer = Visualizer(self.viz_train_images_every_n_epochs, self.viz_n_train_images,
                                        random_images=visualize_random_images)
        self.val_visualizer = Visualizer(self.viz_val_images_every_n_epochs, self.viz_n_val_images,
                                        random_images=visualize_random_images)
        
        self.sigma = torch.tensor(5, device='cuda')
        
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.val_oracle_point_decoder = val_oracle_point_decoder
        
        mAP_device = 'cuda'
        self.mAP_perfect_cls_head = detection.mean_ap.MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            max_detection_thresholds=[50, 100, 200],
            ).to(mAP_device)
        self.mAP_perfect_cls_head.warn_on_many_detections = False 
    
    @classmethod
    def from_config(cls, cfg):
        cls.cfg = cfg
        return {
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "scheduler_patience": cfg.training.scheduler_patience,
        "check_val_every_n_epoch": cfg.training.check_val_every_n_epoch,
        "viz_train_images_every_n_epochs": cfg.training.viz_train_images_every_n_epochs,
        "viz_n_train_images": cfg.training.viz_n_train_images,
        "viz_val_images_every_n_epochs": cfg.training.viz_val_images_every_n_epochs,
        "viz_n_val_images": cfg.training.viz_n_val_images,
        "visualize_random_images": cfg.training.visualize_random_images,
        "use_sam_pretrained": cfg.model.point_decoder.use_sam_pretrained,
        "point_decoder_state_dict_path": cfg.model.point_decoder.point_decoder_state_dict_path if hasattr(cfg.model.point_decoder, "point_decoder_state_dict_path") else None,
        "prefix_to_rmv_in_state_dict": cfg.model.point_decoder.prefix_to_rmv_in_state_dict if hasattr(cfg.model.point_decoder, "prefix_to_rmv_in_state_dict") else None,
        "val_oracle_point_decoder": cfg.training.val_oracle_point_decoder if hasattr(cfg.training, "val_oracle_point_decoder") else False,
        "mask_upscaling_type": cfg.model.point_decoder.mask_upscaling_type,
        "loss_type": cfg.training.loss_type,
        }
    
    def forward(self, x):
        return self.point_decoder(x) 
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_visualizer.reset()
        self.val_visualizer.reset()
    
    def extract_heatmap(self, points, sigma=2):
        scale = 4
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.ones(len(points)).cuda() * sigma
        points = points / scale
        points = points.long().float()
        x = torch.arange(0, 256, 1).cuda()
        y = torch.arange(0, 256, 1).cuda()
        x, y = torch.meshgrid(x, y, indexing='xy')
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        heatmaps = torch.zeros(1, 1, 256, 256).cuda()
        for indices in torch.arange(len(points)).split(256):
            mu_x, mu_y = points[indices, 0].view(-1, 1, 1), points[indices, 1].view(-1, 1, 1)
            heatmaps_ = torch.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[indices].view(-1, 1, 1) ** 2))
            heatmaps_ = torch.max(heatmaps_, dim=0).values
            heatmaps_ = heatmaps_.reshape(1, 1, 256, 256)
            heatmaps = torch.maximum(heatmaps, heatmaps_)
        return heatmaps.float() 
    
    def gen_heatmap_from_bboxes(self, points, boxes):
        num_points = len(points)
        min_sigma = 5. #2.
        sigma = torch.ones(num_points, device=self.device) * min_sigma
        
        for i in range(len(points)):
            bbox = boxes[i] / 4
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            sigma_ = (2 * radius + 1) / 6.0
            sigma_ = max(sigma_, min_sigma)
            sigma[i] = sigma_
            
        return self.extract_heatmap(points, sigma)    
    
    def create_masks(self, bs):
        return torch.ones(bs, 1, 256, 256, dtype=torch.bool, device=self.device)
    
    def loss_step(self, batch, logits):
        
        label_heatmaps = torch.cat([
            self.gen_heatmap_from_bboxes(points, boxes) 
            for points, boxes in zip(batch['center_points'], batch['bboxes'])
        ]) 
        
        if self.loss_type == 'mse':
            masks = self.create_masks(batch['images'].shape[0]).flatten(1)
            loss = F.mse_loss(logits, label_heatmaps, reduction='none').flatten(1)
            loss = ((loss * masks.view(masks.shape[0], -1)).sum(1) / (1e-5 + masks.view(masks.shape[0], -1).sum(1))).mean()
        elif self.loss_type == 'ssim':
            loss = 1-self.ssim_loss(label_heatmaps, logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss, label_heatmaps 
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        
        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param mode: A string indicating the mode of the model. One of 'train', 'val' or 'test'.
        
        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        images = batch['images']
        
        pred_heatmaps, image_embeddings = self.forward(images)
        
        loss, label_heatmaps = self.loss_step(batch, pred_heatmaps)
        
        return loss, pred_heatmaps, image_embeddings, label_heatmaps
    
    def perfect_cls_head_boxes(self, raw_pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, iou_threshold=None):
        """For each predicted bounding box, compute IoU with ground truth boxes and select best matches, discarding others.
        
        Args:
            raw_pred_bboxes (torch.Tensor): Predicted bounding boxes tensor. (n, 4)
            gt_bboxes (torch.Tensor): Ground truth bounding boxes tensor. (m, 4)
            iou_threshold (float, optional): IoU threshold to filter boxes. Boxes with IoU below threshold are discarded.
                Defaults to None.
        
        Returns:
            torch.Tensor: Filtered predicted bounding boxes that best match ground truth boxes
            torch.Tensor: Indexes of the predicted boxes that best match ground truth boxes
        
        Notes:
            - For each gt_box, finds the pred_box with highest IoU and keeps only those
            - If iou_threshold provided, discards boxes with IoU < threshold
        """
        iou_scores = vision_ops.box_iou(gt_bboxes, raw_pred_bboxes) # (m, n)
        if iou_scores.numel() == 0:
            return torch.zeros(0, 4), torch.zeros(0)
        iou_scores, max_idxes = iou_scores.max(dim=1) # (m,)
        if iou_threshold is not None:
            max_idxes = max_idxes[iou_scores > iou_threshold]
        return raw_pred_bboxes[max_idxes], max_idxes # pred boxes that have the highest IoU with gt boxes (avoid multiple pred over the same gt box)
    
    def on_train_epoch_start(self):
        self.train_visualizer.on_epoch_start_setup(self.logger, self.trainer.num_training_batches)
    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        loss, pred_heatmaps, image_embeddings, label_heatmaps = self.model_step(batch)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        
        # Log images
        if self.train_visualizer.visualize_batch_flag(batch_idx):
            self._log_heatmaps(images[0], pred_heatmaps[0], label_heatmaps[0], prefix='train')
        
        return loss
    
    def on_validation_epoch_start(self):
        self.val_losses = []
        self.val_visualizer.on_epoch_start_setup(self.logger, self.trainer.num_val_batches[0])
        
        self.mAP_perfect_cls_head.reset()
    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        gt_bboxes = batch['bboxes']
        gt_center_points = batch['center_points']
        image_paths = batch['image_paths']
        loss, pred_heatmaps, image_embeddings, label_heatmaps = self.model_step(batch)
        
        all_peaks_np = []
        all_perfect_cls_head_pred_bboxes = []
        all_perfect_cls_head_idxes = []
        all_segmentations_np = []
        for img_idx in range(len(gt_bboxes)):
            # heatmap box decoder head should always be in eval mode and no_grad
            # Extract all boxes
            img_peaks_np, img_bbox_xyxy_np, img_segmentations_np, img_predicted_ious_np, img_init_point_h_score = self.heatmap_box_detector(
                image_embeddings[img_idx].unsqueeze(0),
                pred_heatmaps[img_idx].squeeze().cpu().numpy().astype('float32'),
                gt_points = gt_center_points[img_idx] if self.val_oracle_point_decoder else None, # used only for oracle point decoder
                )
            
            if self.save_boxes:
                np.save(self.save_root/f"{Path(image_paths[img_idx]).stem}.npy", img_bbox_xyxy_np)
            
            # Filter to keep perfect boxes
            perfect_cls_head_pred_bboxes, perfect_cls_head_idxes = self.perfect_cls_head_boxes(torch.tensor(img_bbox_xyxy_np, device='cuda'),
                                                                    gt_bboxes[img_idx],
                                                                    iou_threshold=0.5)
            
            all_peaks_np.append(img_peaks_np)
            all_perfect_cls_head_pred_bboxes.append(perfect_cls_head_pred_bboxes)
            all_perfect_cls_head_idxes.append(perfect_cls_head_idxes)
            all_segmentations_np.append(img_segmentations_np)
            
            # Update mAP metrics
            predictions_perfect_cls_head = [{
                'boxes': perfect_cls_head_pred_bboxes,
                'scores': torch.ones(perfect_cls_head_pred_bboxes.shape[0], device='cuda'), #all_boxes perfect score
                'labels': torch.ones(perfect_cls_head_pred_bboxes.shape[0], dtype=torch.int32, device='cuda'),  # Assuming single class
                }]
                
            targets = [{
                    'boxes': gt_bboxes[img_idx],
                    'labels': torch.ones(len(gt_bboxes[img_idx]), dtype=torch.int32, device='cuda')
                    }]
            
            self.mAP_perfect_cls_head.update(predictions_perfect_cls_head, targets)
        
        self.val_losses.append(loss.item())
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        
        # Log visualizations (only the first image in the batch)
        if self.val_visualizer.visualize_batch_flag(batch_idx):
            self._log_heatmaps(images[0], pred_heatmaps[0], label_heatmaps[0], prefix='val', batch_idx=batch_idx)
            
            self._log_complete_img(image=images[0],
                                pred_heatmap=pred_heatmaps[0,0].cpu().numpy(),
                                segmentations=all_segmentations_np[0][all_perfect_cls_head_idxes[0].cpu().tolist()],
                                peaks_np=all_peaks_np[0],
                                pred_boxes=all_perfect_cls_head_pred_bboxes[0].cpu().numpy(),
                                gt_boxes=gt_bboxes[0].cpu().numpy(),
                                pred_boxes_logits=None,
                                prefix='val',
                                batch_idx=batch_idx)
        
        return loss
    
    def on_validation_epoch_end(self):
        epoch_val_loss = np.mean(self.val_losses)
        self.log('val/custom_loss', epoch_val_loss, on_epoch=True, prog_bar=False)
        
        map_perfect_cls_head_metrics = self.mAP_perfect_cls_head.compute()
        self.log('val_mAP_perfect_cls_head_metrics', map_perfect_cls_head_metrics['map'], on_epoch=True, prog_bar=True)
        
        for key, value in map_perfect_cls_head_metrics.items():
            self.log(f'val/{key}_perfect_cls_head_metrics', value.item() if isinstance(value, torch.Tensor) else value)
    
    def configure_optimizers(self):
        # Filter out the parameters of the 'sam' module
        params_to_optimize = [
            param for name, param in self.named_parameters() if not name.startswith("sam.")
        ]
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.scheduler_patience,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": self.check_val_every_n_epoch, #! should be a multiple of check_val_every_n_epoch
            }
        }
    
    def _log_heatmaps(self, image, pred_heatmap, gt_heatmap, prefix, batch_idx=None):
        image = np.transpose(np.clip(unnormalize_image(image.detach().cpu()), 0, 1), (1, 2, 0))
        
        # Rescale heatmap
        gt_heatmap_np = gt_heatmap[0].detach().cpu().numpy() # 1024x1024
        
        pred_heatmap_np = pred_heatmap[0].detach().cpu().numpy() # 1024x1024
        
        max_chm = max(1, gt_heatmap_np.max(), pred_heatmap_np.max())
        
        fig, axes  = plt.subplots(1, 3, figsize=(15, 6))
        axes[0].imshow(image)
        axes[1].imshow(gt_heatmap_np, cmap='viridis', vmin=0, vmax=max_chm)
        axes[2].imshow(pred_heatmap_np, cmap='viridis', vmin=0, vmax=max_chm)
        
        for ax, title in zip(axes, ['Image', 'GT', 'Pred']):
            ax.set_title(title, fontsize=15)
            ax.axis('off')
        fig.tight_layout()
        
        batch_idx_str = f'_{batch_idx}' if batch_idx is not None else ''
        
        # Log to tensorboard
        self.logger.experiment.add_figure(f'{prefix}/heatmap{batch_idx_str}', fig, self.global_step)
    
    def _log_complete_img(self, image: torch.Tensor, pred_heatmap:np.ndarray, segmentations:np.ndarray, peaks_np:np.ndarray,
                        pred_boxes:np.ndarray, gt_boxes:np.ndarray, pred_boxes_logits:torch.Tensor,
                        prefix: str, batch_idx=None):
        
        image_np = np.transpose(np.clip(unnormalize_image(image.detach().cpu()), 0, 1), (1, 2, 0))
        if pred_boxes_logits is not None:
            pred_boxes_scores = torch.sigmoid(pred_boxes_logits).detach().cpu().numpy()
        else:
            pred_boxes_scores = None
        
        pred_heatmap = pred_heatmap.astype(np.float32)
        fig = viz_img_heat_seg_box(image_np, pred_heatmap, peaks_np, segmentations,
                                pred_boxes, gt_boxes, pred_boxes_scores=pred_boxes_scores)
        batch_idx_str = f'_{batch_idx}' if batch_idx is not None else ''
        # Log to tensorboard
        self.logger.experiment.add_figure(f'{prefix}/complete{batch_idx_str}', fig, self.global_step)