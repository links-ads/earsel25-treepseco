import numpy as np
import torch
from torch import nn
from typing import List, Tuple

from . import (SAMPointDecoder,
            HeatmapBoxDetector,
            CustomRcnn)

class TreePseco(nn.Module):
    def __init__(self,
                point_decoder: SAMPointDecoder,
                heatmap_box_detector: HeatmapBoxDetector,
                custom_rcnn: CustomRcnn,
                score_th: float,
                ):
        super().__init__()
        
        self.point_decoder = point_decoder.eval()
        self.heatmap_box_detector = heatmap_box_detector.eval()
        self.custom_rcnn = custom_rcnn.eval()
        self.score_th = score_th
    
    @torch.no_grad()
    def forward(self, x
                )->Tuple[List[np.ndarray], List[torch.Tensor], List[np.ndarray], List[np.ndarray], torch.Tensor]:
        
        assert x.shape[0] == 1, "Batch size should be 1"
        
        # point decoder head should always be in eval mode and no_grad
        pred_heatmaps, image_embeddings = self.point_decoder(x)
        pred_heatmap_np = pred_heatmaps.squeeze().to(torch.float32).cpu().numpy()
        
        # heatmap box decoder head should always be in eval mode and no_grad
        proposals_np = self.heatmap_box_detector(
            image_embeddings,
            pred_heatmap_np,
            proposals_only=True,
            )
        
        detections = self.custom_rcnn(images=x, proposals=[torch.tensor(proposals_np, device=x.device, dtype=torch.float32)])[0]
        tree_idxes = detections['labels'] == 1 # useless since bg boxes already removed
        
        tree_pred_scores = detections['scores'][tree_idxes]
        tree_bboxes = detections['boxes'][tree_idxes]
        
        keep = tree_pred_scores > self.score_th
        
        tree_pred_scores_filtered = tree_pred_scores[keep]
        tree_bboxes_filtered = tree_bboxes[keep]
        tree_bboxes_filtered_np = tree_bboxes_filtered.cpu().numpy()
        tree_pred_scores_filtered_np = tree_pred_scores_filtered.cpu().numpy()
        
        if tree_bboxes_filtered.shape[0] == 0:
            return (tree_bboxes_filtered_np, tree_pred_scores_filtered_np, np.empty((0, 1, 1024, 1024), dtype=bool))
        
        masks, iou_predictions, _ = self.heatmap_box_detector.predict_torch(img_embedding=image_embeddings,
                                                point_coords=None,
                                                point_labels=None,
                                                boxes=tree_bboxes_filtered,
                                                multimask_output=False,
                                                return_logits=False,
                                                )
        masks_np = masks.cpu().numpy()
        
        return (tree_bboxes_filtered_np, tree_pred_scores_filtered_np, masks_np)