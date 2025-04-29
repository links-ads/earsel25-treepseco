import torch
import torchvision
from torch import nn, Tensor
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Union
import warnings

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

class CustomRcnn(nn.Module):
    """
    Adapted R-CNN model using external proposals, accepting a batched image tensor,
    assuming fixed input size (1024x1024) and no postprocessing.

    Assumptions:
    1. Input `images` is a batched tensor [B, C, 1024, 1024] and already normalized.
    2. Input `proposals` (List[Tensor]) and `targets` (List[Dict]) coordinates are
        relative to the 1024x1024 space. List lengths must match batch size B.
    3. No RPN is used internally.
    4. No postprocessing (coordinate scaling) is performed. Output coordinates are
        relative to the 1024x1024 input space.

    The user is responsible for:
    1. Ensuring the input image batch has the correct shape [B, C, 1024, 1024] and is normalized.
    2. Providing proposals and targets lists matching the batch size B, scaled correctly.
    """
    def __init__(self, num_classes: int, use_df_rn50_weights: bool = True):
        super().__init__()
        
        # 1. Backbone Setup
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights).backbone
        
        if use_df_rn50_weights:
            neon_state_dict = torch.load("./pretrained/rn50/NEON.pt", map_location="cpu", weights_only=True)
            filtered_neon_state_dict = {}
            for k, v in neon_state_dict.items():
                # Filter out keys that are not in the default backbone
                if k.startswith("backbone.") and "fpn." not in k:
                    filtered_neon_state_dict[k[len("backbone."):]] = v
            self.backbone.load_state_dict(filtered_neon_state_dict, strict=False)
            print("Using default ResNet50 FPN backbone with NEON weights.")
        
        # 2. RoI Heads Setup
        out_channels = self.backbone.out_channels
        featmap_names = ['0', '1', '2', '3', 'pool']
        box_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes)
        # RoI Heads hyperparameters (using defaults)
        box_score_thresh=0.05
        box_nms_thresh=0.5
        box_detections_per_img=100
        box_fg_iou_thresh=0.5
        box_bg_iou_thresh=0.5
        box_batch_size_per_image=512
        box_positive_fraction=0.25
        bbox_reg_weights=None
        self.roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img
        )
    
    def forward(self,
                images: Tensor, # Batched tensor [B, C, 1024, 1024]
                proposals: List[Tensor], # List [B] of tensors [N, 4], coords relative to 1024x1024
                targets: Optional[List[Dict[str, Tensor]]] = None # List [B] of dicts, targets relative to 1024x1024
                ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (Tensor): Batch of input images [B, C, 1024, 1024], normalized.
            proposals (List[Tensor]): List (length B) of pre-computed proposals per image,
                                    in [x1, y1, x2, y2] format, relative to 1024x1024.
            targets (Optional[List[Dict[str, Tensor]]]): List (length B) of ground-truth dicts
                                                        for training, relative to 1024x1024.
                                                        Required keys: "boxes", "labels".
        
        Returns:
            Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
                - In training mode: Dictionary of losses.
                - In eval mode: List (length B) of detection dictionaries, with coordinates
                                relative to the 1024x1024 input space.
        """
        # --- Input Validation ---
        if images.dim() != 4:
            raise ValueError(f"Expected images to be a 4D tensor [B, C, H, W], got {images.dim()}D")
        
        B, C, H, W = images.shape
        if (H, W) != (1024, 1024):
            raise ValueError(f"Input images must be 1024x1024, got {H}x{W}")
        
        if len(proposals) != B:
            raise ValueError(f"Number of proposal lists ({len(proposals)}) does not match batch size ({B})")
        
        # --- Target Validation (if training) ---
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None in training mode")
            if len(targets) != B:
                raise ValueError(f"Number of target lists ({len(targets)}) does not match batch size ({B})")
            
            # Validate content of each target dictionary in the list
            for target_idx, target in enumerate(targets): # target_idx corresponds to batch index
                boxes = target.get("boxes")
                if boxes is None:
                    raise ValueError(f"Target dict at index {target_idx} missing 'boxes' key.")
                if not isinstance(boxes, torch.Tensor):
                    raise ValueError(f"Target boxes at index {target_idx} must be Tensor, got {type(boxes)}.")
                if boxes.dim() != 2 or boxes.shape[-1] != 4:
                    raise ValueError(f"Expected target boxes [N, 4] at index {target_idx}, got {boxes.shape}.")
                
                # Check for degenerate boxes within this target
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(f"Invalid box {degen_bb} found in target at batch index {target_idx}.")
                
                # Check labels exist
                if "labels" not in target:
                    raise ValueError(f"Target dict at index {target_idx} missing 'labels' key.")
        
        # --- Image Size List for RoIHeads ---
        # RoIHeads expects a list of tuples [(H, W), (H, W), ...] for the batch
        image_sizes_for_roi: List[Tuple[int, int]] = [(H, W)] * B
        
        # --- Backbone Feature Extraction ---
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            warnings.warn("Backbone returned a single Tensor. Wrapping in OrderedDict with key '0'. Ensure RoI Pooler featmap_names are compatible.")
            features = OrderedDict([("0", features)])
        elif not isinstance(features, OrderedDict):
            raise TypeError(f"Backbone output expected to be OrderedDict or Tensor, got {type(features)}")
        
        # --- RoI Heads ---
        # Pass the constructed list of image sizes
        for j in range(len(proposals)):
            if proposals[j].shape[0] == 0:
                proposals[j]=torch.zeros((1,4), device='cuda', dtype=torch.float32)
        
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes_for_roi, targets)
        
        # --- Output Formatting ---
        losses = {}
        losses.update(detector_losses)
        # Return losses in training, raw detections in eval
        if self.training:
            return losses
        return detections
    