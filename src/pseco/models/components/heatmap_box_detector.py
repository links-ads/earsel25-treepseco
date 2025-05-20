import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torchvision.ops.boxes import batched_nms

from .segment_anything.modeling.sam import Sam
from .segment_anything.utils.amg import MaskData, calculate_stability_score, batched_mask_to_box, mask_to_rle_pytorch, rle_to_mask
from src.utils import configurable
from src.pycrown.utils import heatmap2points_pycrown


def peaks2boxes(points: torch.Tensor, box_half_sides: torch.Tensor):
    """point (n,2), box_half_sides (num_boxes)"""
    points = points.repeat(1, 2) # (num_points, 4)
    points = points.view(-1, 1, 4) # (num_points, 1, 4)
    
    box_half_sides = box_half_sides.view(-1,1).repeat(1,4) # (num_boxes, 4)
    signs = torch.tensor([-1, -1, 1, 1])
    zeros_centerd_boxes = box_half_sides * signs # (num_boxes, 4)
    res = points + zeros_centerd_boxes # (num_points, num_boxes, 4)
    return res.view(-1, 4) # (num_points * num_boxes, 4)

def boxes_to_corners(boxes): # (num_boxes, 4)
    # Extract coordinates
    x1, y1, x2, y2 = boxes.unbind(1)
    
    # Stack all coordinates in a flat array
    all_coords = torch.stack([
        x1, y1,  # Top-left
        x2, y1,  # Top-right
        x1, y2,  # Bottom-left
        x2, y2   # Bottom-right
    ], dim=1)  # Shape: (num_boxes, 8)
    
    # Reshape to (num_boxes, 4, 2)
    corners = all_coords.view(boxes.shape[0], 4, 2)
    return corners # (num_boxes, 4, 2)

def h_score_from_float_2d_idx(float_2d_idx:np.ndarray, heatmap:np.ndarray):
    float_2d_idx = float_2d_idx * (256/1024)
    x = float_2d_idx[0]
    y = float_2d_idx[1]
    
    x_floor = int(np.floor(x))
    x_ceil = int(min(255, np.ceil(x)))
    y_floor = int(np.floor(y))
    y_ceil = int(min(255, np.ceil(y)))
    
    h_score = heatmap[[x_floor, x_floor, x_ceil, x_ceil],[y_floor, y_ceil, y_floor, y_ceil]].mean()
    
    return h_score

class HeatmapBoxDetector(nn.Module):
    @configurable
    def __init__(self,
                sam: Sam,
                ws_smooth: int,
                ws_det: int,
                himn: float,
                pred_iou_thresh: float,
                stability_score_thresh: float,
                box_nms_thresh: float,
                points_only_multimask_output: bool,
                box_half_sides: List[int],
                neg_points_at_corners: bool,
                prompt_batch_size: int,
                do_box_filtering: bool,
                min_area: int = None,
                max_area: int = None,
                max_aspect_ratio: float = None,
                use_points: bool = True,
                use_boxes: bool = True,
                ):
        
        super().__init__()
        self.sam = sam
        sam.eval()
        for param in sam.parameters():
            param.requires_grad = False
        
        self.ws_smooth = ws_smooth
        self.ws_det = ws_det
        self.himn = himn
        
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.stability_score_offset = 1.0 #! hardcoded
        
        self.points_only_multimask_output = points_only_multimask_output
        self.box_half_sides = box_half_sides
        
        self.neg_points_at_corners = neg_points_at_corners
        
        assert use_points or use_boxes, "At least one of use_points or use_boxes must be True"
        self.use_points = use_points
        self.use_boxes = use_boxes
        
        self.prompt_batch_size = prompt_batch_size
        
        self.do_box_filtering = do_box_filtering
        if self.do_box_filtering:
            assert min_area is not None and max_area is not None and max_aspect_ratio is not None, "min_area, max_area and max_aspect_ratio must be provided"
        
        self.min_area = min_area
        self.max_area = max_area
        self.max_aspect_ratio = max_aspect_ratio
    
    @classmethod
    def from_config(cls, cfg):
        return {"ws_smooth": cfg.model.heatmap_box_detector.ws_smooth,
                "ws_det": cfg.model.heatmap_box_detector.ws_det,
                "himn": cfg.model.heatmap_box_detector.himn,
                "pred_iou_thresh": cfg.model.heatmap_box_detector.pred_iou_thresh,
                "stability_score_thresh": cfg.model.heatmap_box_detector.stability_score_thresh,
                "box_nms_thresh": cfg.model.heatmap_box_detector.box_nms_thresh,
                "points_only_multimask_output": cfg.model.heatmap_box_detector.points_only_multimask_output,
                "box_half_sides": cfg.model.heatmap_box_detector.box_half_sides,
                "neg_points_at_corners": cfg.model.heatmap_box_detector.neg_points_at_corners,
                "prompt_batch_size": cfg.model.heatmap_box_detector.prompt_batch_size,
                "do_box_filtering": cfg.model.heatmap_box_detector.do_box_filtering,
                "min_area": cfg.model.heatmap_box_detector.min_area,
                "max_area": cfg.model.heatmap_box_detector.max_area,
                "max_aspect_ratio": cfg.model.heatmap_box_detector.max_aspect_ratio,
                }
    
    def get_keep_filter_boxes(self, boxes_xyxy: torch.Tensor):
        """boxes_xyxy: (num_boxes, 4)"""
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        area = h * w
        aspect_ratio = torch.max(h / (w + 1e-6), w / (h + 1e-6))
        
        keep_by_area = (area >= self.min_area) & (area <= self.max_area)
        keep_by_aspect_ratio = aspect_ratio <= self.max_aspect_ratio
        keep_by_filter = keep_by_area & keep_by_aspect_ratio
        return keep_by_filter

    @torch.no_grad()
    def predict_torch(
        self,
        img_embedding: torch.Tensor,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        output_four_masks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the img_embedding.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.
        
        Note:
            B -> the number of resulting masks
            N -> the number of prompts to create a single mask
        Arguments:
            img_embedding (torch.Tensor): The image embedding of a single image -> (1, c, h, w)
            point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels.
            point_labels (torch.Tensor or None): A BxN array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a
                background point.
            boxes (torch.Tensor or None): A Bx4 array given a box prompt to the
                model, in XYXY format.
            mask_input (torch.Tensor): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form Bx1xHxW, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            return_logits (bool): If true, returns un-thresholded masks logits
                instead of a binary mask.
            
            Returns:
            (torch.Tensor): The output masks in BxCxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
            (torch.Tensor): An array of shape BxC containing the model's
                predictions for the quality of each mask.
            (torch.Tensor): An array of shape BxCxHxW, where C is the number
                of masks and H=W=256. These low res logits can be passed to
                a subsequent iteration as mask input.
        """
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        
        # Predict masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=img_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            output_four_masks=output_four_masks,
        )
        
        # Upscale the masks to the original image resolution
        input_size = (1024, 1024) #! model input size hardcoded (vith)
        original_size = (1024, 1024) #! original image size hardcoded
        masks = self.sam.postprocess_masks(low_res_masks, input_size, original_size)
        
        if not return_logits:
            masks = masks > self.sam.mask_threshold
        
        return masks, iou_predictions, low_res_masks
    
    @torch.no_grad()
    def prompt_points_only(self, img_embedding: torch.Tensor, points: torch.Tensor, proposals_only:bool=False):
        """Predict masks for the given prompt points, using the img_embedding.
        
        Note:
            B -> the number of resulting masks
            N -> the number of prompts to create a single mask
        
        Parameters
        ----------
        img_embedding : torch.Tensor
            The image embedding of a single image -> (1, c, h, w)
        points : torch.Tensor
            A Bx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
        """
        data = MaskData()
        for batch_points in torch.split(points, self.prompt_batch_size, dim=0):
            in_points = batch_points.unsqueeze(1).cuda()
            in_labels = torch.ones_like(in_points[:, :, 0], device='cuda')
            masks, iou_preds, _ = self.predict_torch(
                img_embedding,
                in_points,
                in_labels,
                multimask_output=self.points_only_multimask_output,
                return_logits=True,
                output_four_masks=True,
            ) 
            
            # Serialize predictions and store in MaskData
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                init_point=torch.repeat_interleave(in_points.squeeze(1), masks.shape[1], dim=0)
            )
            del masks
            
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = batch_data["iou_preds"] > self.pred_iou_thresh
                batch_data.filter(keep_mask)
            
            # Calculate stability score
            if self.stability_score_thresh > 0.0:
                batch_data["stability_score"] = calculate_stability_score(
                    batch_data["masks"], self.sam.mask_threshold, self.stability_score_offset
                )
                keep_mask = batch_data["stability_score"] >= self.stability_score_thresh
                batch_data.filter(keep_mask)
                del batch_data["stability_score"]
            
            # Threshold masks and calculate boxes
            batch_data["masks"] = batch_data["masks"] > self.sam.mask_threshold
            batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])
            if not proposals_only:
                batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"]) # to occupy less space
            del batch_data["masks"]
            
            # accumulate
            data.cat(batch_data)
            del batch_data
        
        # Filter boxes by area and aspect ratio
        if self.do_box_filtering:
            boxes_xyxy = data["boxes"]
            keep_by_filter = self.get_keep_filter_boxes(boxes_xyxy)
            data.filter(keep_by_filter)
        
        # Remove duplicates among point_only results
        if self.box_nms_thresh < 1.0:
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]), # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)
        
        return data
    
    @torch.no_grad()
    def prompt_boxes_only(self, img_embedding: torch.Tensor, points: torch.Tensor, proposals_only:bool=False):
        """Predict masks for the given img_embedding, using boxes crafted around given points
        (and optionally negative points at boxes corners).
        
        Note:
            B -> the number of resulting masks
            N -> the number of prompts to create a single mask
        
        Parameters
        ----------
        img_embedding : torch.Tensor
            The image embedding of a single image -> (1, c, h, w)
        points : torch.Tensor
            A Bx2 array of point that will be at the center of the generated boxes. Each point is in (X,Y) in pixels.
        """
        prompt_boxes = peaks2boxes(points, torch.tensor(self.box_half_sides).float()) # (num_total_boxes, 4)
        
        data = MaskData()
        for batch_prompt_boxes in torch.split(prompt_boxes, self.prompt_batch_size, dim=0):
            if self.neg_points_at_corners:
                points_coords = boxes_to_corners(batch_prompt_boxes).cuda() # (num_boxes, 4, 2)
                points_labels = torch.zeros_like(points_coords[:, :, 0]).cuda() # (num_boxes, 4)
            else:
                points_coords = None
                points_labels = None
            
            batch_prompt_boxes = batch_prompt_boxes.cuda()
            masks, iou_preds, _ = self.predict_torch(
                img_embedding=img_embedding,
                point_coords=points_coords,
                point_labels=points_labels,
                boxes=batch_prompt_boxes,
                multimask_output=False,
                return_logits=False,
                )
            
            # Serialize predictions and store in MaskData
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                init_point=(batch_prompt_boxes[:,(0,1)] + batch_prompt_boxes[:,(2,3)]) / 2
            )
            del masks
            
            batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])
            if not proposals_only:
                batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"]) # to occupy less space
            del batch_data["masks"]
            
            # accumulate
            data.cat(batch_data)
            del batch_data
        
        # Filter boxes by area and aspect ratio
        if self.do_box_filtering:
            boxes_xyxy = data["boxes"]
            keep_by_filter = self.get_keep_filter_boxes(boxes_xyxy)
            data.filter(keep_by_filter)
        
        # Remove duplicates among boxes_only results
        if self.box_nms_thresh < 1.0:
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)
        
        return data
    
    @torch.no_grad()
    def forward(self, img_embedding: torch.Tensor, heatmap: np.ndarray, gt_points:torch.Tensor = None,
                mode:str='eval', proposals_only:bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """From the heatmap of a single image to masks and bboxes 
        
        Parameters
        ----------
        img_embedding : torch.Tensor
            The image embedding of a single image -> (1, c, h, w)
        
        heatmap : np.ndarray
            The heatmap of a single image -> (h, w) 
        """
        if gt_points is None:
            peaks_np = heatmap2points_pycrown(heatmap, ws_smooth=self.ws_smooth, ws_detection=self.ws_det, hmin=self.himn) # (N, 2)
            # from coordinates in 256x256 to 1024x1024
            peaks_np = peaks_np * (1024/256)
            if mode=='train': #discard some peaks since the cls_head will only use a subset of them
                random_indices = np.random.choice(peaks_np.shape[0], size=min(peaks_np.shape[0],300), replace=False)
                peaks_np = peaks_np[random_indices]
            peaks_tensor = torch.tensor(peaks_np) # (N, 2)
        else:
            peaks_tensor = gt_points.cpu() # (N, 2)
            peaks_np = peaks_tensor.numpy()
        
        if peaks_tensor.shape[0] == 0:
            if proposals_only:
                return np.empty((0, 4), dtype=np.float32)
            else:
                return peaks_np, np.empty((0, 4)), np.empty((0, 1024, 1024)), np.empty((0)), np.empty((0))
        
        img_embedding = img_embedding.cuda()
        
        mask_data = MaskData()
        if self.use_points:
            # get 1 or 3 masks for each peak (maybe less after filtering)
            mask_data_1 = self.prompt_points_only(img_embedding, peaks_tensor, proposals_only)
            mask_data.cat(mask_data_1)
        
        if self.use_boxes:
            # get len(box_half_sides) boxes for each peak (maybe less after filtering)
            mask_data_2 = self.prompt_boxes_only(img_embedding, peaks_tensor, proposals_only)
            mask_data.cat(mask_data_2)
        
        # nms between the two modalities
        if self.box_nms_thresh < 1.0:
            keep_by_nms = batched_nms(
                mask_data["boxes"].float(),
                mask_data["iou_preds"],
                torch.zeros_like(mask_data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            mask_data.filter(keep_by_nms)
        mask_data.to_numpy()
        
        bbox_xyxy = []
        segmentations = [] 
        predicted_ious = []
        h_scores = []
        
        if proposals_only:
            return np.stack(list(mask_data["boxes"]))
        
        else:
            for idx in range(len(mask_data["rles"])):
                bbox_xyxy.append(mask_data["boxes"][idx])
                segmentations.append(rle_to_mask(mask_data["rles"][idx]))
                predicted_ious.append(mask_data["iou_preds"][idx].item())
                if heatmap is not None:
                    h_scores.append(h_score_from_float_2d_idx(mask_data["init_point"][idx], heatmap))
            
            return peaks_np, np.stack(bbox_xyxy), np.stack(segmentations), np.array(predicted_ious), np.array(h_scores)
    