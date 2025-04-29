import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
from typing import Tuple, List, Dict, Union 
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path

def get_transform_with_proposals(augment):
    bbox_params = A.BboxParams(format='pascal_voc',label_fields=["dummy_labels"])
    
    if augment:
        transform = A.Compose(
            [
            A.PadIfNeeded(600, 600, border_mode=0, position="center", fill = 0),
            A.LongestMaxSize(1024),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.8
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.8),
            ], p=0.5),
            
            A.OneOf([
                A.GaussianBlur(p=0.6),
                # A.RandomFog(p=0.4),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(p=0.4),
                A.ISONoise(p=0.4),
            ], p=0.3),
            
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            ], p=0.3),
            
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            ], p=0.2),
            
            ],
            bbox_params=bbox_params
        )
    else:
        transform = A.Compose([
            A.PadIfNeeded(600, 600, border_mode=0, position="center", fill = 0),
            A.LongestMaxSize(1024)
            ], bbox_params=bbox_params)
    
    return transform

class CachedProposalsDataset(Dataset):
    def __init__(self, image_dir, annotation_file, proposals_path_root, partition, augment=False):
        self.image_dir = image_dir
        self.proposals_path_root = proposals_path_root
        self.partition = partition
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
        
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in self.coco_data['images']
        }
        
        annotated_image_ids = set(self.image_to_anns.keys())
        self.image_ids = [img_id for img_id in self.image_id_to_filename.keys() if img_id in annotated_image_ids]
        
        self.spatial_transform = get_transform_with_proposals(augment)
        
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def convert_xyxy_from_coco(self, bbox):
        # Converts COCO [x,y,width,height] to [x_min, y_min, x_max, y_max]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        return [x1, y1, x2, y2]
    
    def __len__(self):
        return len(self.image_ids)
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Union[torch.Tensor, List, int, str]]]
                   ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[int], List[str]]]:
        """Collates batch data, handling variable numbers of boxes and proposals."""
        images = []
        bboxes = []
        proposals = []
        num_boxes = []
        image_paths = []

        for sample in batch:
            images.append(sample['image'])
            bboxes.append(sample['bboxes'])
            proposals.append(sample['proposals'])
            num_boxes.append(sample['num_boxes'])
            image_paths.append(sample['image_path'])

        # Stack images as they have the same dimensions after transforms
        images = torch.stack(images, dim=0)

        # Return a dictionary - Bboxes, center_points, and proposals remain lists of tensors
        # because their counts vary per image.
        out = {
            'images': images,         # Tensor [B, C, H, W]
            'bboxes': bboxes,         # List[Tensor[N_gt, 4]]
            'proposals': proposals,   # List[Tensor[N_prop, 4]]
            'num_boxes': num_boxes,   # List[int]
            'image_paths': image_paths  # List[str]
        }
        return out
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        image_stem = Path(image_filename).stem
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        original_height, original_width = image.shape[:2]
        
        
        # --- Load Proposals ---
        proposal_file = f"{image_stem}.npy"
        proposal_filepath = os.path.join(self.proposals_path_root, proposal_file)
        
        np_proposals = np.load(proposal_filepath).astype(np.float32)
        if self.partition == 'train':
            assert original_height==original_width and (original_height==400 or original_height==600), \
                f"Image {image_path} must be square and either 400x400 or 600x600. Found: {original_height}x{original_width} in {image_path}" 
            np_proposals = np_proposals * 600/1024
            
        elif self.partition == 'val':
            np_proposals = np_proposals * 600/1024
            np_proposals[:, [0, 2]] = np.clip(np_proposals[:, [0, 2]], (600 - original_width) / 2, 600 - ((600 - original_width) / 2))
            np_proposals[:, [1, 3]] = np.clip(np_proposals[:, [1, 3]], (600 - original_height) / 2, 600 - ((600 - original_height) / 2))
            np_proposals[:, [0, 2]] = np_proposals[:, [0, 2]] - ((600 - original_width) / 2)
            np_proposals[:, [1, 3]] = np_proposals[:, [1, 3]] - ((600 - original_height) / 2)
        
        else:
            raise ValueError(f"Invalid partition {self.partition}. Expected 'train' or 'val'.")
        
        keep_proposals = np_proposals[:,0]<np_proposals[:,2]
        keep_proposals &= np_proposals[:,1]<np_proposals[:,3]
        np_proposals = np_proposals[keep_proposals] 
        
        proposals = np_proposals.tolist()
        dummy_proposal_labels = [0] * len(np_proposals)
        
        # --- Load Ground Truth Annotations ---
        annotations = self.image_to_anns.get(image_id, [])
        gt_bboxes_xyxy = [self.convert_xyxy_from_coco(ann['bbox']) for ann in annotations]
        
        num_gt_boxes = len(gt_bboxes_xyxy)
        dummy_labels = [0] * num_gt_boxes
        
        gt_boxes_and_proposals = gt_bboxes_xyxy + proposals # Combine GT and proposals
        gt_boxes_and_proposals_dummy_labels = dummy_labels + dummy_proposal_labels
        
        transformed = self.spatial_transform(
            image=image,
            bboxes=gt_boxes_and_proposals,
            dummy_labels=gt_boxes_and_proposals_dummy_labels,
        )
        
        transformed_image = transformed['image']
        transformed_gt_boxes_and_proposals = transformed['bboxes']  # XYXY format
        
        transformed_bboxes = transformed_gt_boxes_and_proposals[:num_gt_boxes]  # Only GT boxes
        transformed_proposals = transformed_gt_boxes_and_proposals[num_gt_boxes:]  # Only proposals
        
        transformed_image = self.normalize_transform(transformed_image)
        
        bboxes_tensor = torch.tensor(transformed_bboxes, dtype=torch.float32)
        proposals_tensor = torch.tensor(transformed_proposals, dtype=torch.float32)
        
        
        return {
            'image': transformed_image,          # Tensor [C, H, W]
            'bboxes': bboxes_tensor,             # Tensor [N_gt, 4] in XYXY format
            'proposals': proposals_tensor,       # Tensor [N_prop, 4] in XYXY format
            'num_boxes': len(transformed_bboxes),# int
            'image_path': image_path             # str
        }
