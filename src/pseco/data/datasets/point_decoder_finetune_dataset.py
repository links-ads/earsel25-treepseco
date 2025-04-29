import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
from typing import Tuple, List
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def get_transform(augment, pad_only:bool=False):
    """Albumentations transformation of bounding boxs."""
    if augment:
        transform = A.Compose(
            [A.PadIfNeeded(600, 600, border_mode=0, position="center", fill = 0),
            A.LongestMaxSize(1024),
            # A.PadIfNeeded(1024, 1024, border_mode=0, position="top_left"),
            A.SafeRotate(limit=180, p=1.0, border_mode=cv2.BORDER_REFLECT101),
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
                A.RandomFog(p=0.4),
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
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=["dummy_labels"]))
    
    else:
        if pad_only:
            transform = A.Compose([
                A.PadIfNeeded(1024, 1024, border_mode=0, position="center", fill = 0),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['dummy_labels']))
        else:
            transform = A.Compose([
                            A.PadIfNeeded(600, 600, border_mode=0, position="center", fill = 0),
                            A.LongestMaxSize(1024),
                        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['dummy_labels']))
    
    return transform

class PointDecoderFinetuneDataset(Dataset):
    """Purposes:
        1. finetune the point decoder generating the heatmap from the bounding box center points.
        2. train the cls_head
    """
    def __init__(self, image_dir, annotation_file, augment=False, pad_only=False):
        self.image_dir = image_dir
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Create image_id to annotations mapping
        self.image_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
            
        # Create image_id to file_name mapping
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in self.coco_data['images']
        }
        
        # List of all image IDs that have annotations
        self.image_ids = list(self.image_to_anns.keys())
        
        # self.image_ids = self.image_ids[:30] #! for debugging
        
        # Albumentation transforms for spatial augmentation
        self.spatial_transform = get_transform(augment, pad_only)
        
        # Normalization and tensor conversion
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def center_point_from_bbox(self, bbox):
        # For XYXY format
        xc = (bbox[0] + bbox[2]) / 2
        yc = (bbox[1] + bbox[3]) / 2
        return xc, yc
    
    def convert_xyxy_from_coco(self, bbox):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        return [x1, y1, x2, y2]
    
    def __len__(self):
        return len(self.image_ids)
    
    @classmethod
    def collate_fn(cls, batch
        ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[int], List[str]]:
        images = []
        bboxes = []
        center_points = []
        num_boxes = []
        image_paths = []
        
        for sample in batch:
            images.append(sample['image'])
            bboxes.append(sample['bboxes'])
            center_points.append(sample['center_points'])
            num_boxes.append(sample['num_boxes'])
            image_paths.append(sample['image_path'])
        
        images = torch.stack(images, dim=0)
        
        out = {
            'images': images,
            'bboxes': bboxes,
            'center_points': center_points,
            'num_boxes': num_boxes,
            'image_paths': image_paths
        }
        return out
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get all annotations for this image
        annotations = self.image_to_anns[image_id]
        
        # Convert all bboxes to XYXY format for albumentation
        bboxes_xyxy = [self.convert_xyxy_from_coco(ann['bbox']) for ann in annotations]
        
        # Dummy labels for albumentation (required by bbox_params)
        dummy_labels = [0] * len(bboxes_xyxy)
        
        # Apply spatial transformations
        transformed = self.spatial_transform(
            image=image, 
            bboxes=bboxes_xyxy,
            dummy_labels=dummy_labels
        )
        
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']  # Already in XYXY format
        
        # Calculate center points from transformed bboxes
        center_points = [self.center_point_from_bbox(bbox) for bbox in transformed_bboxes]
        
        # Apply normalization transforms
        transformed_image = self.normalize_transform(transformed_image)
        
        # Convert to tensors and move to device
        bboxes_tensor = torch.tensor(transformed_bboxes, dtype=torch.float32)
        centers_tensor = torch.tensor(center_points, dtype=torch.float32)
        
        return {
            'image': transformed_image,
            'bboxes': bboxes_tensor,  # Shape: [num_boxes, 4] in XYXY format
            'center_points': centers_tensor,  # Shape: [num_boxes, 2]
            'num_boxes': len(transformed_bboxes),
            'image_path': image_path
        }