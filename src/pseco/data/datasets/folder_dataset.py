import os
import torch
import albumentations as A
import numpy as np

from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_pad_and_resize_transforms():
    transform = A.Compose([
                    A.PadIfNeeded(600, 600, border_mode=0, position="center", fill = 0),
                    A.LongestMaxSize(1024),
                ])
    return transform

class FolderDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.all_images = os.listdir(images_dir)
        
        # Albumentation transforms for spatial augmentation
        # TODO: there should be a smart pad and resizing procedure to keep the same training resolution
        self.pad_and_resize_transforms = get_pad_and_resize_transforms()
        
        # Normalization and tensor conversion
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        
        # Load image
        image = Image.open(os.path.join(self.images_dir, self.all_images[idx])).convert('RGB')
        image = np.array(image)
        
        # Apply spatial transformations
        transformed = self.pad_and_resize_transforms(
            image=image 
        )
        
        transformed_image = transformed['image']
        
        # Apply normalization transforms
        transformed_image = self.normalize_transform(transformed_image)
        
        return transformed_image