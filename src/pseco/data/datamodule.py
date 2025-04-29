from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .datasets import PointDecoderFinetuneDataset

from .datasets.cached_proposals_dataset import CachedProposalsDataset

class PointDecoderFinetuneDataModule(LightningDataModule):
    def __init__(
        self,
        train_img_dir: str,
        train_ann_file: str,
        val_img_dir: str,
        val_ann_file: str,
        batch_size: int = 1,
        num_workers: int = 4,
        augment: bool = False,
    ):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_ann_file = train_ann_file
        self.val_img_dir = val_img_dir
        self.val_ann_file = val_ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print(f"Augment in training: {self.augment}")
            self.train_dataset = PointDecoderFinetuneDataset(
                self.train_img_dir,
                self.train_ann_file,
                augment=self.augment,
                pad_only=False,
            )
            
            self.val_dataset = PointDecoderFinetuneDataset(
                self.val_img_dir,
                self.val_ann_file,
                augment=False,
                pad_only=False,
            )
        elif stage == 'validate':
            self.val_dataset = PointDecoderFinetuneDataset(
                self.val_img_dir,
                self.val_ann_file,
                augment=False,
                pad_only=False,
            )
        else:
            raise ValueError(f'Invalid stage: {stage}')
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=PointDecoderFinetuneDataset.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=PointDecoderFinetuneDataset.collate_fn
        )

class CachedProposalsDataModule(LightningDataModule):
    def __init__(
        self,
        train_img_dir: str,
        train_ann_file: str,
        val_img_dir: str,
        val_ann_file: str,
        train_proposals_path_root: str,
        val_proposals_path_root: str,
        batch_size: int = 1,
        num_workers: int = 4,
        augment: bool = False,
    ):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_ann_file = train_ann_file
        self.val_img_dir = val_img_dir
        self.val_ann_file = val_ann_file
        self.train_proposals_path_root = train_proposals_path_root
        self.val_proposals_path_root = val_proposals_path_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print(f"Augment in training: {self.augment}")
            self.train_dataset = CachedProposalsDataset(
                self.train_img_dir,
                self.train_ann_file,
                self.train_proposals_path_root,
                partition='train',
                augment=self.augment,
            )
            self.val_dataset = CachedProposalsDataset(
                self.val_img_dir,
                self.val_ann_file,
                self.val_proposals_path_root,
                partition='val',
                augment=False,
            )
        elif stage == 'validate':
            self.val_dataset = CachedProposalsDataset(
                self.val_img_dir,
                self.val_ann_file,
                self.val_proposals_path_root,
                partition='val',
                augment=False,
            )
        else:
            raise ValueError(f'Invalid stage: {stage}')
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=CachedProposalsDataset.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=CachedProposalsDataset.collate_fn
        )