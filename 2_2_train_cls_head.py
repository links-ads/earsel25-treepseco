import torch
torch.set_float32_matmul_precision('medium') # or 'high'
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print as rprint
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.utils import RamGpuUsageProgressBar
from src.pseco.models.cls_head_lit_module import ClsHeadLitModule
from src.pseco.models.components.custom_rcnn import CustomRcnn
from src.pseco.data.datamodule import CachedProposalsDataModule

@hydra.main(config_path="configs", config_name="train_cls_head", version_base="1.1")
def main(cfg: DictConfig):
    
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Print config
    rprint("[bold blue]Configuration:[/bold blue]")
    rprint(OmegaConf.to_container(cfg, resolve=True))
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Initialize model
    custom_rcnn = CustomRcnn(num_classes=2) 
    
    # Initialize lit_module
    lit_module = ClsHeadLitModule(cfg, custom_rcnn=custom_rcnn)
    
    # Initialize data module
    datamodule = CachedProposalsDataModule(
        train_img_dir=cfg.data.train_img_dir,
        train_ann_file=cfg.data.train_ann_file,
        train_proposals_path_root=f"{cfg.training.proposals_path_root}/train",
        val_img_dir=cfg.data.val_img_dir,
        val_ann_file=cfg.data.val_ann_file,
        val_proposals_path_root=f"{cfg.training.proposals_path_root}/eval",
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        augment=cfg.data.augment,
    )
    
    # Configure callbacks
    callbacks = [
        RamGpuUsageProgressBar(),
        ModelCheckpoint(
            dirpath=output_dir, 
            filename='{epoch}-{val_mAP:.5f}', 
            save_top_k=2, 
            monitor='val_mAP', 
            mode='max', 
            save_last=True, 
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val_mAP',
            patience=cfg.training.early_stopping_patience,
            mode='max'
        )
    ]
    
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=cfg.training.experiment_name,
        version=cfg.training.version
    )
    
    # Log the Hydra config as hyperparameters to the logger
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True)) 
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch = cfg.training.check_val_every_n_epoch,
    )
    # trainer.validate(lit_module, datamodule=datamodule)
    # Train model
    trainer.fit(lit_module, datamodule=datamodule)

if __name__ == '__main__':
    main()