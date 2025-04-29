import logging
logging.getLogger('rasterio._env').setLevel(logging.ERROR)
import torch
torch.set_float32_matmul_precision('medium')
import hydra
import pytorch_lightning as pl

from pathlib import Path
from omegaconf import DictConfig
from rich import print as rprint
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from src.pseco.models.components.segment_anything.build_sam import build_sam_vit_h
from src.pseco.data import PointDecoderFinetuneDataModule
from src.utils import RamGpuUsageProgressBar, PartialModelCheckpoint
from src.pseco.models import PointDecoderFinetuneLitModule

@hydra.main(config_path="configs", config_name="train_point_decoder_supervised_finetune", version_base=None)
def main(cfg: DictConfig):
    
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Print config
    rprint("[bold blue]Configuration:[/bold blue]")
    rprint(OmegaConf.to_container(cfg, resolve=True))
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Initialize SAM model
    assert cfg.model.point_decoder.sam_type == 'vit_h', "Invalid SAM model"
    sam_vith_path = Path(original_cwd)/'pretrained/SAM/sam_vit_h_4b8939.pth'
    sam = build_sam_vit_h(checkpoint=sam_vith_path)
    
    # Initialize model
    model = PointDecoderFinetuneLitModule(cfg, sam_model=sam)
    
    # Initialize data module
    datamodule = PointDecoderFinetuneDataModule(
        train_img_dir=cfg.data.train_img_dir,
        train_ann_file=cfg.data.train_ann_file,
        val_img_dir=cfg.data.val_img_dir,
        val_ann_file=cfg.data.val_ann_file,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        augment=cfg.data.augment,
    )
    
    # Configure callbacks
    callbacks = [
        RamGpuUsageProgressBar(),
        PartialModelCheckpoint(
            key_to_save_prefix=cfg.training.checkpoint_weight_prefix,
            key_to_exclude_prefixes=cfg.training.checkpoint_exclude_prefixes,
            rmv_root=True,
            dirpath=output_dir,
            filename='pt_dec_finetune_{epoch:02d}-{val_mAP_perfect_cls_head_metrics:.5f}',
            save_top_k=3,
            monitor='val_mAP_perfect_cls_head_metrics',
            mode='max'
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.early_stopping_patience,
            mode='min'
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
    
    if hasattr(cfg.model.point_decoder, 'point_decoder_state_dict_path'):
        trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()