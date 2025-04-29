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

from src.pseco.models.components.segment_anything.build_sam import build_sam_vit_h
from src.pseco.data import PointDecoderFinetuneDataModule
from src.pseco.models import PointDecoderFinetuneLitModule

@hydra.main(config_path="configs", config_name="gen_boxes_from_pd", version_base=None)
def main(cfg: DictConfig):
    
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Print config
    rprint("[bold blue]Configuration:[/bold blue]")
    rprint(OmegaConf.to_container(cfg, resolve=True))
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Initialize data module
    datamodule = PointDecoderFinetuneDataModule(
        train_img_dir=cfg.data.train_img_dir,
        train_ann_file=cfg.data.train_ann_file,
        val_img_dir=cfg.data.val_img_dir,
        val_ann_file=cfg.data.val_ann_file,
        batch_size=1,
        num_workers=cfg.training.num_workers,
        augment=cfg.data.augment,
    )
    
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
        logger=logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch = cfg.training.check_val_every_n_epoch,
    )
    datamodule.setup(stage='fit')
    
    # Initialize SAM model
    assert cfg.model.point_decoder.sam_type == 'vit_h', "Invalid SAM model"
    sam_vith_path = Path(original_cwd)/'pretrained/SAM/sam_vit_h_4b8939.pth'
    sam = build_sam_vit_h(checkpoint=sam_vith_path)
    
    # Initialize model
    eval_save_root = f"{output_dir}/eval"
    Path(eval_save_root).mkdir(parents=True, exist_ok=True)
    
    model = PointDecoderFinetuneLitModule(cfg, sam_model=sam, save_boxes=True, save_root=eval_save_root)
    trainer.validate(model, dataloaders=datamodule.val_dataloader())
    
    model.save_root = Path(f"{output_dir}/train") 
    model.save_root.mkdir(parents=True, exist_ok=True)
    trainer.validate(model, dataloaders=datamodule.train_dataloader())

if __name__ == '__main__':
    main()