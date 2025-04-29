import torch
torch.set_float32_matmul_precision('high')
import hydra
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print as rprint
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_system_usage_metrics, unnormalize_image
from src.pseco.data.datasets import FolderDataset
from src.pseco.utils import build_tree_pseco

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=1))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def log_fig(logger, idx, image, tree_bboxes_np, masks_np):
    if image.dim() == 4:
        image = image.squeeze(0)
        
    image_np = np.transpose(np.clip(unnormalize_image(image.detach().cpu()), 0, 1), (1, 2, 0))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.imshow(image_np)
    for box, mask in zip(tree_bboxes_np, masks_np):
        show_box(box, ax)
        show_mask(mask, ax, random_color=True)
    
    plt.tight_layout()
    logger.experiment.add_figure(f'pred_{idx}', fig)
    plt.close(fig)

@hydra.main(config_path="configs", config_name="predict_tree_pseco", version_base="1.1")
def main(cfg: DictConfig):
    
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Print config
    rprint("[bold blue]Configuration:[/bold blue]")
    rprint(OmegaConf.to_container(cfg, resolve=True))
    
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='predict_tree_pseco',
        version=''
    )
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Initialize model
    TreePseco = build_tree_pseco(cfg)
    
    # Initialize folder dataset
    dataset = FolderDataset(cfg.data.images_dir)
    dl = DataLoader(dataset, batch_size=1)
    
    progress_bar = tqdm(dl, desc=f"Processing {cfg.data.images_dir} folder", unit="batch")
    
    for idx, x in enumerate(progress_bar):
        x = x.to('cuda')
        tree_bboxes_np, tree_pred_scores_np, masks_np = TreePseco(x)
        
        sys_metrics = get_system_usage_metrics()
        progress_bar.set_postfix(sys_metrics)
        
        log_fig(logger, idx, x, tree_bboxes_np, masks_np)

if __name__ == '__main__':
    main()