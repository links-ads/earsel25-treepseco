import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self,
                visualize_images_every_n_epochs: int,
                n_images_to_visualize: int,
                random_images: bool = False,
                ):
        """Create the visualizer that knows when to log images and how many to log.
        
        Parameters
        ----------
        visualize_images_every_n_epochs : int
            how many epoch to wait between visualizations. If you use this for the validation phase this corresponds
            to the number of validations to wait between visualizations.
        n_images_to_visualize : int
            how many images to visualize every time. If set to -1 visualizes all the batches.
        random_images : bool, optional
            if log random images or always the same, by default False
        """
        self.visualize_images_every_n_epochs = visualize_images_every_n_epochs
        self.n_images_to_visualize = n_images_to_visualize
        self.random_images = random_images
        
        self.epochs_done = -1 #since we increment it at the beginning of the epoch
        
        self.logger = None
        self.dataloader_len = None
        
    def on_epoch_start_setup(self, logger: pl.loggers.TensorBoardLogger, data_loader_len: int):
        # It has to be called in on_xxx_epoch_start hook
        assert isinstance(logger, pl.loggers.TensorBoardLogger), "The logger has to be a TensorBoardLogger"
        
        if self.logger is None:
            self.logger = logger
        if self.dataloader_len is None:
            self.dataloader_len = data_loader_len
            assert self.dataloader_len >= self.n_images_to_visualize, "Number of images to visualize is greater than the number of validation batches"
        
        self.epochs_done += 1
        if self.n_images_to_visualize != -1:
            if self.random_images:
                self.batch_idx_to_visualize = np.random.choice(self.dataloader_len, size=self.n_images_to_visualize, replace=False)
            else:
                self.batch_idx_to_visualize = np.linspace(0, self.dataloader_len, self.n_images_to_visualize, dtype=int)
        else:
            self.batch_idx_to_visualize = np.arange(self.dataloader_len)
            
        if self.visualize_images_every_n_epochs == 0:
            self.do_visualization_on_current_val = False
        else:
            self.do_visualization_on_current_val = self.epochs_done % self.visualize_images_every_n_epochs == 0 # this tells if it is the right validation in which log images
    
    def visualize_batch_flag(self, batch_idx: int) -> bool:
        return self.do_visualization_on_current_val and batch_idx in self.batch_idx_to_visualize
    
    def reset(self):
        self.epochs_done = -1
