import torch
import psutil
import os

from pytorch_lightning.callbacks import TQDMProgressBar

class RamGpuUsageProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        
        # Total system RAM usage for all processes in the current process tree
        process = psutil.Process()
        total_ram = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total_ram += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        items["RAM"] = f"{total_ram / 1024 / 1024 / 1024:.2f}GB"
        
        # GPU memory metrics
        if torch.cuda.is_available():
            # Reserved memory
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            items["GPU"] = f"{reserved:.2f}GB"
            
        return items

def get_system_usage_metrics():
    """Calculates and returns RAM and GPU memory usage metrics."""
    metrics = {}
    
    # --- RAM Usage ---
    # Get the current process
    process = psutil.Process(os.getpid())
    # Get RAM usage of the current process (rss = Resident Set Size)
    # Use memory_info().rss for cross-platform compatibility
    total_ram_bytes = process.memory_info().rss
    # Include memory usage of child processes if any (e.g., dataloader workers)
    try:
        for child in process.children(recursive=True):
            try:
                total_ram_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Ignore if child process has exited or access is denied
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Ignore if main process info cannot be accessed (less likely)
        pass
    
    # Convert bytes to Gigabytes (GiB using 1024^3)
    metrics["RAM"] = f"{total_ram_bytes / (1024**3):.2f}GB"
    
    # --- GPU Memory Usage ---
    if torch.cuda.is_available():
        # torch.cuda.memory_reserved(): Total memory managed by the caching allocator
        # torch.cuda.memory_allocated(): Tensor memory currently occupied
        # Reserved is often more indicative of the total footprint on the GPU
        reserved_bytes = torch.cuda.memory_reserved(0) # Use device index 0
        # allocated_bytes = torch.cuda.memory_allocated(0) # Optional: track allocated too
        
        metrics["GPU"] = f"{reserved_bytes / (1024**3):.2f}GB"
        # metrics["GPU alloc"] = f"{allocated_bytes / (1024**3):.2f}GB" # Optional
    else:
        metrics["GPU"] = "N/A" # Indicate no GPU is used/available
    
    return metrics