import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

def show_sam_masks(segmentations: np.ndarray, ax=None):
    """segmentations: (num_masks, height, width)
    """
    if segmentations.shape[0] == 0:
        return
    areas = np.sum(segmentations, axis=(1, 2))
    sorted_idxs = np.argsort(areas)[::-1]
    sorted_segmentations = segmentations[sorted_idxs]
    
    if ax is None:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    
    img = np.ones((segmentations.shape[1], segmentations.shape[2], 4))
    img[:,:,3] = 0
    for single_segmentation in sorted_segmentations: # single_segmentation is a binary mask (h,w)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[single_segmentation] = color_mask
    ax.imshow(img)

def viz_img_heat_seg_box(image:np.ndarray, heatmap: np.ndarray, peaks_np: np.ndarray, segmentations: np.ndarray,
                    pred_boxes: np.ndarray, gt_boxes: np.ndarray, pred_boxes_scores: np.ndarray=None):
    """Plot a fig with:
        - Original image
        - masks and points
        - gt boxes
        - pred boxes
        
    Args:
        image: Original image (height, width, 3) (1024, 1024, 3)
        heatmap: Heatmap as (height, width) (256, 256) or (1024, 1024), f32 or unit8
        pred_boxes: Predicted bounding boxes as (N, 4) where each box is [x1, y1, x2, y2]
        gt_boxes: Ground truth bounding boxes as (M, 4) where each box is [x1, y1, x2, y2]
        masks: Optional segmentation masks as (num_mask, height, width)
        pred_points: Optional points that generated the masks as (num_points, 2) where each point is [x, y]
        pred_boxes_scores: Optional scores for each predicted box as (N,) in the range [0, 1]
    
    Returns:
        fig: Matplotlib figure containing the visualization
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.tight_layout(pad=3.0)
    
    # 1. Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # 2 Heatmap
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((1024, 1024), Image.BILINEAR)
    heatmap = np.array(heatmap)
    axes[0, 1].imshow(heatmap, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].scatter(peaks_np[:, 0], peaks_np[:, 1], c='r', s=15, marker='x')
    axes[0, 1].set_title("Heatmap")
    axes[0, 1].axis('off')
    
    # 3. Masks visualization
    num_masks = segmentations.shape[0]
    axes[1,0].imshow(image)
    if segmentations is not None and num_masks > 0:
        show_sam_masks(segmentations, axes[1,0])
    axes[1,0].axis('off')
    axes[1, 0].set_title(f"Segmentation Masks ({num_masks} masks)")
    
    # 4. Ground truth bounding boxes on original image
    axes[1, 1].imshow(image)
    axes[1, 1].set_title("GT and Pred. BBoxes")
    axes[1, 1].axis('off')

    # Draw predicted boxes in red
    if pred_boxes_scores is None:
        pred_boxes_scores = [1] * len(pred_boxes)
    for box, score in zip(pred_boxes, pred_boxes_scores):
        if box is not None:
            try:
                if hasattr(box, 'tolist'):
                    x1, y1, x2, y2 = box.tolist()
                else:
                    x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                alpha = max(0, min(1, score))
                rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor=(1, 0, 0, alpha), facecolor='none')
                axes[1, 1].add_patch(rect)
            except Exception as e:
                print(f"Error drawing pred box {box}: {e}")
    
    # Draw ground truth boxes in green
    for box in gt_boxes:
        # Handle different possible formats
        try:
            if hasattr(box, 'tolist'):
                x1, y1, x2, y2 = box.tolist()
            else:
                x1, y1, x2, y2 = box
            
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
            axes[1, 1].add_patch(rect)
        except Exception as e:
            print(f"Error drawing GT box {box}: {e}") 
    
    plt.tight_layout()
    return fig