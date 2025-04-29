import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
import numpy as np
import torch.nn.functional as F

def visualize_segmentation(image, pred_mask, target_mask=None):
    """Visualize segmentation predictions and optional ground truth mask
    
    Args:
        image: Input image tensor of shape [C, H, W]
        pred_mask: Predicted segmentation mask tensor of shape [H, W] 
        target_mask: Optional ground truth mask tensor of shape [H, W]
    """
    # Convert tensor to numpy for visualization
    image_np = image.cpu().permute(1, 2, 0).numpy()
    # Normalize image for display
    # Unnormalize image using ImageNet stats and clip to [0,1] range
    image_np = np.clip(image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    
    # Create figure with tight layout and no borders
    fig = plt.figure(figsize=(12, 12))
    
    # If we have ground truth, create two subplots
    if target_mask is not None:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]
    else:
        ax = fig.add_subplot(111)
        axes = [ax]
    
    for ax in axes:
        # Remove white border
        ax.set_position([0, 0, 1, 1])
        
        # Show original image
        ax.imshow(image_np)
        
    # Plot prediction mask
    pred_mask_np = pred_mask.cpu().numpy()
    # Create a colored overlay for the mask
    colored_mask = np.zeros((*pred_mask_np.shape, 4))  # RGBA
    colored_mask[pred_mask_np == 1] = [1, 0, 0, 0.3]  # Red with 0.3 alpha
    
    axes[0].imshow(colored_mask)
    axes[0].set_title('Prediction', pad=10)
    
    # Plot ground truth if provided
    if target_mask is not None:
        target_mask_np = target_mask.cpu().numpy()
        colored_target = np.zeros((*target_mask_np.shape, 4))
        colored_target[target_mask_np == 1] = [0, 1, 0, 0.3]  # Green with 0.3 alpha
        
        axes[1].imshow(colored_target)
        axes[1].set_title('Ground Truth', pad=10)
    
    # Turn off axes for all subplots
    for ax in axes:
        ax.axis('off')
    
    # Remove all margins
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    return fig

def visualize_predicted_points(image: torch.Tensor, pred_points: torch.Tensor, pred_points_score: torch.Tensor, gt_boxes=None, cluster_labels=None):
    # Convert tensor to numpy for visualization
    image_np = image.cpu().permute(1, 2, 0).numpy()
    # Normalize image for display
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    # Create figure with tight layout and no borders
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    
    # Remove white border
    ax.set_position([0, 0, 1, 1])
    
    ax.imshow(image_np)
    
    if cluster_labels is not None:
        if isinstance(pred_points, torch.Tensor):
            points = pred_points.cpu().numpy()
            
        unique_labels = set(cluster_labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise
                col = [0, 0, 0, 1]
            
            class_mask = cluster_labels == k
            xy = points[class_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=10,
                    label=f'Cluster {k}' if k != -1 else 'Noise')
    
    else:
        # Draw predicted points
        for point, score in zip(pred_points.cpu().numpy(), pred_points_score.cpu().numpy()):
            x, y = point
            # Create circle patch
            circle = patches.Circle(
                (x, y), 2,
                edgecolor='r',
                facecolor='r',
                alpha=1.0  # Higher confidence = more opaque
            )
            ax.add_patch(circle)
            
            # Add confidence score text
            # ax.text(x+12, y+12, f'{score:.2f}', 
            #        bbox=dict(facecolor='red', alpha=score))
    
    # Draw GT boxes
    if gt_boxes is not None:
        for box in gt_boxes.cpu().numpy():
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5,
                edgecolor=(0.22, 1, 0.078),
                facecolor='none'
            )
            ax.add_patch(rect)
    
    plt.axis('off')
    # Remove all margins
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    return fig

def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize an image tensor that was normalized with ImageNet mean and std.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        torch.Tensor: Unnormalized image tensor with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean

def visualize_heatmap(image, heatmap, alpha=0.6, unnormalize=True):
    """
    Create an overlay visualization of the heatmap on the image.

    Args:
        image: [3, H, W] tensor in range [-1, 1] (normalized)
        heatmap: [1, 256, 256] tensor in range [0, 1]
        alpha: transparency of the heatmap overlay

    Returns:
        overlay: [3, H, W] tensor ready for tensorboard
    """
    if unnormalize:
        image = unnormalize_image(image)
    
    image = image.clamp(0, 1)

    # Resize heatmap to match image dimensions
    heatmap = F.interpolate(
        heatmap.unsqueeze(0),  # Add batch dimension [1, 1, 256, 256]
        size=(image.shape[1], image.shape[2]),  # Target size (1024, 1024)
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension

    # Convert heatmap to RGB (red channel)
    heatmap = heatmap.squeeze(0)  # [H, W]
    heatmap_rgb = torch.zeros_like(image)  # [3, H, W]
    heatmap_rgb[0] = heatmap  # Red channel

    # Create overlay
    overlay = image * (1 - alpha) + heatmap_rgb * alpha
    return overlay
    