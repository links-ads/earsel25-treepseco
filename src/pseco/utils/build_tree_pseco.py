from pathlib import Path
import torch

from src.pseco.models.components.segment_anything.build_sam import build_sam_vit_h

def build_tree_pseco(cfg: dict):
    from src.pseco.models import TreePseco
    from src.pseco.models.components import (SAMPointDecoder,
                            HeatmapBoxDetector,
                            )
    from src.pseco.models.components import CustomRcnn    
    
    # Initialize SAM model
    assert cfg.model.point_decoder.sam_type == 'vit_h', "Invalid SAM model"
    sam_vith_path = 'pretrained/SAM/sam_vit_h_4b8939.pth'
    sam = build_sam_vit_h(checkpoint=sam_vith_path)
    
    # POINT DECODER
    mask_upscaling_type = cfg.model.point_decoder.mask_upscaling_type if hasattr(cfg.model.point_decoder, "mask_upscaling_type") else 'default'
    point_decoder = SAMPointDecoder(sam=sam, mask_upscaling_type=mask_upscaling_type, use_sam_pretrained=cfg.model.point_decoder.use_sam_pretrained)
    
    if cfg.model.point_decoder.state_dict.endswith('.ckpt'):
        state_dict = torch.load(cfg.model.point_decoder.state_dict, map_location='cpu', weights_only=True)['state_dict']
    else:
        state_dict = torch.load(cfg.model.point_decoder.state_dict, map_location='cpu', weights_only=True)
        
    point_decoder.load_state_dict(state_dict, strict=False)
    point_decoder = point_decoder.cuda()
    
    # HEATMAP BOX DETECTOR
    heatmap_box_detector = HeatmapBoxDetector(cfg, sam=sam)
    
    # CUSTOM RCNN
    custom_rcnn = CustomRcnn(num_classes=2, use_df_rn50_weights=False)
    if cfg.model.frcnn.state_dict.endswith('.ckpt'):
        state_dict = torch.load(cfg.model.frcnn.state_dict, map_location='cpu', weights_only=True)['state_dict']
    else:
        state_dict = torch.load(cfg.model.frcnn.state_dict, map_location='cpu', weights_only=True)
    adapted_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("custom_rcnn"):
            new_key = k.replace("custom_rcnn.", "")
            adapted_state_dict[new_key] = v
    custom_rcnn.load_state_dict(adapted_state_dict, strict=True)
    custom_rcnn = custom_rcnn.cuda()
    
    model = TreePseco(point_decoder,
                    heatmap_box_detector,
                    custom_rcnn,
                    score_th=cfg.model.frcnn.score_th
                    )
    model = model.eval() # instantiate the model always in eval mode since it is not trainable
    
    return model 