import torch
import torch.nn as nn

from typing import Tuple
from .segment_anything.modeling.mask_decoder import MLP
from .segment_anything.modeling.common import LayerNorm2d
from .segment_anything.modeling.transformer import TwoWayTransformer
from .segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
from .segment_anything.modeling.sam import Sam

class SAMPointDecoder(nn.Module):
    def __init__(self, sam:Sam, mask_upscaling_type:str='default', use_sam_pretrained:bool=True) -> None:
        super().__init__()
        
        self.sam = sam
        
        transformer_dim = 256
        activation = nn.GELU
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )
        self.mask_tokens = nn.Embedding(1, transformer_dim)
        # mask_upscaling_type ='default' 
        if mask_upscaling_type == 'default':
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(),
            )
        elif mask_upscaling_type == 'bilinear':
            self.output_upscaling = nn.Sequential(
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True), # (256, 64, 64) -> (256, 256, 256)
                nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1), # (265, 256, 256) -> (32, 256, 256)
            )
        else:
            raise ValueError("Invalid mask_upscaling_type")
        
        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        if use_sam_pretrained:
            self.transformer.load_state_dict(sam.mask_decoder.transformer.state_dict())
            self.output_hypernetworks_mlp.load_state_dict(sam.mask_decoder.output_hypernetworks_mlps[0].state_dict())
            if mask_upscaling_type == 'default':
                self.output_upscaling.load_state_dict(sam.mask_decoder.output_upscaling.state_dict())
        embed_dim = 256
        self.image_embedding_size = (64, 64)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = False
    
    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def forward_sam(self, x):
        # x -> (b,c,h,w)
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x.cuda())
        return image_embeddings
    
    def forward(self, x, masks=None, image_embeddings=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ x -> (b,c,h,w)
            masks -> (b,h,w)
            image_embeddings -> (b,c,h,w)
        """
        if image_embeddings is None:
            image_embeddings = self.forward_sam(x)
        
        output_tokens = self.mask_tokens.weight[0].unsqueeze(0)
        sparse_embeddings = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        
        image_pe = self.get_dense_pe()
        
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape
        
        # In the original implementation sparse_embeddings are 7 tokens
        # (1 iou_token, 4 for masks, 1 box position token, 1 points position token)
        # Here we have only 1 token that needs to learn how to extract the desired heatmap
        # In this TwoWayTransformer the mask token attend to the image and then vice versa
        # hs is the refined mask token, src are the refined image tokens
        hs, src = self.transformer(src, pos_src, sparse_embeddings)
        # Here we take the token back after it has exchanged information with the image and viceversa
        mask_tokens_out = hs[:, 0, :]
        
        src = src.transpose(1, 2).view(b, c, h, w) # reconstruct the image spatially
        
        upscaled_embedding = self.output_upscaling(src) # learnable upscaling
        
        hyper_in = self.output_hypernetworks_mlp(mask_tokens_out).unsqueeze(1) # refine mask token and reduce its dimension
        b, c, h, w = upscaled_embedding.shape
        
        # predict the heatmap with dot product between the mask token and the upsampled image embedding in every pixel position
        pred_heatmaps = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        if masks is not None:
            pred_heatmaps *= masks
        
        return pred_heatmaps, image_embeddings
