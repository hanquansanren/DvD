import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (SiLU, avg_pool_nd, checkpoint, conv_nd, linear, normalization,
                 timestep_embedding, zero_module)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


        
class DDIMWithTransformer(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_heads, num_layers, ff_dim, dropout=0.1,train_mode=None):
        super(DDIMWithTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.train_mode= train_mode
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, model_channels),
        )

        # Input blocks (Transformer blocks)
        self.input_blocks = nn.ModuleList([TransformerBlock(model_channels, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        # Middle block (Bottleneck)
        self.middle_block = TransformerBlock(model_channels, num_heads, ff_dim, dropout)

        # Output blocks (inverse layers)
        self.output_blocks = nn.ModuleList([TransformerBlock(model_channels, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        # Final output layer
        self.out = nn.Sequential(
            # nn.LayerNorm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
        
        self.x_projection = nn.Conv2d(in_channels=68, out_channels=128, kernel_size=3, padding=1) 


    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype


    def forward(self, x, timesteps, y=None, y512=None, init_flow=None, local_corr=None, 
                trg_feat=None, src_feat=None, src_64=None, mask_x=None, tv=None, source_0=None):
        # Embedding the time steps
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # [3, 128]
        # x = x.type(self.inner_dtype)

        if self.train_mode == "stage_1_transformer" and init_flow is not None: # or self.train_mode == "sr":
            # b, _, H, W = init_flow.shape
            # h = th.cat([x, init_flow, local_corr], dim=1) #original [24, 2, 64, 64] [24, 2, 64, 64] [24, 81, 64, 64]
            # h = th.cat([src_64, x, init_flow], dim=1) #7= [24, 3, 64, 64] [24, 2, 64, 64] [24, 2, 64, 64]
            x = torch.cat([src_feat, x, init_flow], dim=1) #68= [24, 64, 64, 64] [24, 2, 64, 64] [24, 2, 64, 64]  
            x = self.x_projection(x)

        # Reshape the input tensor for Transformer (flatten the spatial dimensions)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(2, 0, 1)  # [H*W, N, C] [4096,3,128]



        # Incorporate timestep embedding into each Transformer block
        for block in self.input_blocks: # 
            x = block(x + emb.unsqueeze(0))  # Add timestep embedding to input [4096,3,128]
        x = self.middle_block(x + emb.unsqueeze(0))  # Middle block also gets the timestep embedding
        for block in self.output_blocks:
            x = block(x + emb.unsqueeze(0))  # Add timestep embedding to each output block
        
        # Convert back to the image shape
        x = x.permute(1, 2, 0).view(b, c, h, w)
        x = self.out(x)
        
        if init_flow is not None:
            x += init_flow
        return x
