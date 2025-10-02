import math

import einops
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
import torch.nn.functional as F
from . import dist_util
from .cross_attn import Decoder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class VGGPyramid(nn.Module):
    def __init__(self, input_size):
        super(VGGPyramid, self).__init__()
        self.input_size = input_size

        self.level_0 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.level_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.level_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        if self.input_size == 64 or self.input_size == 32 or self.input_size == 16:
            self.level_3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        elif self.input_size == 128:
            self.level_3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        # self._load_pretrained_weights()

    # def _load_pretrained_weights(self):
    #     from torchvision.models import vgg16, VGG16_Weights
    #     pretrained_vgg = vgg16(weights=VGG16_Weights.DEFAULT)
    #     pretrained_features = list(pretrained_vgg.features.children())
    #     # Map pretrained weights to level_1
    #     self.level_1[0].weight.data = pretrained_features[2].weight.data.clone()
    #     self.level_1[0].bias.data = pretrained_features[2].bias.data.clone()
        
    #     # Map pretrained weights to level_2 and level_3
    #     self.level_2[0].weight.data = pretrained_features[5].weight.data.clone()
    #     self.level_2[0].bias.data = pretrained_features[5].bias.data.clone()
    #     self.level_2[2].weight.data = pretrained_features[7].weight.data.clone()
    #     self.level_2[2].bias.data = pretrained_features[7].bias.data.clone()

    #     self.level_3[0].weight.data = pretrained_features[10].weight.data.clone()
    #     self.level_3[0].bias.data = pretrained_features[10].bias.data.clone()
    #     self.level_3[2].weight.data = pretrained_features[12].weight.data.clone()
    #     self.level_3[2].bias.data = pretrained_features[12].bias.data.clone()
    #     self.level_3[4].weight.data = pretrained_features[14].weight.data.clone()
    #     self.level_3[4].bias.data = pretrained_features[14].bias.data.clone()


    def forward(self, x, eigth_resolution=True):
        outputs_list = []
        outputs = {}
        outputs['level_0'] = self.level_0(x)
        outputs['level_1'] = self.level_1(outputs['level_0'])
        outputs['level_2'] = self.level_2(outputs['level_1'])
        outputs['level_3'] = self.level_3(outputs['level_2'])
        
        outputs_list.append(outputs['level_0'])  
        outputs_list.append(outputs['level_1'])  
        outputs_list.append(outputs['level_2'])  
        outputs_list.append(outputs['level_3'])
        return outputs_list

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio,
        separate_cross_attn,
        **block_kwargs,
    ):
        super().__init__()
        # hidden_size = hidden_size*3
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # hidden_size = int(hidden_size/3)
        self.separate_cross_attn = separate_cross_attn
        if self.separate_cross_attn == 'seq':
            self.cross_obs_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_obs_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )
            self.cross_act_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn_act = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )
        else:
            self.cross_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )
        

    def forward(self, x, t, tv, cond, msk6=None, msk_line=None, r=None): 
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=1)
        )
        if self.separate_cross_attn=='seq': 
            obs_conditioned, _ = self.cross_obs_attn(
                query=self.cross_obs_norm(x),  # (N, T, D)
                key=msk6,  # (N, steps * T, D)
                value=msk6,  # (N, steps * T, D)
                need_weights=False,
            )
            x = x + obs_conditioned  # (N, T, D)
            
            conditioned, _ = self.cross_attn(
                query=self.cross_norm(x),  # (N, T, D)
                key=cond,  # (N, steps, D)
                value=cond,  # (N, steps, D)
                need_weights=False,
            )
            x = x + conditioned  # (N, T, D)
            
            act_conditioned, _ = self.cross_attn_act(
                query=self.cross_act_norm(x),  # (N, T, D)
                key=msk_line,  # (N, steps, D)
                value=msk_line,  # (N, steps, D)
                need_weights=False,
            )
            x = x + act_conditioned  # (N, T, D)
        elif self.separate_cross_attn=='para':
            conditioned1, _ = self.cross_attn(
                query=self.cross_norm(x),  # (N, T, D)
                key=cond,  # (N, steps, D)
                value=cond,  # (N, steps, D)
                need_weights=False,
            )
            x1 = x + conditioned1  # (N, T, D)
            conditioned2, _ = self.cross_attn(
                query=self.cross_norm(x),  # (N, T, D)
                key=msk6,  # (N, steps, D)
                value=msk6,  # (N, steps, D)
                need_weights=False,
            )
            x2 = x + conditioned2  # (N, T, D)
            conditioned3, _ = self.cross_attn(
                query=self.cross_norm(x),  # (N, T, D)
                key=msk_line,  # (N, steps, D)
                value=msk_line,  # (N, steps, D)
                need_weights=False,
            )
            x3 = x + conditioned3  # (N, T, D)   
            if tv is True:
                conditioned4, _ = self.cross_attn(
                    query=self.cross_norm(x),  # (N, T, D)
                    key=r,  # (N, steps, D)
                    value=r,  # (N, steps, D)
                    need_weights=False,
                )
                x4 = x + conditioned4  # (N, T, D)   
            
            
            x1 = x1 + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x1), shift_msa, scale_msa)
            )          
            x1 = x1 + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x1), shift_mlp, scale_mlp)
            ) 
            x2 = x2 + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x2), shift_msa, scale_msa)
            )          
            x2 = x2 + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x2), shift_mlp, scale_mlp)
            ) 
            x3 = x3 + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x3), shift_msa, scale_msa)
            )
            x3 = x3 + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x3), shift_mlp, scale_mlp)
            )
            if tv is True:
                x4 = x4 + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x4), shift_msa, scale_msa)
                )
                x4 = x4 + gate_mlp.unsqueeze(1) * self.mlp(
                    modulate(self.norm2(x4), shift_mlp, scale_mlp)
                ) 
                return x4, x3, x2, x1
            else:
                return x3, x2, x1            
        elif self.separate_cross_attn=='one':
            conditioned, _ = self.cross_attn(
                query=self.cross_norm(x),  # (N, T, D)
                key=cond,  # (N, steps, D)
                value=cond,  # (N, steps, D)
                need_weights=False,
            )
            x = x + conditioned  # (N, T, D)
            
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

class FinalLayer2(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, tv):
        if tv is True:
            shift, scale = self.adaLN_modulation(t.repeat(1, 4)).chunk(2, dim=1)
        else:
            shift, scale = self.adaLN_modulation(t.repeat(1, 3)).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio=4.0,
        time_frequency_embedding_size=256,
        learn_sigma = False,
        separate_cross_attn='para', # para one
        use_pretrain_VGG = False,
        resume=False,
        tv=False
    ):
        super().__init__()
        self.in_channels = in_channels # 2
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_pretrain_VGG = use_pretrain_VGG
        self.separate_cross_attn = separate_cross_attn
        self.tv = tv
        if self.use_pretrain_VGG: # false
            # from ..feature_backbones.VGG_features import VGGPyramid
            self.pyramid = VGGPyramid(train=True) # torchvision
        else:
            self.pyramid = VGGPyramid(input_size) # rebuild true
        # print(self.pyramid)
        self.obs_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        # if self.tv is True:
        self.r_embedder = PatchEmbed(
            input_size, patch_size, in_chans=258, embed_dim=hidden_size, bias=True # 256
        )
        self.c_embedder = PatchEmbed(
            input_size, patch_size, in_chans=256, embed_dim=hidden_size, bias=True # 256
        )
        self.m_embedder = PatchEmbed(
            input_size, patch_size, in_chans=384, embed_dim=hidden_size, bias=True # 256
        )
        self.l_embedder = PatchEmbed(
            input_size, patch_size, in_chans=64, embed_dim=hidden_size, bias=True # 256
        )                
        self.t_embedder = TimestepEmbedder(hidden_size, time_frequency_embedding_size)
        # self.num_conditioning_steps = num_conditioning_steps
        # self.previous_obs_embedder = PreviousObservationEmbedder(
        #     self.obs_embedder, hidden_size
        # )
        # self.act_embedder = ActionEmbedder(num_actions, hidden_size)
        num_patches = self.obs_embedder.num_patches
        self.num_patches = num_patches
        
        # Will use fixed sin-cos embedding:
        self.noised_obs_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    separate_cross_attn=self.separate_cross_attn
                )
                for _ in range(depth)
            ]
        )
        # temporal_embed = get_1d_sincos_pos_embed_from_grid(
        #     hidden_size,
        #     np.arange(num_conditioning_steps, dtype=np.float32),
        # )
        # self.temporal_embed = nn.Parameter(
        #     torch.from_numpy(temporal_embed).float().unsqueeze(0), requires_grad=False
        # )
        
        if self.separate_cross_attn=='para':
            if self.tv is True:
                self.decoder = Decoder(n_layers=6, n_head=6, d_model=384 * 4, d_k= 64 * 4,
                            d_v=64 * 4,
                            d_inner=2048, n_position=input_size//2)
                self.final_layer2 = FinalLayer2(hidden_size*4, patch_size, self.out_channels)
            else:
                self.decoder = Decoder(n_layers=6, n_head=6, d_model=384 * 3, d_k= 64 * 3,
                            d_v=64 * 3,
                            d_inner=2048, n_position=32)
                self.final_layer2 = FinalLayer2(hidden_size*3, patch_size, self.out_channels)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if resume==False:
            self.initialize_weights()
        else:
            pass

    # def load_pretrained_weights(self, weights):
    #     key_mapping = {
    #         "pos_embed": "noised_obs_pos_embed",
    #         "x_embedder.proj.weight": "obs_embedder.proj.weight",
    #         "x_embedder.proj.bias": "obs_embedder.proj.bias",
    #     }
    #     for map_from, map_to in key_mapping.items():
    #         if map_from in weights:
    #             weights[map_to] = weights.pop(map_from)
    #     missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
    #     print(
    #         f"Loaded pretrained weights. Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}"
    #     )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        noised_obs_pos_embed = get_2d_sincos_pos_embed(
            self.noised_obs_pos_embed.shape[-1],
            int(self.obs_embedder.num_patches**0.5),
        )
        self.noised_obs_pos_embed.data.copy_(
            torch.from_numpy(noised_obs_pos_embed).float().unsqueeze(0)
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.obs_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.obs_embedder.proj.bias, 0)
        # nn.init.normal_(self.act_embedder.embedding_table.weight, std=0.02)

        w2 = self.c_embedder.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.c_embedder.proj.bias, 0)
              
        if self.m_embedder is not None:
            w3 = self.m_embedder.proj.weight.data
            nn.init.xavier_uniform_(w3.view([w3.shape[0], -1]))
            nn.init.constant_(self.m_embedder.proj.bias, 0)

        if self.l_embedder is not None:
            w4 = self.l_embedder.proj.weight.data
            nn.init.xavier_uniform_(w4.view([w4.shape[0], -1]))
            nn.init.constant_(self.l_embedder.proj.bias, 0) 
        
        # if self.tv is True:
        if self.r_embedder is not None:
            w5 = self.r_embedder.proj.weight.data
            nn.init.xavier_uniform_(w5.view([w5.shape[0], -1]))
            nn.init.constant_(self.r_embedder.proj.bias, 0)

        if self.use_pretrain_VGG == False:
            for m in self.pyramid.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


        if self.separate_cross_attn=='para':
            nn.init.constant_(self.final_layer2.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer2.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer2.linear.weight, 0)
            nn.init.constant_(self.final_layer2.linear.bias, 0)
        else:
            # Zero-out output layers:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.obs_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, y512=None, mask_y512=None, init_flow=None, local_corr=None, 
                trg_feat = None, src_feat = None, src_64 = None, mask_x=None, tv=None, source_0=None, 
                tmode = None, line_msk = None, mask_cat = None, init_feat=None, iter=False, mode =None):
        x = (self.obs_embedder(x) + self.noised_obs_pos_embed)  # (N, T, D), where T = H * W / patch_size ** 2 # [3, 1024, 384]
        t_flag = t


        if mode is None:
            if t[0]>600:
                t = torch.tensor([2, 2]).to(device=t.device)
            elif 600>t[0]>300:
                t = torch.tensor([1, 1]).to(device=t.device)
        t = self.t_embedder(t)  # (N, D) # [3, 384]

        
        if src_feat == None: # true
            if mask_y512 is not None: # [16, 384, 64, 64] # true
                msk6 = self.m_embedder(mask_y512) + self.noised_obs_pos_embed  # [3, 1024, 384]
            if mask_cat is not None:  # [16, 1, 512, 512]  # false
                y512 = torch.cat([y512, mask_cat], dim=1) # [3, 4, 512, 512]
                
            feat = self.pyramid(y512, eigth_resolution=True)[-1] # [3, 256, 64, 64]
            if init_flow.shape[-1]==32:
                feat = F.interpolate(feat, size=(32), mode='bilinear', align_corners=True)
            elif init_flow.shape[-1]==16:
                feat = F.interpolate(feat, size=(16), mode='bilinear', align_corners=True)
            cond = self.c_embedder(feat) + self.noised_obs_pos_embed # [3, 1024, 384]

            if tv is True:
                if t_flag[0]>600 and iter == True:
                    init_feat = feat 
                elif len(t_flag)>1 and iter == True:
                    idx = [i for i, num in enumerate(t_flag) if num == 2]
                    init_feat[idx] = feat[idx]
                r_in = torch.cat([init_flow, init_feat], dim=1) # [b, 258, 64, 64]
                r = self.r_embedder(r_in) + self.noised_obs_pos_embed # [b, 1024, 384]
            if line_msk is not None: # [16, 64, 64, 64] # true
                msk_line = self.l_embedder(line_msk) + self.noised_obs_pos_embed  # [3, 1024, 384]
            else:
                msk_line = None
        else: # false
            # src_feat = src_feat.flatten(2).transpose(1, 2)
            cond = self.c_embedder(src_feat) + self.noised_obs_pos_embed # [3, 1024, 384]
            # init_flow_cat = F.interpolate(init_flow, size=(32), mode='bilinear', align_corners=True)
            # init_flow_cat = init_flow_cat.flatten(2).transpose(1, 2)  # NCHW -> NTD [3, 2, 32, 32]-> [3, 1024, 2]
            # cond = torch.cat([src_feat, init_flow_cat], dim=2)  # [3, 1024, 384]
        if iter is True or tv is True: # true
            for block in self.blocks:
                x4, x3, x2, x1  = block(x, t, tv, cond, msk6, msk_line, r)  # 4*(N, T, D) -> [N, 1024, 384]
            n, _, d = x3.size()
            if feat.shape[-1]==16:
                x = torch.cat([x1, x2, x3, x4], dim=2).transpose(1, 2).contiguous().view(n, 4*d, 8, 8) # (N, T, 3D)->[N, 384*4, 32, 32] [1, 1536, 64, 64]
            elif feat.shape[-1]==32:
                x = torch.cat([x1, x2, x3, x4], dim=2).transpose(1, 2).contiguous().view(n, 4*d, 16, 16) # (N, T, 3D)->[N, 384*4, 32, 32] [1, 1536, 64, 64]
            else:
                x = torch.cat([x1, x2, x3, x4], dim=2).transpose(1, 2).contiguous().view(n, 4*d, 32, 32) # (N, T, 3D)->[N, 384*4, 32, 32] [1, 1536, 64, 64]
            x = self.decoder(x)  # [12, 1024, 1152]  [N, 384*4, 32, 32]
            x = self.final_layer2(
                x, t, tv
            )  # (N, T, patch_size ** 2 * out_channels) # [3, 1024, 8]  
        elif self.separate_cross_attn=='para': # true
            for block in self.blocks:
                x3, x2, x1 = block(x, t, tv, cond, msk6, msk_line)  # 3*(N, T, D)
            n, _, d = x3.size()
            x = torch.cat([x1, x2, x3], dim=2).transpose(1, 2).contiguous().view(n, 3*d, 32, 32) # (N, T, 3D)-> [N, 1152, 32, 32]
            x = self.decoder(x)  # [12, 1024, 1152]  
            x = self.final_layer2(
                x, t, tv
            )  # (N, T, patch_size ** 2 * out_channels) # [3, 1024, 8]    
        else:
            for block in self.blocks:
                x = block(x, t, tv, cond, msk6, msk_line)  # (N, T, D)
            x = self.final_layer(
                x, t
            )  # (N, T, patch_size ** 2 * out_channels) # [3, 1024, 8]
            
        x = self.unpatchify(x)  # (N, out_channels, H, W) # [3, 2, 64, 64]
        if init_flow is not None:
            x += init_flow
        return x, feat

    @property
    def device(self):
        return next(self.parameters()).device



















#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)



DiT_models2 = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
