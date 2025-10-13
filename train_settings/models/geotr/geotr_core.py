import copy
import os
from typing import Optional

from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import save_image

from .extractor import BasicEncoder, BasicEncoder2
from .position_encoding import build_position_encoding

from train_settings.dvd.improved_diffusion import dist_util, logger
from collections import OrderedDict

import torch
import numpy as np
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure
from train_settings.dvd.improved_diffusion.nn import (SiLU, avg_pool_nd, checkpoint, conv_nd, linear, normalization,
                 timestep_embedding, zero_module)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout




## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin




class attnLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_list = nn.ModuleList(
            [
                copy.deepcopy(nn.MultiheadAttention(d_model, nhead, dropout=dropout))
                for i in range(2)
            ]
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_list = nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(d_model)) for i in range(2)]
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2_list = nn.ModuleList(
            [copy.deepcopy(nn.Dropout(dropout)) for i in range(2)]
        )
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward_post(
        self,
        tgt,
        memory_list,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        memory_pos=None,
    ):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        for memory, multihead_attn, norm2, dropout2, m_pos in zip(
            memory_list,
            self.multihead_attn_list,
            self.norm2_list,
            self.dropout2_list,
            memory_pos,
        ):
            tgt2 = multihead_attn(
                query=self.with_pos_embed(tgt, pos),
                key=self.with_pos_embed(memory, m_pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + dropout2(tgt2)
            tgt = norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        memory_pos=None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory_list,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        memory_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory_list,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                memory_pos,
            )
        return self.forward_post(
            tgt,
            memory_list,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            memory_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransDecoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, query_embed):
        pos = self.position_embedding(
            torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3]).bool().cuda()
        )  # torch.Size([1, 128, 36, 36])

        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            query_embed = layer(query_embed, [imgf], pos=pos, memory_pos=[pos, pos])
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h, w)

        return query_embed


class TransEncoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf):
        pos = self.position_embedding(
            torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3]).bool().cuda()
        )  # torch.Size([1, 128, 36, 36])
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            imgf = layer(imgf, [imgf], pos=pos, memory_pos=[pos, pos])
        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)

        return imgf


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, imgf, coords1):
        mask = 0.25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x




class GeoTr2(nn.Module):
    def __init__(self, num_attn_layers, num_token=32*32):
        super(GeoTr2, self).__init__()
        self.num_attn_layers = num_attn_layers

        self.hidden_dim = hdim = 256

        self.fnet = BasicEncoder2(output_dim=hdim, norm_fn="instance")
        
        self.TransEncoder = TimestepEmbedSequential(TransEncoder(self.num_attn_layers, hidden_dim=hdim))
        self.TransDecoder = TimestepEmbedSequential(TransDecoder(self.num_attn_layers, hidden_dim=hdim))
        self.query_embed = nn.Embedding(num_token, self.hidden_dim)

        self.update_block = UpdateBlock(self.hidden_dim)
        # Time embedding
        time_embed_dim = 128 * 4
        self.time_embed = nn.Sequential(
            nn.Linear(128, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        H = H*8
        W = W*8
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, x, timesteps, y=None, y512=None, init_flow=None, local_corr=None, 
                trg_feat=None, src_feat=None, src_64=None, mask_x=None, tv=None, source_0=None):
        # fmap: [b, 68, 64, 64]
        
        emb = self.time_embed(timestep_embedding(timesteps, 128)) #[1, 512]
        
        if self.train_mode == "stage_1_doctr" and init_flow is not None: # or self.train_mode == "sr":
            # b, _, H, W = init_flow.shape
            # h = th.cat([x, init_flow, local_corr], dim=1) #original [24, 2, 64, 64] [24, 2, 64, 64] [24, 81, 64, 64]
            # h = th.cat([src_64, x, init_flow], dim=1) #7= [24, 3, 64, 64] [24, 2, 64, 64] [24, 2, 64, 64]
            fmap = torch.cat([src_feat, x, init_flow], dim=1) #68= [24, 64, 64, 64] [24, 2, 64, 64] [24, 2, 64, 64]

        
        fmap = self.fnet(fmap)  # [b, 68, 64, 64] -> [b, 256, 32, 32]
        fmap = torch.relu(fmap)

        fmap = self.TransEncoder(fmap, emb) # [1, 256, 32, 32]
        fmap = self.TransDecoder(fmap, self.query_embed.weight) # [1, 256, 32, 32]

        # convex upsample baesd on fmap
        coodslar, coords0, coords1 = self.initialize_flow(fmap)
        coords1 = coords1.detach() # [1, 2, 32, 32]
        mask, coords1 = self.update_block(fmap, coords1)
        bm_up = self.upsample_flow(coords1 - coords0, mask) # [1, 2, 256, 256] ±256
        bm_up = F.interpolate(bm_up, size=(64, 64), mode='bilinear', align_corners=True)/256. # [-1,+1] 偏移场
        # [Bs, 2, 64, 64]
        # bm_up = coodslar + bm_up

        return bm_up




class GeoTr(nn.Module):
    def __init__(self, num_attn_layers, num_token):
        super(GeoTr, self).__init__()
        self.num_attn_layers = num_attn_layers

        self.hidden_dim = hdim = 256

        self.fnet = BasicEncoder(output_dim=hdim, norm_fn="instance")

        self.TransEncoder = TransEncoder(self.num_attn_layers, hidden_dim=hdim)
        self.TransDecoder = TransDecoder(self.num_attn_layers, hidden_dim=hdim)
        self.query_embed = nn.Embedding(num_token, self.hidden_dim)

        self.update_block = UpdateBlock(self.hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1):
        fmap = self.fnet(image1) 
        fmap = torch.relu(fmap) # [1, 256, 36, 36]

        fmap = self.TransEncoder(fmap)
        fmap = self.TransDecoder(fmap, self.query_embed.weight) # [1, 256, 36, 36]

        # convex upsample baesd on fmap
        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach() # [1, 2, 36, 36]
        mask, coords1 = self.update_block(fmap, coords1)
        bm_up = self.upsample_flow(coords1 - coords0, mask) # [1, 2, 288, 288] ±288
        # bm_up = coodslar + bm_up

        return bm_up

# class U2NETP(nn.Module):

#     def __init__(self, in_ch=3, out_ch=1):
#         super(U2NETP, self).__init__()

#         self.stage1 = RSU7(in_ch, 16, 64)
#         self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.stage2 = RSU6(64, 16, 64)
#         self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.stage3 = RSU5(64, 16, 64)
#         self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.stage4 = RSU4(64, 16, 64)
#         self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.stage5 = RSU4F(64, 16, 64)
#         self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.stage6 = RSU4F(64, 16, 64)

#         # decoder
#         self.stage5d = RSU4F(128, 16, 64)
#         self.stage4d = RSU4(128, 16, 64)
#         self.stage3d = RSU5(128, 16, 64)
#         self.stage2d = RSU6(128, 16, 64)
#         self.stage1d = RSU7(128, 16, 64)

#         self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

#         self.outconv = nn.Conv2d(6, out_ch, 1)

#     def forward(self, x):
#         hx = x

#         # stage 1
#         hx1 = self.stage1(hx)
#         hx = self.pool12(hx1)

#         # stage 2
#         hx2 = self.stage2(hx)
#         hx = self.pool23(hx2)

#         # stage 3
#         hx3 = self.stage3(hx)
#         hx = self.pool34(hx3)

#         # stage 4
#         hx4 = self.stage4(hx)
#         hx = self.pool45(hx4)

#         # stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)

#         # stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample_like(hx6, hx5)

#         # decoder
#         hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
#         hx5dup = _upsample_like(hx5d, hx4)

#         hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
#         hx4dup = _upsample_like(hx4d, hx3)

#         hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample_like(hx3d, hx2)

#         hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample_like(hx2d, hx1)

#         hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

#         # side output
#         d1 = self.side1(hx1d)

#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2, d1)

#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3, d1)

#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4, d1)

#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5, d1)

#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6, d1)

#         d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

#         return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
#             d4), torch.sigmoid(d5), torch.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        features = []
        hx6 = self.stage6(hx)

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        features.append(hx2dup)
        features.append(hx3dup)
        features.append(hx4dup)
        features.append(hx5dup)
        features.append(hx6up)

        # side output
        d1 = self.side1(hx1d)

        d2_ = self.side2(hx2d)
        d2 = _upsample_like(d2_, d1)

        d3_ = self.side3(hx3d)
        d3 = _upsample_like(d3_, d1)

        d4_ = self.side4(hx4d)
        d4 = _upsample_like(d4_, d1)

        d5_ = self.side5(hx5d)
        d5 = _upsample_like(d5_, d1)

        d6_ = self.side6(hx6)
        d6 = _upsample_like(d6_, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), hx6, hx5d, hx4d, hx3d, hx2d, hx1d



class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6, num_token=(288//8)**2)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        # save_image(msk[0],"vis_hp/debug_vis/msk.png")
        msk = (msk > 0.5).float()
        # msk = keep_largest_connected_component(msk).float()
        # save_image(msk[0],"vis_hp/debug_vis/msk_thres.png")
        x = msk * x
        msk = F.interpolate(msk, size=(512), mode='bilinear', align_corners=True) 
        # save_image(x[0],"vis_hp/debug_vis/msk_doc.png") # 0-1

        bm = self.GeoTr(x) # 0~288 bm
        # bm = None
        # bm = (2 * (bm / 286.8) - 1) * 0.99 #归一化到[-1,1]
        
        return bm, x

class Seg(nn.Module):
    def __init__(self):
        super(Seg, self).__init__()
        self.msk = U2NETP(3, 1)

    def forward(self, x):
        d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d = self.msk(x)
        d1 = (d0 > 0.5).float()
        mskx = d1 * x
        d0 = F.interpolate(d0, size=(512), mode='bilinear', align_corners=True)    
        
        return mskx, d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d

class GeoTr_Seg_Inf(nn.Module):
    def __init__(self):
        super(GeoTr_Seg_Inf, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6, num_token=(288//8)**2)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        # save_image(msk[0],"vis_hp/debug_vis/msk.png")
        # msk = (msk > 0.1).float()
        
        # save_image(msk[0],"vis_hp/debug_vis/msk_thres.png")
        x = msk * x
        # save_image(x[0]/200,"vis_hp/debug_vis/msk_doc.png") # 0-1
        msk = F.interpolate(msk, size=(512), mode='bilinear', align_corners=True) 
        # msk = keep_largest_connected_component(msk).float()
        # msk = (msk > 0.7).float() # re 阈值过滤
        
        bm = self.GeoTr(x) # 0~288 bm
        # bm = None
        # bm = (2 * (bm / 286.8) - 1) * 0.99 #归一化到[-1,1]
        
        return bm, msk

class GeoTr_Seg_womask(nn.Module):
    def __init__(self):
        super(GeoTr_Seg_womask, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6, num_token=(288//8)**2)
        
    def forward(self, x):
        # msk, _1,_2,_3,_4,_5,_6 = self.msk(x/255.)
        # # save_image(msk[0],"vis_hp/debug_vis/msk.png")
        # msk = (msk > 0.5).float()
        # # msk = keep_largest_connected_component(msk).float()
        # # save_image(msk[0],"vis_hp/debug_vis/msk_thres.png")
        # x = msk * x
        # # save_image(x[0]/200,"vis_hp/debug_vis/msk_doc.png") # 0-1

        bm = self.GeoTr(x) # 0~288 bm
        # bm = None
        # bm = (2 * (bm / 286.8) - 1) * 0.99 #归一化到[-1,1]
        
        return bm, None


def keep_largest_connected_component(mask):
    # 将mask从PyTorch tensor转换为numpy数组
    mask_np = mask.cpu().numpy()
    
    # 使用scipy的label函数标记连通域
    labeled_mask, num_features = label(mask_np)
    
    # 如果没有连通域，返回原mask
    if num_features == 0:
        return mask
    
    # 计算每个连通域的大小
    unique, counts = np.unique(labeled_mask, return_counts=True)
    
    # 忽略背景（值为0的部分），选择最大连通域
    counts[0] = 0  # 背景部分不计算
    largest_component = unique[np.argmax(counts)]
    
    # 创建一个新的掩码，仅保留最大连通域
    largest_mask = (labeled_mask == largest_component).astype(np.uint8)
    
    # # # 补全最大连通域中的空洞
    # structure = np.ones((1,1, 2, 2), dtype=np.uint8)
    # largest_mask = binary_fill_holes(largest_mask, structure=structure).astype(np.uint8)
    
    # 将结果转换回PyTorch tensor
    largest_mask_tensor = torch.from_numpy(largest_mask).to(mask.device)
    
    return largest_mask_tensor



def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        return model
        

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        # a = dist_util.load_state_dict(path, map_location="cpu")
        # new_state_dict = OrderedDict()
        # for key, value in a.items():
        #     new_key = key.replace('model.', '')  # 去除 'model.'
        #     new_state_dict[new_key] = value
        # model.cpu().load_state_dict(new_state_dict, strict=True)
        
        
        
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        return model

