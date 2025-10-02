import numpy as np
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image




@th.no_grad()
def coords_grid(batch, ht, wd, device):
    coords = th.meshgrid(th.arange(ht, device=device), th.arange(wd, device=device))
    coords = th.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)



def extract_features(
    pyramid,
    im_target,
    im_source,
    im_target_256,
    im_source_256,
    im_target_pyr=None,
    im_source_pyr=None,
    im_target_pyr_256=None,
    im_source_pyr_256=None,
):
    if im_target_pyr is None:
        im_target_pyr = pyramid(im_target, eigth_resolution=True) # [[1, 64, 512, 512],[1, 128, 128, 128],[1, 256, 64, 64]]
    if im_source_pyr is None:
        im_source_pyr = pyramid(im_source, eigth_resolution=True) # [[1, 64, 512, 512],[1, 128, 128, 128],[1, 256, 64, 64]]
    c10 = im_target_pyr[-3]
    c20 = im_target_pyr[-3]
    c11 = im_target_pyr[-2]  # load_size original_res/4xoriginal_res/4
    c21 = im_source_pyr[-2]
    c12 = im_target_pyr[-1]  # load_size original_res/8xoriginal_res/8
    c22 = im_source_pyr[-1]

    # pyramid, 256 reso
    if im_target_pyr_256 is None:
        im_target_pyr_256 = pyramid(im_target_256)
    if im_source_pyr_256 is None:
        im_source_pyr_256 = pyramid(im_source_256)
    c13 = im_target_pyr_256[-2]  # load_size 256/8 x 256/8 # [1, 256, 32, 32]
    c23 = im_source_pyr_256[-2]  # load_size 256/8 x 256/8 # [1, 256, 32, 32]
    c14 = im_target_pyr_256[-1]  # load_size 256/16 x 256/16 # [1, 512, 16, 16]
    c24 = im_source_pyr_256[-1]  # load_size 256/16 x 256/16 # [1, 512, 16, 16]

    return c14, c24, c13, c23, c12, c22, c11, c21, c10, c20


def extract_features_single2(
    pyramid,
    im_source,
    im_source_256,
    im_source_pyr=None,
    im_source_pyr_256=None,
):
    # if im_target_pyr is None:
    #     im_target_pyr = pyramid(im_target, eigth_resolution=True) # [[1, 64, 512, 512],[1, 128, 128, 128],[1, 256, 64, 64]]
    if im_source_pyr is None:
        im_source_pyr = pyramid(im_source, sixteen_resolution=True) # [[1, 64, 512, 512],[1, 128, 128, 128],[1, 512, 32, 32]]
    # c10 = im_target_pyr[-3]
    c20 = im_source_pyr[-3]
    # c11 = im_target_pyr[-2]  # load_size original_res/4xoriginal_res/4
    c21 = im_source_pyr[-2]
    # c12 = im_target_pyr[-1]  # load_size original_res/8xoriginal_res/8
    c22 = im_source_pyr[-1]

    # # pyramid, 256 reso
    # # if im_target_pyr_256 is None:
    # #     im_target_pyr_256 = pyramid(im_target_256)
    # if im_source_pyr_256 is None:
    #     im_source_pyr_256 = pyramid(im_source_256) #  [24, 128, 128, 128] [24, 64, 512, 512]
    # # c13 = im_target_pyr_256[-2]  # load_size 256/8 x 256/8 # [1, 256, 32, 32]
    # c23 = im_source_pyr_256[-2]  # load_size 256/8 x 256/8 # [1, 256, 32, 32]
    # # c14 = im_target_pyr_256[-1]  # load_size 256/16 x 256/16 # [1, 512, 16, 16]
    # c24 = im_source_pyr_256[-1]  # load_size 256/16 x 256/16 # [1, 512, 16, 16]

    return c22, c21, c20




def extract_features_single(
    pyramid,
    im_source,
    im_source_256,
    im_source_pyr=None,
    im_source_pyr_256=None,
):
    # if im_target_pyr is None:
    #     im_target_pyr = pyramid(im_target, eigth_resolution=True) # [[1, 64, 512, 512],[1, 128, 128, 128],[1, 256, 64, 64]]
    if im_source_pyr is None:
        im_source_pyr = pyramid(im_source, eigth_resolution=True) # [[1, 64, 512, 512],[1, 128, 128, 128],[1, 256, 64, 64]]
    # c10 = im_target_pyr[-3]
    c20 = im_source_pyr[-3]
    # c11 = im_target_pyr[-2]  # load_size original_res/4xoriginal_res/4
    c21 = im_source_pyr[-2]
    # c12 = im_target_pyr[-1]  # load_size original_res/8xoriginal_res/8
    c22 = im_source_pyr[-1]

    # # pyramid, 256 reso
    # # if im_target_pyr_256 is None:
    # #     im_target_pyr_256 = pyramid(im_target_256)
    # if im_source_pyr_256 is None:
    #     im_source_pyr_256 = pyramid(im_source_256) #  [24, 128, 128, 128] [24, 64, 512, 512]
    # # c13 = im_target_pyr_256[-2]  # load_size 256/8 x 256/8 # [1, 256, 32, 32]
    # c23 = im_source_pyr_256[-2]  # load_size 256/8 x 256/8 # [1, 256, 32, 32]
    # # c14 = im_target_pyr_256[-1]  # load_size 256/16 x 256/16 # [1, 512, 16, 16]
    # c24 = im_source_pyr_256[-1]  # load_size 256/16 x 256/16 # [1, 512, 16, 16]

    return c22, c21, c20

@th.no_grad()
def extract_raw_features_single2(pyramid, source, source_256, feature_size=64): 
    # dit用 single 2+1是原始的，17号之后用2+2(concat) 20号改为2+1(cross attention)
    if feature_size == 64:
        c22, c21, c20 = extract_features_single(
            pyramid, source, source_256, None, None
        )# 各抽了VGG的五层特征1是target，2是source [24, 256, 64, 64] [24, 128, 128, 128] [24, 64, 512, 512]
    elif feature_size == 32:
        c22, c21, c20 = extract_features_single2(
            pyramid, source, source_256, None, None
        )# 各抽了VGG的五层特征1是target，2是source [24, 512, 32, 32] [24, 128, 128, 128] [24, 64, 512, 512]
    # src_feat = [c24, c23, c22, c21]
    # src_feat_list = []
    # corr_list = []
    # for src in src_feat: # 把多尺度的特征都统一采样到64*64
    #     src_feat_list.append(F.interpolate(src, size=(feature_size), mode='bilinear', align_corners=True))

    # for src in src_feat_list:
    #     corr = correlation(src, src)[:, None] # [1, 1, 64, 64, 64, 64] #￥#######20240923改
    #     corr_list.append(corr) # 4*[1, 1, 64, 64, 64, 64]

    # raw_corr = sum(corr_list) / len(corr_list) # [1, 1, 64, 64, 64, 64] 4个尺度互相关的平均互相关


    # c20 = F.interpolate(c20, size=(feature_size), mode='bilinear', align_corners=False) # [24, 64, 512, 512] ->[24, 64, 64, 64]
    # c21 = F.interpolate(c21, size=(feature_size), mode='bilinear', align_corners=False) # [24, 128, 128, 128]->[24, 128, 64, 64]
    # c22 = F.interpolate(c22, size=(feature_size), mode='bilinear', align_corners=False) # [24, 256, 64, 64]  ->[24, 256, 64, 64]
    # c = th.cat([c20, c21, c22], dim=1) # [b, 448, 64, 64]

    return  c22 # [24, 512, 32, 32]

@th.no_grad()
def extract_raw_features_single(pyramid, source, source_256, feature_size=64): # unet用
    c22, c21, c20 = extract_features_single(
        pyramid, source, source_256, None, None
    )# 各抽了VGG的五层特征1是target，2是source [24, 256, 64, 64] [24, 128, 128, 128] [24, 64, 512, 512]
    # src_feat = [c24, c23, c22, c21]
    # src_feat_list = []
    # corr_list = []
    # for src in src_feat: # 把多尺度的特征都统一采样到64*64
    #     src_feat_list.append(F.interpolate(src, size=(feature_size), mode='bilinear', align_corners=True))

    # for src in src_feat_list:
    #     corr = correlation(src, src)[:, None] # [1, 1, 64, 64, 64, 64] #￥#######20240923改
    #     corr_list.append(corr) # 4*[1, 1, 64, 64, 64, 64]

    # raw_corr = sum(corr_list) / len(corr_list) # [1, 1, 64, 64, 64, 64] 4个尺度互相关的平均互相关

    # c10 = F.interpolate(c10, size=(feature_size), mode='bilinear', align_corners=True)
    c20 = F.interpolate(c20, size=(feature_size), mode='bilinear', align_corners=False) # [24, 64, 64, 64]
    return  c20 # [B, 64, 64, 64]


