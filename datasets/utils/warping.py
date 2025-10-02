import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class register_model2(nn.Module):
    def __init__(self, img_size=(64, 1024, 1024), mode='bilinear'):
        super(register_model2, self).__init__()
        self.spatial_trans = SpatialTransformer2(img_size, mode)

    def forward(self, x):
        img = x[0] # [1, 3, 512, 512]
        flow = x[1] # [1, 2, 512, 512]
        out = self.spatial_trans(img, flow)
        return out


class SpatialTransformer2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # # create sampling grid
        # vectors = [torch.arange(0, s) for s in size]
        # grids = torch.meshgrid(vectors,indexing="ij")
        # grid = torch.stack(grids) # 列序优先
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor)

        # # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # # adds it to the state dict. this is annoying since everything in the state dict
        # # is included when saving weights to disk, so the model files are way bigger
        # # than they need to be. so far, there does not appear to be an elegant solution.
        # # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid) # 先wx后hy,列序优先

    def forward(self, src, flow):
        # new locations
        # new_locs = self.grid + flow #  [1, 2, 256, 256] [b, 2, 256, 256] # 广播机制
        # a = self.grid
        # new_locs = flow.clone() # 先hy后wx列序优先 [1, 2, 384, 512]
        # new_locs2 = new_locs[:, [1, 0],...]
        # shape = flow.shape[2:] # h,w
        # shape = shape[::-1] # [512, 384]
        # shape = [1024,1024]

        # need to normalize grid values to [-1, 1] for resampler
        # for i in range(len(shape)):
        #     flow[:, i, ...] = 2 * (flow[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        # if len(shape) == 2:
        flow = flow.permute(0, 2, 3, 1) # [46, 2, 256, 256]->[46, 256, 256, 2]
            # new_locs = new_locs[..., [1, 0]] # 改回行序优先
        # elif len(shape) == 3:
        #     new_locs = new_locs.permute(0, 2, 3, 4, 1)
        #     new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, flow, align_corners=True, mode=self.mode, padding_mode="zeros")



class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # # create sampling grid
        # vectors = [torch.arange(0, s) for s in size]
        # grids = torch.meshgrid(vectors,indexing="ij")
        # grid = torch.stack(grids) # 列序优先
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor)

        # # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # # adds it to the state dict. this is annoying since everything in the state dict
        # # is included when saving weights to disk, so the model files are way bigger
        # # than they need to be. so far, there does not appear to be an elegant solution.
        # # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid) # 先wx后hy,列序优先

    def forward(self, src, flow):
        # new locations
        # new_locs = self.grid + flow #  [1, 2, 256, 256] [b, 2, 256, 256] # 广播机制
        # a = self.grid
        new_locs = flow # 先hy后wx列序优先 [1, 2, 384, 512]
        # new_locs2 = new_locs[:, [1, 0],...]
        shape = flow.shape[2:] # h,w
        # shape = shape[::-1] # [512, 384]
        # shape = [1024,1024]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) # [46, 2, 256, 256]->[46, 256, 256, 2]
            # new_locs = new_locs[..., [1, 0]] # 改回行序优先
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode="zeros")









class register_model(nn.Module):
    def __init__(self, img_size=(64, 1024, 1024), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0] # [1, 3, 512, 512]
        flow = x[1] # [1, 2, 512, 512]
        out = self.spatial_trans(img, flow)
        return out
