#!/usr/bin/env python3.7

import os
import re
import sys
import uuid

import imageio
import numpy as np
import h5py
import cv2
# from scipy.io import loadmat
# import hdf5storage as h5
import torch
import torch.nn.functional as F
import random

# import kornia.augmentation as KA
# import kornia.geometry.transform as KG

def tight_crop(img, mask, bm): # [512,512,3]unit8 [512,512]unit8 [448,448,2] float64
    # msk=((img[:,:,0]!=0)&(img[:,:,1]!=0)&(img[:,:,2]!=0)).astype(np.uint8)
    size=mask.shape
    [y, x] = (mask[:,:,0]).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    img = img[miny : maxy + 1, minx : maxx + 1, :]
    mask = mask[miny : maxy + 1, minx : maxx + 1, :]
    
    # hw_rate = (maxy-miny)/(maxx-minx) # 不需要考虑长宽比，因为测试时都是裁剪好的图片
    
    s = 45
    img = np.pad(img, ((s, s), (s, s), (0, 0)), 'constant')
    mask = np.pad(mask, ((s, s), (s, s), (0, 0)), 'constant')
    cx1 = random.randint(5, s - 5)
    cx2 = random.randint(5, s - 5) + 1
    cy1 = random.randint(5, s - 5)
    cy2 = random.randint(5, s - 5) + 1

    img = img[cy1 : -cy2, cx1 : -cx2, :]
    mask = mask[cy1 : -cy2, cx1 : -cx2, :]
    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2
    
    
    bm[:,:,1]=bm[:,:,1]-t
    bm[:,:,0]=bm[:,:,0]-l
    bm=511*bm/np.array([511.0-l-r, 511.0-t-b]) # 0~1
    # bm0=cv2.resize(bm[:,:,0],(512,512))
    # bm1=cv2.resize(bm[:,:,1],(512,512))
    # bm=np.stack([bm0,bm1],axis=-1)
    
    return img, mask, bm

# 这是一个用于裁剪图片的函数，图片中间是一个拍照文档，现有的函数
# 因为使用了“img[miny : maxy + 1, minx : maxx + 1, :]” 背景被过度裁剪了，我想在裁剪后保留完整的背景，如何修改函数
def tight_crop_new(img, mask, bm): 
    # img [512,512,3]unit8 
    # mask [512,512]unit8 
    # bm [448,448,2] float64

    size = mask.shape
    [y, x] = (mask[:, :, 0]).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    # # 为了保留背景，直接操作原图，不裁剪图像尺寸
    # new_img = img.copy()
    # new_mask = mask.copy()

    # 随机添加边界内偏移（确保不超出图像边界）
    offset = 25
    cx1 = random.randint(5, offset)
    cx2 = random.randint(5, offset)
    cy1 = random.randint(5, offset)
    cy2 = random.randint(5, offset)

    # 调整裁剪范围并保持图像背景完整
    final_minx = max(0, minx - cx1)
    final_maxx = min(size[1], maxx + cx2)
    final_miny = max(0, miny - cy1)
    final_maxy = min(size[0], maxy + cy2)

    # 裁剪出包含文档的区域，但保留背景尺寸
    cropped_img = img[final_miny:final_maxy, final_minx:final_maxx, :]
    cropped_mask = mask[final_miny:final_maxy, final_minx:final_maxx, :]

    # 更新 bm 的坐标
    t = final_miny
    b = size[0] - final_maxy
    l = final_minx
    r = size[1] - final_maxx

    bm[:, :, 1] = bm[:, :, 1] - t
    bm[:, :, 0] = bm[:, :, 0] - l
    bm = 511 * bm / np.array([511.0 - l - r, 511.0 - t - b]) # 0~1

    return cropped_img, cropped_mask/255., bm


def augmentation(img, mask, bm, bg=None):  # [512,512,3]unit8 [512,512,1]unit8 [448,448,2] float64 [512,512,3] unit8
    # tight crop
    img, mask, bm = tight_crop_new(img, mask, bm) 
    # replace bg
    [fh, fw, _] = img.shape
    chance=random.random()
    # chance = 0.25
    if chance > 0.3:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1)) # (600, 600, 3)
        bg = bg[: fh, : fw, :]
        msk = mask
    elif chance < 0.3 and chance> 0.2:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
        msk = mask
        # cv2.imwrite("vis_hp/debug_vis/tex2.png", bg)
    else:
        bg=np.zeros((fh, fw, 3))
        msk=np.ones((fh, fw, 3))
    img = bg * (1 - msk) + img * msk
    # cv2.imwrite("vis_hp/debug_vis/replace.png", img)
    mask = cv2.resize(mask, (512, 512))
    img = cv2.resize(img, (512, 512))
    # msk=((bm[:,:,0]!=0)&(bm[:,:,1]!=0)&(bm[:,:,2]!=0)).astype(np.uint8)    
    return img, mask, bm 

    



# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def read(file):
    if file.endswith('.float3'): return readFloat(file)
    elif file.endswith('.flo'): return readFlow(file)
    elif file.endswith('.ppm'): return readImage(file)
    elif file.endswith('.pgm'): return readImage(file)
    elif file.endswith('.png'): return readImage(file)
    elif file.endswith('.jpg'): return readImage(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)


def write(file, data):
    if file.endswith('.float3'): return writeFloat(file, data)
    elif file.endswith('.flo'): return writeFlow(file, data)
    elif file.endswith('.ppm'): return writeImage(file, data)
    elif file.endswith('.pgm'): return writeImage(file, data)
    elif file.endswith('.png'): return writeImage(file, data)
    elif file.endswith('.jpg'): return writeImage(file, data)
    elif file.endswith('.pfm'): return writePFM(file, data)
    else: raise Exception('don\'t know how to write %s' % file)

def load_gt_flow_npz(bm_path):
    # # bm = np.transpose(h5py.File(bm_path,'r',libver='latest', swmr=True)["bm"])
    # try:
    #     bm = h5.loadmat(bm_path)['bm'] # (1024, 1024, 2) from 0~1024
    # except:
    #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #     print(bm_path)
    # bm = (bm/np.array([1024,1024])).astype(np.float32) # (1024, 1024, 2) from 0~1
    # bm[:,:,0] = bm[:,:,0]*512 # (1024, 1024, 2) from 0~512
    # bm[:,:,1] = bm[:,:,1]*384
    # bm = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0) # (1,2,384,512)
    # bm = F.interpolate(bm,size=(384,512),mode='bilinear',
    #                    align_corners=True)   # (1,2,384,512)
    try:
        bm = np.load(bm_path)['warped_BM'][:447,:447,:]*511 + 0.4# (448, 448, 2) range[0-1] # 先y后x，行序优先
        # bm[:,:,0] = bm[:,:,0]*447 # (448, 448, 2) from 0~448
        # bm[:,:,1] = bm[:,:,1]*447
        bm0=cv2.resize(bm[:,:,0],(512,512))
        bm1=cv2.resize(bm[:,:,1],(512,512))
        bm=np.stack([bm0,bm1],axis=-1)    
        bm = np.roll(bm, shift=1, axis=-1) # # 先x后y，行序优先, 绝对位置bm
        # bm = bm.transpose((2,0,1)) 
    except:
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(bm_path)
    # bm = (bm/np.array([1024,1024])).astype(np.float32)  # (1024, 1024, 2) from 0~1
    # bm[:,:,0] = bm[:,:,0]*520 # (1024, 1024, 2) from 0~512
    # bm[:,:,1] = bm[:,:,1]*520
    # bm = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0) # [1, 2, 1024, 1024]
    # bm = F.interpolate(bm,size=(384,512),mode='bilinear',
    #                     align_corners=True)   # (1,2,384,512)
    return bm

def load_gt_flow_mat(bm_path):
    try:
        # bm = h5.loadmat(bm_path)['bm']# (448, 448, 2) range[0-1] # 先y后x，行序优先
        with h5py.File(bm_path, 'r') as f:
            bm = f['bm'][:].transpose((2,1,0))[:447,:447,:]*(511/447) - 1.2 # (447, 447, 2)
        bm0=cv2.resize(bm[:,:,0],(512,512))
        bm1=cv2.resize(bm[:,:,1],(512,512))
        bm=np.stack([bm0,bm1],axis=-1)    
        # bm[:,:,0] = bm[:,:,0]*448 # (448, 448, 2) from 0~448
        # bm[:,:,1] = bm[:,:,1]*448
        # bm = np.roll(bm, shift=1, axis=-1)
    except:
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(bm_path)
    return bm # 先x后y，行序优先, 绝对位置bm (448, 448, 2) from 0~448




def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def readFlow(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D.astype(np.float32)


def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data

    return imageio.imread(name)


def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)

    return imageio.imwrite(name, data)


def writeFlow(flow, name_to_save, save_dir):
    name=os.path.join(save_dir, name_to_save)
    f = open(name, 'wb')
    magic=202021.25
    np.array([magic], dtype=np.float32).tofile(f)
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def writeMask(mask, name_to_save, save_dir):
    name = os.path.join(save_dir, name_to_save)
    mask = mask.astype(np.uint8)
    if mask.max() != 255:
        mask *= 255
    imageio.imwrite(name, mask.astype(np.uint8))


def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)
