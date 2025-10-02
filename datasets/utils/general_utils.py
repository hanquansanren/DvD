from PIL import Image
from pathlib import Path
from typing import *
from einops import rearrange

import numpy as np
import torch
import os
import shutil
import cv2

# from util.image import scale_image, random_tight_crop, tight_crop_image,random_tight_crop_imgonly
# from util.load import load_array, load_image,load_npz
# from util.mapping import scale_map, tight_crop_map, tight_crop_map_docaligner
# from inv3d_util.mask import scale_mask, tight_crop_mask
from torchvision.utils import save_image as tv_save_image
import torch.nn.functional as F


def training_init(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group('nccl', init_method='env://')
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.manual_seed_all(40)
    # create model
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    return local_rank



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_loader_withHW(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        width, height = img.size
        return img.convert('RGB'), height, width 

def cv2_loader_withHW(path):
    # with open(path, 'rb') as f:
    img = cv2.imread(path,flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    return img, height, width 



def collate_batch(batch_list):
    data1 = [item[0] for item in batch_list]
    data2 = [item[1] for item in batch_list]
    labels = [item[2] for item in batch_list]
    return data1, data2, labels

    # data1 = [item[0] for item in batch_list]
    # labels = [item[1] for item in batch_list]
    # return data1, labels
    # # features = torch.stack([sample[0] for sample in batch])  
    # # labels = torch.stack([sample[1] for sample in batch])  
    # # return features,labels   

def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region

def docreg_bm_norm(file: str, resolution: Optional[int]):
    file = Path(file)
    bm = np.load(file)[file.name[:-4]][0] # (4032, 3024, 2) numpy
    bm = ((bm+1)/2)*resolution # (0-288)
    bm = scale_map(bm, resolution) # (288, 288, 2) 
    bm = rearrange(bm, "h w c -> c h w") # (2, 288, 288) 
    bm = torch.from_numpy(bm).float() # *resolution # tensor 0-288
    return bm



def prepare_image(
    image_file: Path, mask_file: Path, color_jitter: bool, **scale_settings
):
    
    image = load_image(image_file) # (1770, 1327, 3)
    # mask = load_image(mask_file)[..., :1] # (1770, 1327, 1)
    mask = load_npz(mask_file)[..., :1].astype(np.uint8)
    mask = select_max_region(mask)
    H,W,_ = image.shape

    # test point
    # image = scale_image(image, **scale_settings) # (288, 288, 3)
    # image = rearrange(image, "h w c -> c h w")
    # image = image.astype("float32") / 255
    # image = torch.from_numpy(image)
    # flow = load_array(flow_file) # (1024, 1024, 2)
    # flow = rearrange(flow, "h w c -> c h w")
    # flow = torch.from_numpy(flow).float()
    # B2A = spatial_trans(image, flow[None], 0)
    # tv_save_image(image, "backup/test/ori.png")
    # tv_save_image(B2A[0], "backup/test/ttt.png")


    # assert image.shape[0]>0,print("input",image.shape) 
    # assert image.shape[1]>0,print("input",image.shape) 
    # image = tight_crop_image(image, mask.squeeze()) # (349, 245, 3)
    image,t,b,l,r = random_tight_crop_imgonly(mask,image,H,W)

    assert image.shape[0]>0, print("crop",image.shape) 
    assert image.shape[1]>0, print("crop",image.shape) 
    image = scale_image(image, **scale_settings) # (288, 288, 3)
    # image = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(image) if color_jitter else image
    image = rearrange(image, "h w c -> c h w")
    image = image.astype("float32") / 255
    image = torch.from_numpy(image)
    # recon = scale_image(recon, **scale_settings) # (288, 288, 3)    
    
    # img_pil = Image.fromarray(image)
    # image = transfrom(img_pil)
    
    # t,b,l,r = None,None,None,None
    return image,t,b,l,r,H,W



def prepare_masked_image(
    image_file: Path, recon_file: Path, transfrom, uv_file: Path, color_jitter: bool, **scale_settings
):
    mask = load_array(uv_file)[..., :1] # [448,448,1]
    
    image = load_image(image_file) # (448, 448, 3)
    recon = load_image(recon_file) # (448, 448, 3)
    # assert image.shape[0]>0,print("input",image.shape) 
    # assert image.shape[1]>0,print("input",image.shape) 
    # image = tight_crop_image(image, mask.squeeze()) # (349, 245, 3)
    image,recon,t,b,l,r = random_tight_crop(mask,image,recon)
    assert image.shape[0]>0, print("crop",image.shape) 
    assert image.shape[1]>0, print("crop",image.shape) 
    # image = scale_image(image, **scale_settings) # (288, 288, 3)
    # recon = scale_image(recon, **scale_settings) # (288, 288, 3)    
    # image = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(image) if color_jitter else image
    img_pil = Image.fromarray(image)
    recon_pil = Image.fromarray(recon)

    image = transfrom(img_pil)
    recon = transfrom(recon_pil)
    # image = rearrange(image, "h w c -> c h w")
    # image = image.astype("float32") / 255
    # image = torch.from_numpy(image)
    # recon = rearrange(recon, "h w c -> c h w")
    # recon = recon.astype("float32") / 255
    # recon = torch.from_numpy(recon)
    return image,recon,t,b,l,r


def prepare_bm_inv3d(file: Path, resolution: Optional[int], t,b,l,r):
    file = Path(file)
    assert file.suffix in [".npz", ".mat", ".npy"]

    bm = load_array(file).astype("float32")*1600.0 # (512, 512, 2) 0-1600

    bm = tight_crop_map(bm,t,b,l,r) #  (512, 512, 2) crop and 0-1
    bm = scale_map(bm, resolution) #   (288, 288, 2) 
    bm = np.roll(bm, shift=1, axis=-1)# 变回先x后y
    bm = rearrange(bm, "h w c -> c h w")
    bm = torch.from_numpy(bm).float()*resolution # 0-288
    # bm=(bm-0.5)*2
    return bm

def prepare_bm_docregis(file: Path, resolution: Optional[int], t,b,l,r, H,W):
    file = Path(file)
    assert file.suffix in [".npz", ".mat", ".npy"]

    bm = load_array(file).astype("float32") # numpy (512, 512, 2) (0,1)
    # bm = (bm+1)/2 # (0,1)
    bm[...,0] *= H
    bm[...,1] *= W

    bm = tight_crop_map_docaligner(bm,t,b,l,r,H,W) #  (512, 512, 2) crop and 0-1
    bm = scale_map(bm, resolution) #   (288, 288, 2) 
    bm = np.roll(bm, shift=1, axis=-1)# 变回先x后y
    bm = rearrange(bm, "h w c -> c h w") # (2, 288, 288)
    bm = torch.from_numpy(bm).float()*resolution # 0-288
    # bm=(bm-0.5)*2
    return bm






class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

    def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.n_count += n
            self.avg = self.sum / self.n_count



# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


def save_checkpoint(state, is_best, epoch, checkpoint_name, filename='checkpoint.pth.tar'):
    if os.path.exists("checkpoints/{}".format(checkpoint_name)) is False:
        os.makedirs("checkpoints/{}".format(checkpoint_name), exist_ok=True)
    torch.save(state, 'checkpoints/{}/'.format(checkpoint_name) + filename + '_latest.pth.tar')
    if epoch%10 == 0:
        # os.rename('checkpoint/' + filename + '_latest.pth.tar', 'checkpoint/' + filename + '_%d.pth.tar' % (epoch))
        shutil.copyfile('checkpoints/{}/'.format(checkpoint_name) + filename + '_latest.pth.tar', 'checkpoints/{}/'.format(checkpoint_name) + filename + '_%d.pth.tar' % (epoch))
    if is_best:
        shutil.copyfile('checkpoints/{}/'.format(checkpoint_name) + filename + '_latest.pth.tar', 'checkpoints/{}/'.format(checkpoint_name) + filename + '_best.pth.tar')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, ret=None, opt=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,ret,opt)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,ret,opt)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path,ret,opt):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
