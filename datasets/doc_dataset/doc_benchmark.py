import os
import random
from PIL import Image
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
from os.path import join as pjoin
from scipy.io import savemat
import pickle
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset

# from dkm.utils.docaligner_utils import rectification

import torch.nn.functional as F
import glob
from torchvision.utils import save_image
import cv2

from datasets.utils.general_utils import pil_loader, is_image_file, collate_batch, prepare_masked_image,\
                    prepare_bm_inv3d, prepare_image, prepare_bm_docregis,\
                    docreg_bm_norm, pil_loader_withHW, cv2_loader_withHW



# def getDatasets(dir):
# 	return os.listdir(dir)
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def getDatasets(dir):
    warp_type = os.listdir(dir)
    total = []
    for i in range(len(warp_type)):
        for j in os.listdir(pjoin(dir,warp_type[i])):
            total.append(pjoin(dir,"/"+warp_type[i],j))
    return total # ['/curved/0000.jpg', '/curved/0001.jpg']

def coords_grid_tensor(perturbed_img_shape):
    im_x, im_y = np.mgrid[0:384:complex(perturbed_img_shape[0]),
                            0:512:complex(perturbed_img_shape[1])]
    coords = np.stack((im_y,im_x), axis=2)
    coords = torch.from_numpy(coords).float().permute(2,0,1) # (384, 512, 2)
    return coords.unsqueeze(0) # [2, 384, 512]

class Doc_benchmark(Dataset):
    def __init__(
        self,data_root,input_transform) -> None:
        self.data_root = data_root
        self.input_transform = input_transform
        self.init_img_parms()
        # self.reg_model_bilin = register_model((ht,wt), 'bilinear')

    def init_img_parms(self):
        self.data_paths = os.listdir(self.data_root) # ['00001', '00002',..., '00006']


    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        return im

    def __len__(self):
        return len(self.data_paths)

    def cv2_image_load(self, sample_path, new_size=(512, 512)):
        img_ori = cv2.imread(sample_path)[:, :, ::-1].astype(np.uint8)
        img = cv2.resize(img_ori, new_size)
        return img, img_ori

    def find_jpg_in_folder(self, folder_path):
        jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        return jpg_files

    def __getitem__(self, idx): # 0,1,2,3(random sampled idx)

        sample_path = os.path.join(self.data_root,self.data_paths[idx])
        img, img_ori = self.cv2_image_load(sample_path, new_size= (512, 512))
        img = self.input_transform(img)/255.
        img_ori = self.input_transform(img_ori)
        
        data_dict = {
            "source_image": img,
            "source_image_ori": img_ori,
            "path": sample_path,
        }
        
        return data_dict



class Doc_dewarping_Data1(Dataset):
    def __init__(self, root_path, transforms=None, resolution=288, model_setting = "dewarpnet"):
        self.work_path = root_path # './triple_data'
        if transforms is not None:
            self.transforms = transforms
        self.shuffle = True
        # self.database = {} # 文件顺序排列 # {'0001_2007_000063.jpg': ['./triple_data/0001/2...000063.jpg', 0]}
        # self.label_package = {} # 再加一级标签顺序索引
        # self.triplet_db = []
        self.label_paths = []
        self.init_img_parms()
        self.loader = pil_loader
        self.resolution = resolution
        self.dataset_type = "docaligner"
        self.model_setting = model_setting
        # self.spatial_trans = SpatialTransformer((288, 288))
        # self.create_triplet_db()


    def init_img_parms(self):
        self.labels = os.listdir(self.work_path) # ['00001', '00002',..., '00006']
        for label in self.labels:
            label_path = os.path.join(self.work_path, label) # './inv3d/data/train/00001'
            self.label_paths.append(label_path) # ['./inv3d/data/train/00001',...]


    def __getitem__(self, index):
        sample_path = self.label_paths[index]
        if self.model_setting == "doctr":
            img,t,b,l,r, H,W = prepare_image(
                os.path.join(sample_path, 'warped_document.png'),os.path.join(sample_path, "warped_UV.npz"),
                # os.path.join(sample_path, 'warped_BM.npy'),
                color_jitter = True,
                # spatial_trans = self.spatial_trans, 
                resolution=self.resolution,
            ) # (3, 288, 288),0-1
            bm = prepare_bm_docregis(os.path.join(sample_path, 'warped_BM.npz'), 
                self.resolution,t,b,l,r,H,W) # (2, 288, 288) 未归一化[0,288],先x后y行序优先
            # bm = (bm/288.0-0.5)*2
            # # 备注：dewarpnet是归一化到[-1,1]做label,inv3d和doctr是[0,288]，这里遵循了inv3d和doctr的做法
        return img, bm # (3, 288, 288)，(2, 288, 288)


    def __len__(self):
        return len(self.label_paths)

