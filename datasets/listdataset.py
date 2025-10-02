import os
import os.path

import random
import cv2
import jpeg4py
import numpy as np
import torch
import torch.utils.data as data
from packaging import version
from torchvision.utils import save_image
import torch.nn.functional as F

import kornia.augmentation as KA
from datasets.utils.warping import register_model2, register_model
from datasets.util import define_mask_zero_borders
from utils_data.io import load_flo, load_gt_flow_npz, load_gt_flow_mat, augmentation


def coords_grid_tensor(perturbed_img_shape):
    im_x, im_y = np.mgrid[0:perturbed_img_shape[0]-1:complex(perturbed_img_shape[0]),
                            0:perturbed_img_shape[1]-1:complex(perturbed_img_shape[1])]
    coords = np.stack((im_y,im_x), axis=2) # 先x后y，行序优先
    # coords = torch.from_numpy(coords).float().permute(2,0,1) # (512, 512, 2)
    return coords#.unsqueeze(0) # [2, 512, 512]


def default_loader(root, path_imgs, path_flo, path_abd):
    if isinstance(path_imgs, list): # 双图配准读取双图像+label
        imgs = [os.path.join(root, path) for path in path_imgs]
        flo = os.path.join(root, path_flo)

        if imgs[0].endswith('.jpg') or imgs[0].endswith('.jpeg'):
            try:
                # b = load_flo(flo) # 绝对偏移量 # (520, 520, 2)
                # aa = jpeg4py.JPEG(imgs[0]) # (520, 520, 3)
                # a = [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs] 
                return [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs], load_flo(flo)
            except:
                return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo)
        else:
            return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_gt_flow_npz(flo)
    else: # 单图矫正读取图像+label
        img1 = cv2.imread(path_imgs, 1)[:, :, ::-1].astype(np.uint8)
        abd = cv2.imread(path_abd, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # print(abd.dtype)
        # print(abd.shape)
        _, mask = cv2.threshold(abd, 1, 255, cv2.THRESH_BINARY)
        return cv2.resize(img1, (512, 512)), load_gt_flow_npz(path_flo), cv2.resize(mask, (512, 512))

def default_loader2(root, path_imgs, path_flo, path_abd):
    if isinstance(path_imgs, list): # 双图配准读取双图像+label
        imgs = [os.path.join(root, path) for path in path_imgs]
        flo = os.path.join(root, path_flo)

        if imgs[0].endswith('.jpg') or imgs[0].endswith('.jpeg'):
            try:
                # b = load_flo(flo) # 绝对偏移量 # (520, 520, 2)
                # aa = jpeg4py.JPEG(imgs[0]) # (520, 520, 3)
                # a = [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs] 
                return [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs], load_flo(flo)
            except:
                return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo)
        else:
            return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_gt_flow_npz(flo)
    else: # 单图矫正读取图像+label
        img1 = cv2.imread(path_imgs, 1)[:, :, ::-1].astype(np.uint8)
        abd = cv2.imread(path_abd, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # print(abd.dtype)
        # print(abd.shape)
        _, mask = cv2.threshold(abd, 1, 255, cv2.THRESH_BINARY)
        return cv2.resize(img1, (512, 512)), load_gt_flow_npz(path_flo), cv2.resize(mask, (512, 512))

def default_loader3(root, path_imgs, path_flo, path_abd):
    if isinstance(path_imgs, list): # 双图配准读取双图像+label
        imgs = [os.path.join(root, path) for path in path_imgs]
        flo = os.path.join(root, path_flo)

        if imgs[0].endswith('.jpg') or imgs[0].endswith('.jpeg'):
            try:
                # b = load_flo(flo) # 绝对偏移量 # (520, 520, 2)
                # aa = jpeg4py.JPEG(imgs[0]) # (520, 520, 3)
                # a = [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs] 
                return [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs], load_flo(flo)
            except:
                return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo)
        else:
            return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_gt_flow_npz(flo)
    else: # 单图矫正读取图像+label
        img1 = cv2.imread(path_imgs, 1)[:, :, ::-1].astype(np.uint8)
        abd = cv2.imread(path_abd, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # print(abd.dtype)
        # print(abd.shape)
        _, mask = cv2.threshold(abd, 1, 255, cv2.THRESH_BINARY)
        return cv2.resize(img1, (512, 512)), load_gt_flow_mat(path_flo), cv2.resize(mask, (512, 512))

def linear_interpolation(f0, fT, t, T):
    """
    Perform linear interpolation between f0 and fT.

    Parameters:
        f0 (numpy.ndarray): Regular grid coordinates of shape (448, 448, 2).
        fT (numpy.ndarray): Actual object coordinates of shape (448, 448, 2).
        t (float): Current interpolation step (0 <= t <= T).
        T (float): Total number of steps.

    Returns:
        numpy.ndarray: Interpolated coordinates of shape (448, 448, 2).
    """
    if not (0 <= t <= T):
        raise ValueError("t must be in the range [0, T].")

    # Linear interpolation formula
    ft = (t / T) * fT + (1 - t / T) * f0
    return ft





class ListDataset(data.Dataset):
    """General Dataset creation class"""
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=default_loader, load_valid_mask=False, load_size=False,
                 load_occlusion_mask=False, get_mapping=False, compute_mask_zero_borders=False):
        """

        Args:
            root: root directory containing image pairs and flow folders
            path_list: path to csv files with ground-truth information
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            loader: image and flow loader type
            load_valid_mask: is the loader outputting a valid mask ?
            load_size: is the loader outputting load_size of original source image ?
            load_occlusion_mask: is the loader outputting a ground-truth occlusion mask ?
            get_mapping: get mapping ?
            compute_mask_zero_borders: output mask of zero borders ?
        Output in __getitem__:
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                load_occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the target
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """
        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.load_valid_mask = load_valid_mask
        self.load_size = load_size
        self.load_occlusion_mask = load_occlusion_mask
        self.get_mapping = get_mapping
        self.mask_zero_borders = compute_mask_zero_borders
        self.reg_model_bilin = register_model((512,512), 'bilinear')
        # self.get_doc_mask = get_doc_mask

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns: dictionary with fieldnames
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the flow
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the flow image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in flow coordinate system, relating flow to source image
            else:
                flow_map: flow fields in flow coordinate system, relating flow to source image
        """
        # for all inputs[0] must be the source and inputs[1] must be the flow
        inputs_paths, flow_path, abd_path = self.path_list[index]

        if not self.load_valid_mask: # true
            if self.load_size: # false
                inputs, flow, mask = self.loader(self.root, inputs_paths, flow_path, abd_path)
            else:
                inputs, bm, mask = self.loader(self.root, inputs_paths, flow_path, abd_path) #(512, 512, 3) (448, 448, 2) (512,512) # 先x后y，行序优先
                mask = mask.reshape(512,512,1)
                # cv2.imwrite("vis_hp/debug_vis/doc.png", inputs)
                # cv2.imwrite("vis_hp/debug_vis/mask.png", mask)
                if isinstance(inputs, list):
                    pass
                else:
                    inputs = [inputs, None]
                # source_size = inputs[0].shape
            # if self.co_transform is not None: # false
            #     inputs, flow = self.co_transform(inputs, flow)
            
            # # test point 1
            # im_A = torch.from_numpy(inputs[0]/255.0).permute(2,0,1).float() # [3,512,512] CHW tensor
            # bm_vis = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0)*(512/448) # [1, 2, 512, 512] BCHW tensor 0-512 先x后y 行序优先
            # bm_vis = F.interpolate(bm_vis, size=512, mode='bilinear', align_corners=True)  # [1, 2, 512, 512]
            # inv_a0 = self.reg_model_bilin([im_A[None,...], bm_vis])[0]
            # save_image(im_A,"vis_hp/debug_vis/OO.png")
            # save_image(inv_a0,"vis_hp/debug_vis/inv_a00.png")
            # print("aaa")
            
            base = coords_grid_tensor((448,448)) # [448, 448, 2] 先x后y 行序优先
            flow = bm - base # 绝对偏移量 0-±448
            # cmask = get_gt_correspondence_mask(flow)
            # # test point 2
            # cv2.imwrite("vis_hp/debug_vis/mask.png", 255*mask.astype(np.uint8).reshape(512,512,1)) # 全白就是正常的
        else: # false
            if self.load_occlusion_mask:
                if self.load_size:
                    inputs, flow, mask, occ_mask, source_size = self.loader(self.root, inputs_paths, flow_path,
                                                                            return_occlusion_mask=True)
                else:
                    # loader comes with a mask of valid correspondences
                    inputs, flow, mask, occ_mask = self.loader(self.root, inputs_paths, flow_path,
                                                               return_occlusion_mask=True)
                    source_size = inputs[0].shape
            else:
                if self.load_size:
                    inputs, flow, mask, source_size = self.loader(self.root, inputs_paths, flow_path)
                else:
                    # loader comes with a mask of valid correspondences
                    inputs, flow, mask = self.loader(self.root, inputs_paths, flow_path)
                    source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, flow, mask = self.co_transform(inputs, flow, mask)

        # if self.mask_zero_borders: # false
        #     mask_valid = define_mask_zero_borders(np.array(inputs[1]))

        # after co transform that could be reshapping the flow
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
            mask = self.source_image_transform(mask)
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            flow = self.flow_transform(flow)
            flow = F.interpolate((flow/448*512).unsqueeze(0), size=512, mode='bilinear', align_corners=True)[0]
            # 绝对偏移量 0-±512
        output = {'source_image': inputs[0]/255.,
                  'doc_mask': mask,
                #   'target_image': inputs[1],
                #   'correspondence_mask': cmask.astype(bool) if \
                #         version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8),
                #   'source_image_size': source_size,
                #   'sparse': False
                  }
        
        
        # if self.load_occlusion_mask: # false
        #     output['occlusion_mask'] = occ_mask

        # if self.mask_zero_borders: # false
        #     output['mask_zero_borders'] = mask_valid.astype(bool) if \
        #         version.parse(torch.__version__) >= version.parse("1.1") else mask_valid.astype(np.uint8)

        # if self.get_mapping: # false
        #     output['correspondence_map'] = convert_flow_to_mapping(flow)
        # else:
        output['flow_map'] = flow
        return output

    def __len__(self):
        return len(self.path_list)




class Aug_ListDataset(data.Dataset):
    """General Dataset creation class"""
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=default_loader2, load_valid_mask=False, load_size=False,
                 load_occlusion_mask=False, get_mapping=False, compute_mask_zero_borders=False, augmentation=augmentation,
                 cj= KA.ColorJitter(0.1, 0.1, 0.1, 0.1)):
        """

        Args:
            root: root directory containing image pairs and flow folders
            path_list: path to csv files with ground-truth information
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            loader: image and flow loader type
            load_valid_mask: is the loader outputting a valid mask ?
            load_size: is the loader outputting load_size of original source image ?
            load_occlusion_mask: is the loader outputting a ground-truth occlusion mask ?
            get_mapping: get mapping ?
            compute_mask_zero_borders: output mask of zero borders ?
        Output in __getitem__:
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                load_occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the target
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """
        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.augmentation = augmentation
        self.load_valid_mask = load_valid_mask
        self.load_size = load_size
        self.load_occlusion_mask = load_occlusion_mask
        self.get_mapping = get_mapping
        self.mask_zero_borders = compute_mask_zero_borders
        self.reg_model_bilin = register_model2((1024,1024), 'bilinear')
        self.base = coords_grid_tensor((512,512)) # [448, 448, 2] 先x后y 行序优先
        # self.get_doc_mask = get_doc_mask
        self.txpths=[]
        self.workpath = '/home/phd23_weiguang_zhang/DiffMatch'
        self.cj = cj
        with open(os.path.join('/home/phd23_weiguang_zhang/DiffMatch/checkpoints/backup/augtexnames.txt'),'r') as f:
            for line in f:
                txpth=line.strip()
                self.txpths.append(txpth)

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns: dictionary with fieldnames
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the flow
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the flow image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in flow coordinate system, relating flow to source image
            else:
                flow_map: flow fields in flow coordinate system, relating flow to source image
        """
        # for all inputs[0] must be the source and inputs[1] must be the flow
        inputs_paths, flow_path, abd_path = self.path_list[index]

        if not self.load_valid_mask: # true
            if self.load_size: # false
                inputs, flow, mask = self.loader(self.root, inputs_paths, flow_path, abd_path)
            else:
                inputs, bm, mask = self.loader(self.root, inputs_paths, flow_path, abd_path) #(512, 512, 3) (448, 448, 2) from 0~448 (512,512) # 先x后y，行序优先
                mask = mask.reshape(512,512,1)
                # cv2.imwrite("vis_hp/debug_vis/doc.png", inputs)
                # cv2.imwrite("vis_hp/debug_vis/mask.png", mask)
                
                tex_id=random.randint(0,len(self.txpths)-1)
                txpth=self.txpths[tex_id] 
                tex=cv2.imread(os.path.join(self.workpath,'checkpoints/backup',txpth)).astype(np.uint8)
                tex=tex[:, :, ::-1]
                bg=cv2.resize(tex,(512, 512),interpolation=cv2.INTER_NEAREST) # [512,512,3]
                # cv2.imwrite("vis_hp/debug_vis/tex.png", bg)
                
                inputs, mask, bm  = self.augmentation(inputs, mask, bm, bg)
                # cv2.imwrite("vis_hp/debug_vis/doc2.png", inputs)
                mask = mask.reshape(512,512,1)
                # cv2.imwrite("vis_hp/debug_vis/mask2.png", mask*255.)
            
        # t = random.randint(0, 14)
        t = 0
        # print(t)
        bm_inter = linear_interpolation(self.base , bm, t, 20).astype(np.float32) # (448, 448, 2) 0-448 绝对bm

        # backup
        # inputs[0] = torch.from_numpy(inputs[0]/255.0).permute(2,0,1).float() # [3,1024,1024] CHW tensor
        # mask = torch.from_numpy(mask).permute(2,0,1).float() # [1,1024,1024] CHW tensor
        # bm_vis = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0).float()*(1024/448) # [1, 2, 512, 512] BCHW tensor 0-512 先x后y 行序优先
        # bm_vis = F.interpolate(bm_vis, size=1024, mode='bilinear', align_corners=True)  # [1, 2, 512, 512]
        # # inv_a0 = self.reg_model_bilin([im_A[None,...], bm_vis])[0]
        
        # bm_inter_vis = torch.from_numpy(bm_inter.transpose(2,0,1)).unsqueeze(0).float()*(1024/448) # [1, 2, 448, 448] BCHW tensor 0-512 先x后y 行序优先
        # bm_inter_vis = F.interpolate(bm_inter_vis, size=1024, mode='bilinear', align_corners=True)  # [1, 2, 1024, 1024]
        # inputs[0] = self.reg_model_bilin([inputs[0][None,...], bm_inter_vis])
        # mask = self.reg_model_bilin([mask[None,...], bm_inter_vis])
        # inputs[0] = F.interpolate(inputs[0], size=512, mode='bilinear', align_corners=True)
        # mask = F.interpolate(mask, size=512, mode='bilinear', align_corners=True)
        # backup


        inputs = torch.from_numpy(inputs).permute(2,0,1).float() # [3,1024,1024] CHW tensor
        mask = torch.from_numpy(mask).permute(2,0,1).float() # [1,1024,1024] CHW tensor
        # bm_vis = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0).float()*(1024/448) # [1, 2, 512, 512] BCHW tensor 0-512 先x后y 行序优先
        # bm_vis = F.interpolate(bm_vis, size=1024, mode='bilinear', align_corners=True)  # [1, 2, 512, 512]
        # inv_a0 = self.reg_model_bilin([im_A[None,...], bm_vis])[0]
        
        # # test point1
        # bm_vis = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0).float() # [1, 2, 512, 512] BCHW tensor 0-512 先x后y 行序优先
        # # bm_vis = F.interpolate(bm_vis, size=512, mode='bilinear', align_corners=True)  # [1, 2, 512, 512]
        # for i in range(len(bm_vis.shape[2:])):
        #     bm_vis[:, i, ...] = 2 * (bm_vis[:, i, ...] / (bm_vis.shape[2:][i] - 1) - 0.5)
        # inv_a0 = self.reg_model_bilin([inputs[None,...]/255., bm_vis])[0]
        # save_image(inv_a0,"vis_hp/debug_vis/inv_a.png")
        
        
        
        bm_inter_vis = torch.from_numpy(bm_inter.transpose(2,0,1)).unsqueeze(0).float().clone()#*(511/447) # [1, 2, 448, 448] BCHW tensor 0-512 先x后y 行序优先
        # bm_inter_vis = F.interpolate(bm_inter_vis, size=512, mode='bilinear', align_corners=True)  # [1, 2, 1024, 1024]
        shape = bm_inter_vis.shape[2:] # h,w
        for i in range(len(shape)):
            bm_inter_vis[:, i, ...] = 2 * (bm_inter_vis[:, i, ...] / (shape[i] - 1) - 0.5)
        inputs_inter = self.reg_model_bilin([inputs[None,...], bm_inter_vis])
        mask_inter = self.reg_model_bilin([mask[None,...], bm_inter_vis])
        inputs_inter = self.cj(inputs_inter/255.)
        # inputs_inter = F.interpolate(inputs_inter, size=512, mode='bilinear', align_corners=True)
        # mask_inter = F.interpolate(mask_inter, size=512, mode='bilinear', align_corners=True)
        
        # save_image(inputs,"vis_hp/debug_vis/input.png")
        # save_image(inputs_inter,"vis_hp/debug_vis/input_inter.png")
        # save_image(mask_inter,"vis_hp/debug_vis/mask.png")
        # print("aaa")
        
        # bm_new = self.reg_model_bilin([bm_inter_vis, bm_vis])
        # inv_a1 = self.reg_model_bilin([inputs[0], bm_new])[0]
        # # save_image(im_A,"vis_hp/debug_vis/o.png")
        # # save_image(inv_a0,"vis_hp/debug_vis/inv_a.png")
        # # save_image(inputs[0],"vis_hp/debug_vis/a1.png")
        # save_image(inv_a1, "vis_hp/debug_vis/inv_a1.png")
        # print("aaa")
        
        ##############################################################################
        flow = bm - self.base # 绝对偏移量 0-±448 f_gt
        flow_inter = bm_inter - self.base # 绝对偏移量 0-±448 f_inter
        

        # if self.source_image_transform is not None:
            # inputs[0] = self.source_image_transform(inputs[0])
            # mask = self.source_image_transform(mask)
        if self.flow_transform is not None:
            # 绝对偏移量 0-±512
            flow_inter = self.flow_transform(flow_inter)
            # flow_inter = F.interpolate((flow_inter/447*511).unsqueeze(0), size=512, mode='bilinear', align_corners=True)[0]
            # 绝对偏移量 0-±512
            flow = self.flow_transform(flow)
            # flow = F.interpolate((flow/447*511).unsqueeze(0), size=512, mode='bilinear', align_corners=True)[0]
        output = {'source_image': inputs_inter[0],
                  'doc_mask': mask_inter[0],
                #   'source_image_0':inputs
                  }
        
        
        # if self.load_occlusion_mask: # false
        #     output['occlusion_mask'] = occ_mask

        # if self.mask_zero_borders: # false
        #     output['mask_zero_borders'] = mask_valid.astype(bool) if \
        #         version.parse(torch.__version__) >= version.parse("1.1") else mask_valid.astype(np.uint8)

        # if self.get_mapping: # false
        #     output['correspondence_map'] = convert_flow_to_mapping(flow)
        # else:
        output['flow_map'] = flow # f_gt
        output['flow_map_inter'] = flow_inter # f_inter
        return output

    def __len__(self):
        return len(self.path_list)
    


class Aug_Doc3d_ListDataset(data.Dataset):
    """General Dataset creation class"""
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=default_loader3, load_valid_mask=False, load_size=False,
                 load_occlusion_mask=False, get_mapping=False, compute_mask_zero_borders=False, augmentation=augmentation,
                 cj= KA.ColorJitter(0.1, 0.1, 0.1, 0.1) ):
        """

        Args:
            root: root directory containing image pairs and flow folders
            path_list: path to csv files with ground-truth information
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            loader: image and flow loader type
            load_valid_mask: is the loader outputting a valid mask ?
            load_size: is the loader outputting load_size of original source image ?
            load_occlusion_mask: is the loader outputting a ground-truth occlusion mask ?
            get_mapping: get mapping ?
            compute_mask_zero_borders: output mask of zero borders ?
        Output in __getitem__:
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                load_occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the target
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """
        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.augmentation = augmentation
        self.load_valid_mask = load_valid_mask
        self.load_size = load_size
        self.load_occlusion_mask = load_occlusion_mask
        self.get_mapping = get_mapping
        self.mask_zero_borders = compute_mask_zero_borders
        self.reg_model_bilin = register_model2((1024,1024), 'bilinear')
        self.base = coords_grid_tensor((512,512))
        # self.get_doc_mask = get_doc_mask
        self.txpths=[]
        self.workpath = '/home/phd23_weiguang_zhang/DiffMatch'
        self.cj = cj
        with open(os.path.join('/home/phd23_weiguang_zhang/DiffMatch/checkpoints/backup/augtexnames.txt'),'r') as f:
            for line in f:
                txpth=line.strip()
                self.txpths.append(txpth)

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns: dictionary with fieldnames
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the flow
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the flow image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in flow coordinate system, relating flow to source image
            else:
                flow_map: flow fields in flow coordinate system, relating flow to source image
        """
        # for all inputs[0] must be the source and inputs[1] must be the flow
        inputs_paths, flow_path, abd_path = self.path_list[index]

        if not self.load_valid_mask: # true
            if self.load_size: # false
                inputs, flow, mask = self.loader(self.root, inputs_paths, flow_path, abd_path)
            else:
                inputs, bm, mask = self.loader(self.root, inputs_paths, flow_path, abd_path) #(512, 512, 3), (448, 448, 2) from 0~448, (512,512) # 先x后y，行序优先
                mask = mask.reshape(512,512,1)
                # cv2.imwrite("vis_hp/debug_vis/doc.png", inputs)
                # cv2.imwrite("vis_hp/debug_vis/mask1.png", mask)
                
                tex_id=random.randint(0,len(self.txpths)-1)
                txpth=self.txpths[tex_id] 
                tex=cv2.imread(os.path.join(self.workpath,'checkpoints/backup',txpth)).astype(np.uint8)
                tex=tex[:, :, ::-1]
                bg=cv2.resize(tex,(512, 512),interpolation=cv2.INTER_NEAREST) # [512,512,3]
                # cv2.imwrite("vis_hp/debug_vis/tex.png", bg)
                
                inputs, mask, bm  = self.augmentation(inputs, mask, bm, bg)
                # cv2.imwrite("vis_hp/debug_vis/doc2.png", inputs)
                mask = mask.reshape(512,512,1)
                # cv2.imwrite("vis_hp/debug_vis/mask2.png", mask*255.)

            # [448, 448, 2] 先x后y 行序优先
            # arr = [0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12]
            # # arr = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            # t = random.choice(arr)
            t=0
            # t = 13# random.randint(0, 14)
            bm_inter = linear_interpolation(self.base, bm, t, 20).astype(np.float32) # (512, 512, 2) 0-512 绝对bm

            inputs = torch.from_numpy(inputs).permute(2,0,1).float() # [3,1024,1024] CHW tensor
            mask = torch.from_numpy(mask).permute(2,0,1).float() # [1,1024,1024] CHW tensor
            
            # # test point1
            # bm_vis = torch.from_numpy(bm.transpose(2,0,1)).unsqueeze(0).float() # [1, 2, 512, 512] BCHW tensor 0-512 先x后y 行序优先
            # # bm_vis = F.interpolate(bm_vis, size=512, mode='bilinear', align_corners=True)  # [1, 2, 512, 512]
            # for i in range(len(bm_vis.shape[2:])):
            #     bm_vis[:, i, ...] = 2 * (bm_vis[:, i, ...] / (bm_vis.shape[2:][i] - 1) - 0.5)
            # inv_a0 = self.reg_model_bilin([inputs[None,...]/255., bm_vis])[0]
            # save_image(inv_a0,"vis_hp/debug_vis/inv_a.png")
            
            bm_inter_vis = torch.from_numpy(bm_inter.transpose(2,0,1)).unsqueeze(0).float().clone() # [1, 2, 512, 512] BCHW tensor 0-512 先x后y 行序优先
            # bm_inter_vis = F.interpolate(bm_inter_vis, size=512, mode='bilinear', align_corners=True)  # [1, 2, 1024, 1024]
            shape = bm_inter_vis.shape[2:] # h,w
            for i in range(len(shape)):
                bm_inter_vis[:, i, ...] = 2 * (bm_inter_vis[:, i, ...] / (shape[i] - 1) - 0.5)
            inputs_inter = self.reg_model_bilin([inputs[None,...], bm_inter_vis])
            mask_inter = self.reg_model_bilin([mask[None,...], bm_inter_vis])
            inputs_inter = self.cj(inputs_inter/255.)
            # inputs_inter = F.interpolate(inputs_inter, size=512, mode='bilinear', align_corners=True)
            # mask_inter = F.interpolate(mask_inter, size=512, mode='bilinear', align_corners=True)
            # mask_inter = (mask_inter > 0.3).float()
            
            # test point 2
            # save_image(inputs/255.,"vis_hp/debug_vis/input.png")
            # save_image(inputs_inter,"vis_hp/debug_vis/input_inter77.png")
            # save_image(mask_inter,"vis_hp/debug_vis/mask.png")
            # test_img = inputs_inter*mask_inter
            # test_img_inv = inputs_inter*(1-mask_inter)
            # save_image(test_img,"vis_hp/debug_vis/test.png")
            # save_image(test_img_inv,"vis_hp/debug_vis/test_inv.png")
            # print("aaa")
            
            # bm_new = self.reg_model_bilin([bm_inter_vis, bm_vis])
            # inv_a1 = self.reg_model_bilin([inputs[0], bm_new])[0]
            # # save_image(im_A,"vis_hp/debug_vis/o.png")
            # # save_image(inv_a0,"vis_hp/debug_vis/inv_a.png")
            # # save_image(inputs[0],"vis_hp/debug_vis/a1.png")
            # save_image(inv_a1, "vis_hp/debug_vis/inv_a1.png")
            # print("aaa")
            
            ##############################################################################
            flow = bm - self.base # 绝对偏移量 0-±512 f_gt
            flow_inter = bm_inter - self.base # 绝对偏移量 0-±512 f_inter
        

        # if self.source_image_transform is not None:
            # inputs[0] = self.source_image_transform(inputs[0])
            # mask = self.source_image_transform(mask)
        if self.flow_transform is not None:
            # 绝对偏移量 0-±512
            flow_inter = self.flow_transform(flow_inter)
            # flow_inter = F.interpolate((flow_inter).unsqueeze(0), size=512, mode='bilinear', align_corners=True)[0]
            # 绝对偏移量 0-±512
            flow = self.flow_transform(flow)
            # flow = F.interpolate((flow).unsqueeze(0), size=512, mode='bilinear', align_corners=True)[0]
        output = {'source_image': inputs_inter[0],
                  'doc_mask': mask_inter[0],
                #   'source_image_0':inputs
                  }
        
        
        # if self.load_occlusion_mask: # false
        #     output['occlusion_mask'] = occ_mask

        # if self.mask_zero_borders: # false
        #     output['mask_zero_borders'] = mask_valid.astype(bool) if \
        #         version.parse(torch.__version__) >= version.parse("1.1") else mask_valid.astype(np.uint8)

        # if self.get_mapping: # false
        #     output['correspondence_map'] = convert_flow_to_mapping(flow)
        # else:
        output['flow_map'] = flow # f_gt
        output['flow_map_inter'] = flow_inter # f_inter
        return output

    def __len__(self):
        return len(self.path_list)





