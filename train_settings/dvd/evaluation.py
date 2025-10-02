import os

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from train_settings.dvd.feature_backbones.VGG_features import VGGPyramid

from utils_flow.visualization_utils import visualize, visualize_dewarping

from .eval_utils import extract_raw_features_single, extract_raw_features_single2
from .improved_diffusion import dist_util
from .improved_diffusion.gaussian_diffusion import GaussianDiffusion
import torch
from torchvision.utils import save_image as tv_save_image

def prepare_data(settings, batch_preprocessing, SIZE, data):
    if 'source_image_ori' in data:
        source_vis = data['source_image_ori']  # B, C, 512, 512 torch.uint8 cpu
    else:
        source_vis = data['source_image']
    if 'target_image' in data:
        target_vis = data['target_image']
    else:
        target_vis = None
    _, _, H_ori, W_ori = source_vis.shape

    # data = batch_preprocessing(data)

    source = data['source_image'].to(dist_util.dev())  # [1, 3, 914, 1380]  torch.float32
    if 'source_image_0' in data:
        source_0 = data['source_image_0'].to(dist_util.dev())
    else:
        source_0 = None
    if 'target_image' in data:  
        target = data['target_image']  # [1, 3, 914, 1380]  torch.float32
    else:
        target = None 
    if 'flow_map' in data:    
        batch_ori = data['flow_map']   # [1, 2, 914, 1380]  torch.float32
    else:
        batch_ori = None    
    if 'flow_map_inter' in data:    
        batch_ori_inter = data['flow_map_inter']   # [1, 2, 914, 1380]  torch.float32
    else:
        batch_ori_inter = None    
    if target is not None:
        target = F.interpolate(target, size=512, mode='bilinear', align_corners=False) # [1, 3, 512, 512] 
        target_256 = data['target_image_256'].to(dist_util.dev()) # [1, 3, 256, 256]
    else:
        target = None
        target_256 = None
    # source = F.interpolate(source, size=512, mode='bilinear', align_corners=False) #[1, 3, 512, 512] 
    # source_256 = data['source_image_256'].to(dist_util.dev()) # [1, 3, 256, 256]
    
    if settings.env.eval_dataset == 'hp-240':# false
        source_256 = source
        target_256 = target

    else: # true
        data['source_image_256'] = torch.nn.functional.interpolate(input=source.float(), size=(256, 256), mode='area')
        source_256 = data['source_image_256'].to(dist_util.dev())
        
        if 'target_image_256' in data:
            target_256 = data['target_image_256']
        else:
            target_256 = None
    if 'correspondence_mask' in data:
        mask = data['correspondence_mask'] # torch.bool [1, 914, 1380]
    else:
        mask = torch.ones((1, 512, 512), dtype=torch.bool).to(dist_util.dev()) # None

    return data, H_ori, W_ori, source, target, batch_ori, batch_ori_inter, source_256, target_256, source_vis, target_vis, mask, source_0



def run_sample_lr_dewarping(
    settings, logger, diffusion, model, radius, source, feature_size, 
    raw_corr, init_flow, c20, source_64, pyramid, doc_mask, 
    seg_map_all=None, textline_map=None, init_feat=None
):
    # init_flow = init_flow * feature_size
    # coords = initialize_flow(init_flow.shape[0], feature_size, feature_size, dist_util.dev())
    # coords_warped = coords + init_flow

    # local_corr = local_Corr(
    #     raw_corr.view(1, 1, feature_size, feature_size, feature_size, feature_size).to(dist_util.dev()),
    #     coords_warped.to(dist_util.dev()),
    #     radius,
    # )

    # local_corr = F.interpolate(
    #     local_corr.view(1, (2 * radius + 1) ** 2, feature_size, feature_size),
    #     size=feature_size,
    #     mode='bilinear',
    #     align_corners=True,
    # )

    # init_flow = F.interpolate(init_flow, size=feature_size, mode='bilinear', align_corners=True)
    # init_flow /= feature_size

    
    model_kwsettings = {'init_flow': init_flow, 'src_feat': c20, 'src_64':None, 
                        'y512':source, 'tmode':settings.env.train_mode,
                        'mask_cat': doc_mask,
                        'init_feat': init_feat,
                        'iter': settings.env.iter} # 'trg_feat': trg_feat
    # [1, 81, 64, 64] [1, 2, 64, 64] [1, 64, 64, 64]
    if settings.env.use_gt_mask == False:
        model_kwsettings['mask_y512'] = seg_map_all # [b, 384, 64, 64]
    if settings.env.use_line_mask == True:
        model_kwsettings['line_msk'] = textline_map # 
    image_size_h, image_size_w = feature_size, feature_size
    # tv_save_image(source,"vis_hp/debug_vis/source.png")
    # tv_save_image(doc_mask,"vis_hp/debug_vis/mask512_8877.png")
    logger.info(f"\nStarting sampling")

    sample, _ = diffusion.ddim_sample_loop(
        model,
        (1, 2, image_size_h, image_size_w), # 1,2,64,64
        noise=None,
        clip_denoised=settings.env.clip_denoised, # false
        model_kwargs=model_kwsettings,
        eta=0.0,
        progress=True,
        denoised_fn=None,
        sampling_kwargs={'src_img': source}, # 'trg_img': target
        logger=logger,
        n_batch=settings.env.n_batch,
        time_variant = settings.env.time_variant,
        pyramid=pyramid
    )

    sample = th.clamp(sample, min=-1, max=1)
    return sample



def run_evaluation_docunet(
    settings, logger, val_loader, diffusion: GaussianDiffusion, model, 
    pretrained_dewarp_model,pretrained_line_seg_model=None,pretrained_seg_model=None
):
    os.makedirs(f'vis_hp/{settings.env.eval_dataset_name}/{settings.name}', exist_ok=True)
    # batch_preprocessing = DocBatchPreprocessing(
    #             settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False
    #         )
    batch_preprocessing = None
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pyramid = VGGPyramid(train=False).to(dist_util.dev())
    SIZE = None
    trian_t = []
    for i, data in pbar:
        radius = 4
        raw_corr = None
        image_size = 64 
        data_path = data['path']
        # ref test
        # source_288 = F.interpolate(data['source_image']/255., size=(288), mode='bilinear', align_corners=True).to(dist_util.dev())
        source_288 = F.interpolate(data['source_image'], size=(288), mode='bilinear', align_corners=True).to(dist_util.dev())
        # tv_save_image(data['source_image']/255., "vis_hp/msk5/in{}".format(data['path'][0].split('/')[-1]))
        
        
        if settings.env.time_variant == True:
            init_feat = torch.zeros((data['source_image'].shape[0], 256, image_size, image_size), dtype=torch.float32).to(dist_util.dev()) 
        else:
            init_feat = None
        
        
        with torch.inference_mode():
            ref_bm, mask_x = pretrained_dewarp_model(source_288) # [1,2,288,288] 0~288  0~1
            # base = coords_grid_tensor((288,288)).to(ref_bm.device) # [1, 2, 288, 288]
            # ref_flow = ref_bm - base 
            ref_flow = ref_bm/287.0 # [-1, 1]  # [1,2,288,288]
        if settings.env.use_init_flow: 
            init_flow = F.interpolate(ref_flow, size=(image_size), mode='bilinear', align_corners=True) # [24, 2, 64, 64]                  
        else:
            init_flow = torch.zeros((data['source_image'].shape[0], 2, image_size, image_size), dtype=torch.float32).to(dist_util.dev()) 
        # mask_x = F.interpolate(mask_x, size=(512), mode='bilinear', align_corners=True) # 0-1
        # data['source_image'] = mask_x*data['source_image'].to(dist_util.dev()) # 0-255
        # mask_x_vis = mask_x*data['source_image'].to(dist_util.dev()) # 不存在最优mask阈值策略
        # tv_save_image(mask_x_vis, "vis_hp/msk_wore/{}".format(data['path'][0].split('/')[-1])) # 0~1 (288,288)
        (
            data,
            H_ori, # 512
            W_ori, # 512
            source, # [1, 3, 512, 512] 0-1
            target, # None
            batch_ori, # None
            batch_ori_inter, # None
            source_256,# [1, 3, 256, 256] 0-1
            target_256, # None
            source_vis, # [1, 3, H, W] cpu仅用于可视化
            target_vis, # None
            mask, # [1, 512, 512] 全白
            source_0
        ) = prepare_data(settings, batch_preprocessing, SIZE, data)
        
        with torch.no_grad():
            if settings.env.use_gt_mask == False:
                # ref_bm, mask_x = self.pretrained_dewarp_model(source_288) # [1,2,288,288] bm 0~288 mskx0-256
                mskx, d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d = pretrained_seg_model(source_288)
                hx6 = F.interpolate(hx6, size=image_size, mode='bilinear', align_corners=False)
                hx5d = F.interpolate(hx5d, size=image_size, mode='bilinear', align_corners=False)
                hx4d = F.interpolate(hx4d, size=image_size, mode='bilinear', align_corners=False)
                hx3d = F.interpolate(hx3d, size=image_size, mode='bilinear', align_corners=False)
                hx2d = F.interpolate(hx2d, size=image_size, mode='bilinear', align_corners=False)
                hx1d = F.interpolate(hx1d, size=image_size, mode='bilinear', align_corners=False)

                seg_map_all = torch.cat((hx6, hx5d, hx4d, hx3d, hx2d, hx1d), dim=1) # [b, 384, 64, 64]
                # tv_save_image(mskx,"vis_hp/debug_vis/mskx.png")
                if settings.env.use_line_mask:
                    textline_map, textline_mask = pretrained_line_seg_model(mskx) # [3, 64, 256, 256]
                    textline_map = F.interpolate(textline_map, size=image_size, mode='bilinear', align_corners=False) #  [3, 64, 64, 64]
            else:
                seg_map_all = None
                textline_map = None
        
        
         
        
        if settings.env.train_VGG:
            c20 = None
            feature_size = image_size
        else:
            feature_size = image_size
            if settings.env.train_mode == 'stage_1_dit_cat' or settings.env.train_mode =='stage_1_dit_cross':
                with th.no_grad():
                    c20  = extract_raw_features_single2(pyramid, source, source_256, feature_size) # [24, 1, 64, 64, 64, 64]
                # 平均互相关，VGG最浅层特征的下采样（512*512->64*64）
            else:
                with th.no_grad():
                    c20  = extract_raw_features_single(pyramid, source, source_256, feature_size) # [24, 1, 64, 64, 64, 64]
                # 平均互相关，VGG最浅层特征的下采样（512*512->64*64）
        
        source_64 = None # F.interpolate(source, size=(feature_size), mode='bilinear', align_corners=True)
        logger.info(f"Starting sampling with VGG Features")
        # init_flow = correlation_to_flow_w_argmax(
        #         raw_corr.view(1, 1, feature_size, feature_size, feature_size, feature_size),
        #         output_shape=(feature_size, feature_size),
        #     )  # B, 2, 64, 64 初始偏移场
        
        import time
        begin_train = time.time()
        sample = run_sample_lr_dewarping(
            settings,
            logger,
            diffusion,
            model,
            radius, # 4
            source, # [B, 3, 512, 512] 0~1
            feature_size, # 64
            raw_corr, # None
            init_flow, # [B, 2, 64, 64]   -1~1
            c20, # # [B, 64, 64, 64]   
            source_64, # None
            pyramid,
            mask_x, #mask_x,  # F.interpolate(mskx, size=(512), mode='bilinear', align_corners=True)[:,:1,:,:] , # mask_x 
            seg_map_all,
            textline_map,
            init_feat
        ) # sample: [1, 2, 64, 64] 偏移量 [-1,1]范围 五步DDIM的结果
        
        
        
        trian_t.append(time.time()-begin_train)  # 从这里宣布结束训练当前epoch

        
        
        
        

        
        # if settings.env.use_sr_net == True: # false
        #     logger.info('Running super resolution')
        #     sample_sr = None
        #     for j in range(1):
        #         batch_ori, sample_sr, init_flow_sr = run_sample_sr(
        #             settings, logger, diffusion_sr, model_sr, pyramid, data, sample, sample_sr
        #         )

        #         sample_ = F.interpolate(sample_sr, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
        #         sample_[:, 0, :, :] = sample_[:, 0, :, :] * W_ori
        #         sample_[:, 1, :, :] = sample_[:, 1, :, :] * H_ori

        #         sample_ = sample_.permute(0, 2, 3, 1)[mask]
        #         batch_ori_ = batch_ori.permute(0, 2, 3, 1)[mask]
        #         epe = th.sum((sample_ - batch_ori_.to(sample_.device)) ** 2, dim=1).sqrt()
        #         logger.info(f'sr iter: {i}, epe: {epe.mean()}')

        #     sample = F.interpolate(sample_sr, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
        #     sample[:, 0, :, :] = sample[:, 0, :, :] * W_ori
        #     sample[:, 1, :, :] = sample[:, 1, :, :] * H_ori
        #     init_flow = F.interpolate(init_flow_sr, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
        #     # init_flow[:, 0, :, :] = init_flow[:, 0, :, :] * W_ori
        #     # init_flow[:, 1, :, :] = init_flow[:, 1, :, :] * H_ori
        # sample = th.mean(sample[0], dim=0, keepdim=True)
        if settings.env.use_sr_net == False:
            sample = F.interpolate(sample, size=(H_ori, W_ori), mode='bilinear', align_corners=True) # [-1,+1] 偏移场
            # sample[:, 0, :, :] = sample[:, 0, :, :] * W_ori
            # sample[:, 1, :, :] = sample[:, 1, :, :] * H_ori 
            base = F.interpolate(coords_grid_tensor((512,512))/511., size=(H_ori, W_ori), mode='bilinear', align_corners=True)
            # sample = ( ((sample + base.to(sample.device)) )*2 - 1 )
            sample = ( ((sample + base.to(sample.device))*1 )*2 - 1 )*0.987 #  (2 * (bm / 286.8) - 1) * 0.99
            ref_flow = None
            if ref_flow is not None: 
                ref_flow = F.interpolate(ref_flow, size=(H_ori, W_ori), mode='bilinear', align_corners=True) # [-1,+1] 偏移场
                # ref_flow[:, 0, :, :] = ref_flow[:, 0, :, :] * W_ori 
                # ref_flow[:, 1, :, :] = ref_flow[:, 1, :, :] * H_ori 
                ref_flow  = (ref_flow + base.to(ref_flow.device))*2 -1
            # init_flow = F.interpolate(init_flow, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
        else:
            raise ValueError("Invalid value")
        
        if settings.env.visualize:
            visualize_dewarping(settings, sample, data, i, source_vis, data_path, ref_flow)

        
        # sample = sample.permute(0, 2, 3, 1)[mask]
        # init_flow[:, 0, :, :] = init_flow[:, 0, :, :] * W_ori
        # init_flow[:, 1, :, :] = init_flow[:, 1, :, :] * H_ori
        # init_flow = init_flow.permute(0, 2, 3, 1)[mask]
    # print("Elapsed time:{:.2f} minutes ".format(trian_t/60))
    print(len(trian_t))
    print("Elapsed time:{:.2f} avg_second ".format(sum(trian_t) / len(trian_t)))


def coords_grid_tensor(perturbed_img_shape):
    im_x, im_y = np.mgrid[0:perturbed_img_shape[0]-1:complex(perturbed_img_shape[0]),
                            0:perturbed_img_shape[1]-1:complex(perturbed_img_shape[1])]
    coords = np.stack((im_y,im_x), axis=2) # 先x后y，行序优先
    coords = th.from_numpy(coords).float().permute(2,0,1).to(dist_util.dev())  # (2, 512, 512)
    return coords.unsqueeze(0) # [2, 512, 512]


def validate(local_rank, args, val_loader, model, criterion):
    for i, sample in enumerate(val_loader):
        input1, label = sample # [2, 3, 288, 288],[2, 2, 288, 288]
        input1 = input1.to(local_rank,non_blocking=True)
        label = label.to(local_rank,non_blocking=True)
        # label = (label/288.0-0.5)*2
        
        with torch.no_grad():
            output = model(input1) # [3b, 2, 288, 288]
            # loss = F.l1_loss(output, label) #  合成图像强监督
            # test point
            # bm_test=(output/288.0-0.5)*2
            bm_test = (output/992.0-0.5)*2
            label = (label/992.0-0.5)*2
            # bm_test = output
            bm_test = F.interpolate(bm_test, size=(1000,1000), mode='bilinear', align_corners=True)
            label = F.interpolate(label, size=(1000,1000), mode='bilinear', align_corners=True)
            input1 =  F.interpolate(input1, size=(1000,1000), mode='bilinear', align_corners=True)
            regis_image1 = F.grid_sample(input=input1, grid=bm_test.permute(0,2,3,1), align_corners=True)
            regis_image2 = F.grid_sample(input=input1, grid=label.permute(0,2,3,1), align_corners=True)
            
            # regis_image2 = F.grid_sample(input=a_sample[None], grid=bm_test[None].permute(0,2,3,1), align_corners=True)
            tv_save_image(input1[0], "backup/test/ori.png")
            tv_save_image(regis_image1[0], "backup/test/aaa.png")
            tv_save_image(regis_image2[0], "backup/test/gt.png")
            
            # warped_src = warp(source_vis.to(sample.device).float(), sample) # [1, 3, 1629, 981]
            # warped_src = warped_src[0].permute(1, 2, 0).detach().cpu().numpy() # (1873, 1353, 3)
            # warped_src = Image.fromarray((warped_src).astype(np.uint8))
            # warped_src.save(f"vis_hp/{settings.env.eval_dataset_name}/{settings.name}/dewarped_pred/warped_{data_path[0].split('/')[-1]}")



    return None