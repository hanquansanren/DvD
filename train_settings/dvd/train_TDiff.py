import torch
import torchvision.transforms as transforms
from termcolor import colored

from datasets.load_pre_made_dataset import \
    Doc_Dataset, Aug_Doc_Dataset, Doc3d_Dataset, Mix_Dataset

from datasets.batch_processing import GLUNetBatchPreprocessing
from utils_data.image_transforms import ArrayToTensor
from utils_data.loaders import Loader
from train_settings.models.geotr.geotr_core import GeoTr, GeoTr_Seg, GeoTr_Seg_womask, GeoTr_Seg_Inf,\
                                                     reload_segmodel, reload_model, Seg

from ..models.geotr.unet_model import UNet

from .improved_diffusion import dist_util, logger
from .improved_diffusion.resample import create_named_schedule_sampler
from .improved_diffusion.script_util import (args_to_dict,
                                             create_model_and_diffusion,
                                             model_and_diffusion_defaults)
from .improved_diffusion.train_util import TrainLoop


def run(settings):
    settings.description = 'train settings for dvd'

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())
    logger.configure(dir=f"{settings.env.train_mode}_{settings.env.dataset_name}")
                     
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        device=dist_util.dev(),
        train_mode=settings.env.train_mode,
        tv=settings.env.time_variant,
        **args_to_dict(settings, model_and_diffusion_defaults().keys())
    )
    # print(model)
    if settings.env.resume_checkpoint:
        state_dict = dist_util.load_state_dict(settings.env.resume_checkpoint, map_location='cpu')
        # # 删除部分参数
        # exclude_params = ['input_blocks.0.0.weight', 'input_blocks.0.0.bias']  # 替换为你想忽略的参数名
        # for param in exclude_params:
        #     if param in state_dict:
        #         del state_dict[param]
        model.load_state_dict(state_dict, strict=False)

    settings.device = dist_util.dev()
    print(f"Setting device to {settings.device}")
    model = model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(settings.env.schedule_sampler, diffusion)
    # if settings.env.use_gt_mask == True:
    #     pretrained_dewarp_model = GeoTr_Seg_womask()
    # elif settings.env.use_gt_mask == False:
    pretrained_line_seg_model = UNet(n_channels=3, n_classes=1)
    pretrained_seg_model = Seg()
    # line_model_ckpt = torch.load(settings.env.line_seg_model_path, map_location='cpu')
    # print(checkpoint)
    # print(pretrained_line_seg_model)
    # new_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('module.unet')}
    # torch.save({'model': new_state_dict}, './checkpoints/backup/line_model.pth')
    
    # new_state_dict = {}
    # for key, value in line_model_ckpt.items():
    #     # 如果key以 'module.unet.' 开头，去掉前缀
    #     if key.startswith('module.seg.'):
    #         new_key = key[len('module.seg.'):] 
    #         new_state_dict[new_key] = value
    #     else:
    #         pass
    #         # new_state_dict[key] = value

    # # 保存修改后的模型权重
    # torch.save({'model': new_state_dict}, './checkpoints/backup/seg_model.pth')
    
    line_model_ckpt = dist_util.load_state_dict(settings.env.line_seg_model_path, map_location='cpu')['model']
    pretrained_line_seg_model.load_state_dict(line_model_ckpt, strict=True)    
    pretrained_line_seg_model.to(dist_util.dev())
    pretrained_line_seg_model.eval()

    seg_model_ckpt = dist_util.load_state_dict(settings.env.new_seg_model_path, map_location='cpu')['model']
    pretrained_seg_model.load_state_dict(seg_model_ckpt, strict=True)    
    pretrained_seg_model.to(dist_util.dev())
    pretrained_seg_model.eval()
    

    # pretrained_dewarp_model = GeoTr_Seg_Inf()
    # reload_segmodel(pretrained_dewarp_model.msk, settings.env.seg_model_path)
    # reload_model(pretrained_dewarp_model.GeoTr, settings.env.dewarping_model_path)
    # pretrained_dewarp_model.to(dist_util.dev())
    # pretrained_dewarp_model.eval()
    
    
    
    logger.log("creating data loader...")    
    
    # 1. Define training and validation datasets
    # datasets, pre-processing of the images is done within the network function !
    if settings.env.dataset_name == 'doc_debug':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        train_dataset, _ = Doc_Dataset(root=settings.env.doc_debug,
                                        source_image_transform=img_transforms,
                                        target_image_transform=None,
                                        flow_transform=flow_transform,
                                        split=1,
                                        get_mapping=False)
        train_loader = Loader('train', train_dataset, batch_size=settings.env.batch_size, shuffle=True,
                            drop_last=False, training=True, num_workers=settings.env.n_threads)  
    elif settings.env.dataset_name == 'aug_doc':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        train_dataset, _ = Aug_Doc_Dataset(root=settings.env.doc_debug,
                                        source_image_transform=img_transforms,
                                        target_image_transform=None,
                                        flow_transform=flow_transform,
                                        split=1,
                                        get_mapping=False)
    elif settings.env.dataset_name == 'doc3d':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        train_dataset, _ = Doc3d_Dataset(root=settings.env.doc_debug,
                                        source_image_transform=img_transforms,
                                        target_image_transform=None,
                                        flow_transform=flow_transform,
                                        split=1,
                                        get_mapping=False)

        
    train_loader = Loader('train', train_dataset, batch_size=settings.env.batch_size, shuffle=True,
                        drop_last=False, training=True, num_workers=settings.env.n_threads)    
    
    # Setting dataset name into diffusion because of the semantic setting.
    setattr(diffusion, 'dataset', settings.env.dataset_name) 

    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    logger.log("training...")
    batch_preprocessing = GLUNetBatchPreprocessing(settings, apply_mask=False, apply_mask_zero_borders=False,
                                                sparse_ground_truth=False)
    
    # 4. Define loss module    
    TrainLoop(
        model=model, 
        pretrained_dewarp_model = pretrained_seg_model,
        pretrained_line_seg_model = pretrained_line_seg_model,
        diffusion=diffusion,
        settings=settings,
        batch_preprocessing=batch_preprocessing,
        data=train_loader,
        batch_size=settings.env.batch_size, 
        microbatch=settings.env.microbatch, 
        lr=settings.env.lr, 
        ema_rate=settings.env.ema_rate, 
        log_interval=settings.env.log_interval, 
        save_interval=settings.env.save_interval, 
        resume_checkpoint=settings.env.resume_checkpoint,
        use_fp16=settings.env.use_fp16, 
        fp16_scale_growth=settings.env.fp16_scale_growth, 
        schedule_sampler=schedule_sampler,
        weight_decay=settings.env.weight_decay, 
        lr_anneal_steps=settings.env.lr_anneal_steps, 
        resume_step=settings.env.resume_step,
        use_gt_mask = settings.env.use_gt_mask, 
        use_init_flow = settings.env.use_init_flow,
        train_mode = settings.env.train_mode,
        use_line_mask = settings.env.use_line_mask
    ).run_loop_dewarping()





