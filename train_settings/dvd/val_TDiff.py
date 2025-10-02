import os
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

import datasets
from utils_data.image_transforms import ArrayToTensor

from .evaluation import run_evaluation_docunet
from .improved_diffusion import dist_util, logger
from .improved_diffusion.script_util import (args_to_dict,
                                             create_model_and_diffusion,
                                             model_and_diffusion_defaults)
from train_settings.models.geotr.geotr_core import GeoTr, GeoTr_Seg, GeoTr_Seg_Inf,\
                                                    reload_segmodel, reload_model, Seg
                                                      
from datasets.doc_dataset.doc_benchmark import Doc_dewarping_Data1
from train_settings.dvd.evaluation import validate


from ..models.geotr.unet_model import UNet

class WrappedDiffusionModel(torch.nn.Module):
    def __init__(self, model, t, model_kwargs):
        super().__init__()
        self.model = model
        self.t = t
        self.model_kwargs = model_kwargs

    def forward(self, x):
        return self.model(x, self.t, **self.model_kwargs)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def run(settings):
    dist_util.setup_dist()
    logger.configure(dir=f"SAMPLING_{settings.env.eval_dataset}_{settings.name}")
    logger.log(f"Corruption Disabled. Evaluating on Original {settings.env.eval_dataset}")
    logger.log("Loading model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        device=dist_util.dev(),
        train_mode=settings.env.train_mode, # stage 1
        tv=settings.env.time_variant,
        **args_to_dict(settings, model_and_diffusion_defaults().keys()),
    )
    setattr(diffusion, "settings", settings)
    
    
    
    
    # pretrained_dewarp_model = GeoTr(num_attn_layers=6, num_token=(288//8)**2)
    pretrained_dewarp_model = GeoTr_Seg_Inf()
    reload_segmodel(pretrained_dewarp_model.msk, settings.env.seg_model_path)
    pretrained_dewarp_model.to(dist_util.dev())
    pretrained_dewarp_model.eval()

    if settings.env.use_line_mask:
        pretrained_line_seg_model = UNet(n_channels=3, n_classes=1)
        pretrained_seg_model = Seg()
        line_model_ckpt = dist_util.load_state_dict(settings.env.line_seg_model_path, map_location='cpu')['model']
        pretrained_line_seg_model.load_state_dict(line_model_ckpt, strict=True)    
        pretrained_line_seg_model.to(dist_util.dev())
        pretrained_line_seg_model.eval()

        seg_model_ckpt = dist_util.load_state_dict(settings.env.new_seg_model_path, map_location='cpu')['model']
        pretrained_seg_model.load_state_dict(seg_model_ckpt, strict=True)    
        pretrained_seg_model.to(dist_util.dev())
        pretrained_seg_model.eval()




    model.cpu().load_state_dict(dist_util.load_state_dict(settings.env.model_path, map_location="cpu"), strict=False)
    logger.log(f"Model loaded with {settings.env.model_path}")


    model.to(dist_util.dev())
    print(get_parameter_number(model))
    model.eval()
    
    logger.log("Creating data loader...")
    logger.info('\n:============== Logging Configs ==============')
    for key, value in settings.env.__dict__.items():
        if key in ['model_path', 'timestep_respacing', 'eval_dataset']:
            logger.info(f"\t{key}:\t{value}") 
    logger.info(':===============================================\n')



    if settings.env.eval_dataset_name == "docunet" or settings.env.eval_dataset_name == "dir300" or settings.env.eval_dataset_name == "anyphoto" or settings.env.eval_dataset_name == "docreal": 
        # 1. Define training and validation datasets
        input_transform = transforms.Compose([ArrayToTensor(get_float=True)])  # only put channel first
        test_set = datasets.Doc_benchmark(
            settings.env.eval_dataset, 
            input_transform,
        )
        
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True,
                                drop_last=False, num_workers=8)
        logger.info(f"Starting sampling")
        run_evaluation_docunet(
            settings, logger, test_loader, diffusion, model, pretrained_dewarp_model,pretrained_line_seg_model,pretrained_seg_model)
    elif settings.env.eval_dataset_name == "doc_val": 
        val_set = Doc_dewarping_Data1(root_path= settings.env.eval_dataset, transforms=input_transform, resolution=288, model_setting = "doctr")
        val_loader = DataLoader(val_set, batch_size=1, num_workers=4,
                                drop_last=False, pin_memory=True,shuffle=False)
        prec1 = validate(val_loader, pretrained_dewarp_model)

    dist.barrier()
    logger.log("sampling complete")
