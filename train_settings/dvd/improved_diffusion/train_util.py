# Python standard library imports
import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from train_settings.dvd.feature_backbones.VGG_features import VGGPyramid

from ..eval_utils import extract_raw_features_single, extract_raw_features_single2
from ..evaluation import prepare_data
from . import dist_util, logger
from .fp16_util import (make_master_params, master_params_to_model_params,
                        model_grads_to_master_grads, unflatten_master_params,
                        zero_grad)
from .nn import update_ema
from .resample import UniformSampler
from torchvision.utils import save_image as tv_save_image

from PIL import Image

INITIAL_LOG_LOSS_SCALE = 20.0

def coords_grid_tensor(perturbed_img_shape):
    im_x, im_y = np.mgrid[0:287:complex(perturbed_img_shape[0]),
                            0:287:complex(perturbed_img_shape[1])]
    coords = np.stack((im_y,im_x), axis=2) # 先x后y，行序优先
    coords = torch.from_numpy(coords).float().permute(2,0,1) # (512, 512, 2)
    return coords.unsqueeze(0) # [2, 512, 512]


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        pretrained_dewarp_model,
        pretrained_line_seg_model,
        diffusion,
        settings,
        batch_preprocessing,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        resume_step = 0,
        use_gt_mask = True,
        use_init_flow = False,
        train_mode = None,
        use_line_mask = True
    ):  
        self.model = model
        self.pretrained_dewarp_model = pretrained_dewarp_model
        self.pretrained_line_seg_model = pretrained_line_seg_model
        self.diffusion = diffusion
        self.settings = settings
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        # print(self.ema_rate)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = resume_step
        self.use_gt_mask = use_gt_mask
        self.use_init_flow = use_init_flow
        self.use_line_mask = use_line_mask
        self.train_mode = train_mode
        self.global_batch = self.batch_size * dist.get_world_size()
        # print(self.global_batch)
        self.model_params = list(self.model.parameters())
        # self.load_model_params = list(self.model.parameters())
        # del self.load_model_params[5]
        # del self.load_model_params[6]
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("resume form cheakpoint and optimizer in step {}".format(self.resume_step))
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            print("resume form cheakpoint without optimizer in step 0")
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True
            )

        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.batch_processing = batch_preprocessing

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location='cpu')
            # # print(len(state_dict.key()))
            # exclude_params = ['input_blocks.0.0.weight', 'input_blocks.0.0.bias']  # 替换为你想忽略的参数名
            # for param in exclude_params:
            #     if param in state_dict:
            #         del state_dict[param]
                    
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            self.model.load_state_dict(
                    state_dict, strict=False
                )
            self.model.to(dist_util.dev())

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params) #  master_params

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            
            # exclude_params = [4, 5]  # 替换为你想忽略的参数名
            # for param in exclude_params:
            #     if param in state_dict['state']:
            #         del state_dict['state'][param]
            # original_list = list(range(450))
            # state_dict['param_groups'][0]['params'] = [x for x in original_list if x not in exclude_params]
            
            
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()
   

    def run_loop_dewarping(self):
        self.feature_size = 64
        self.image_size = 64 
        self.radius = 4
        
        while True:
            if self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
                break

            indices = list(range(0, len(self.data))) # 656项*18=11808
    
            self.pyramid = VGGPyramid(train=False).to(dist_util.dev())
            SIZE = None

            # batch_preprocessing = DocBatchPreprocessing(
            #     self.settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False
            # )
            batch_preprocessing = None
            for i, data in zip(indices, self.data): 
                source_288 = F.interpolate(data['source_image'], size=(288), mode='bilinear', align_corners=True).to(dist_util.dev()) # 0~1    
                
                
                
                #     data['source_image'] = data['source_image']*data['doc_mask']*255. # 0~1
                # tv_save_image(data['source_image'],"vis_hp/debug_vis/source.png")
                
                if self.settings.env.time_variant == True:
                    init_feat = torch.zeros((data['source_image'].shape[0], 256, self.image_size, self.image_size), dtype=torch.float32).to(dist_util.dev()) 
                else:
                    init_feat = None
                
                
                if self.use_init_flow: # false
                    # source_288 = F.interpolate(data['source_image'], size=(288), mode='bilinear', align_corners=True).to(dist_util.dev()) # 0~1    
                    with torch.no_grad():
                        ref_bm, mask_x = self.pretrained_dewarp_model(source_288) # [1,2,288,288] bm 0~288 mskx0-256
                    base = coords_grid_tensor((288,288)).to(ref_bm.device)
                    ref_flow = ref_bm - base
                    ref_flow = ref_flow/287.0 # [-1, 1]  # [1,2,288,288]
                    init_flow = F.interpolate(ref_flow, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True) # [24, 2, 64, 64]                  
                else:
                    # init_flow = None
                    # source_288 = None
                    init_flow = torch.zeros((data['source_image'].shape[0], 2, self.image_size, self.image_size), dtype=torch.float32).to(dist_util.dev()) 
                # if self.use_gt_mask == False:
                #     data['source_image'] = F.interpolate(mask_x, size=(512), mode='bilinear', align_corners=True) # 0-256
                # tv_save_image(data['source_image']/255., "vis_hp/debug_vis/mskx.png")
                # tv_save_image(data['source_image']/255., "vis_hp/debug_vis/mskx.png")
                (
                    data,
                    H_ori, # 512
                    W_ori, # 512
                    source, # [b, 3, 512, 512] 0-1
                    target, # None
                    batch_ori, # label [b, 2, 512, 512] 0-512 绝对偏移量
                    batch_ori_inter, # label_inter [b, 2, 512, 512] 0-512 绝对偏移量
                    source_256, # [b, 3, 256, 256] 0-1
                    target_256, # None
                    source_pil, # [24, 3, 512, 512] cpu仅用于可视化 0-256
                    target_pil, # None
                    mask, # [1, 512, 512] 全白
                    source_0 # debug可视化，训练时不用
                ) = prepare_data(self.settings, batch_preprocessing, SIZE, data)
                
                with torch.no_grad():
                    if self.use_gt_mask == False:
                        # ref_bm, mask_x = self.pretrained_dewarp_model(source_288) # [1,2,288,288] bm 0~288 mskx0-256
                        mskx, d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d = self.pretrained_dewarp_model(source_288)
                        hx6 = F.interpolate(hx6, size=self.image_size, mode='bilinear', align_corners=False)
                        hx5d = F.interpolate(hx5d, size=self.image_size, mode='bilinear', align_corners=False)
                        hx4d = F.interpolate(hx4d, size=self.image_size, mode='bilinear', align_corners=False)
                        hx3d = F.interpolate(hx3d, size=self.image_size, mode='bilinear', align_corners=False)
                        hx2d = F.interpolate(hx2d, size=self.image_size, mode='bilinear', align_corners=False)
                        hx1d = F.interpolate(hx1d, size=self.image_size, mode='bilinear', align_corners=False)

                        seg_map_all = torch.cat((hx6, hx5d, hx4d, hx3d, hx2d, hx1d), dim=1) # [b, 384, 64, 64]
                        # tv_save_image(mskx,"vis_hp/debug_vis/mskx.png")
                        if self.use_line_mask:
                            textline_map, textline_mask = self.pretrained_line_seg_model(mskx) # [3, 64, 256, 256]
                            textline_map = F.interpolate(textline_map, size=self.image_size, mode='bilinear', align_corners=False) #  [3, 64, 64, 64]
                    else:
                        seg_map_all = None
                        textline_map = None
                
                
                if self.settings.env.train_VGG:
                    c20 = None
                else:
                    if self.train_mode == 'stage_1_dit_cat' or 'stage_1_dit_cross': # dit
                        with torch.no_grad():
                            c20 = extract_raw_features_single2(self.pyramid, source, source_256, feature_size=self.image_size) # [b, 448, 64, 64]
                    else:    # unet
                        with torch.no_grad():
                            c20 = extract_raw_features_single(self.pyramid, source, source_256, self.feature_size) # [24, 1, 64, 64, 64, 64]
                
                _, _, H, W = batch_ori.shape # 512,512
                batch_ori[:,0,:,:] /= (W-1) # label 归一化到[-1,1]
                batch_ori[:,1,:,:] /= (H-1) # label 归一化到[-1,1] # [24, 2, 512, 512]
                if batch_ori_inter is not None:
                    batch_ori_inter[:,0,:,:] /= (W-1) # label 归一化到[-1,1]
                    batch_ori_inter[:,1,:,:] /= (H-1) # label 归一化到[-1,1] # [24, 2, 512, 512]
                batch = F.interpolate(batch_ori, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True) # [24, 2, 64, 64]                  
                # tv_save_image(source,"vis_hp/debug_vis/source.png")
                # tv_save_image(textline_mask,"vis_hp/debug_vis/tx_line.png")
                # tv_save_image(d0,"vis_hp/debug_vis/mask512_77.png")
                # tv_save_image(data['doc_mask'],"vis_hp/debug_vis/mask512_77.png")
                source_288 = None
                cond = {'source_0':source_0, 'y': source_288, 'y512': source, 'init_flow': init_flow, 
                        'src_feat': c20, 'src_64': None, 'tv': self.settings.env.time_variant, 'tmode':self.train_mode,
                        'mask_cat': data['doc_mask'],
                        'init_feat': init_feat,
                        'iter': self.settings.env.iter,
                        'mode':"train"} # 'trg_feat': trg_feat data['doc_mask']
                if self.use_gt_mask == False:
                    cond['mask_y512'] = seg_map_all # [b, 384, 64, 64]
                if self.use_line_mask == True:
                    cond['line_msk'] = textline_map #  [b, 64, 64, 64]
                # tv_save_image(cond['y512'],"vis_hp/debug_vis/source.png")
                # tv_save_image(cond['mask_y512'],"vis_hp/debug_vis/mask512_77.png")
                self.run_step(batch, batch_ori, mask, cond, self.pyramid, batch_ori_inter)
                
                
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def get_gpu_memory_usage(device_id=None):
        allocated_bytes = torch.cuda.memory_allocated(0)
        return allocated_bytes

    def run_step(self, batch, batch_ori, mask, cond, pyramid=None, batch_ori_inter=None):
        if cond['iter'] == True:
            self.forward_backward_iteration(batch, batch_ori, batch_ori_inter, mask, cond, pyramid)
        elif cond['tmode'] == "stage_1_dit_cat" or cond['tmode'] == "stage_1_dit_cross":
            self.forward_backward_new_dit(batch, batch_ori, batch_ori_inter, mask, cond)

        
        # elif cond['tv'] == False:
        #     self.forward_backward(batch, batch_ori, mask, cond)
        # elif cond['tv'] == "new":
        #     self.forward_backward_new(batch, batch_ori, batch_ori_inter, mask, cond)
                    
        if self.use_fp16: # false
            self.optimize_fp16()
        else: # true
            self.optimize_normal()
        self.log_step()

    def forward_backward_new_dit(self, batch, batch_ori, batch_ori_inter, mask, cond): # [24, 2, 64, 64] [24, 2, 512, 512] [24, 512, 512] init_flow','src_feat'
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 64, 64] 
            micro_ori = batch_ori[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            micro_ori_inter = batch_ori_inter[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t, weights= self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if weights == None:
                weights = torch.ones(24).to(dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_losses_new_dit, 
                self.ddp_model,
                micro,
                micro_ori,
                micro_ori_inter,
                mask,
                t,
                model_kwargs=cond,
            )

            if last_batch or not self.use_ddp: # true
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            
            
            if self.use_fp16: # false
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)

    def forward_backward_iteration(self, batch, batch_ori, batch_ori_inter, mask, cond, pyramid): # [24, 2, 64, 64] [24, 2, 512, 512] [24, 512, 512] init_flow','src_feat'
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 64, 64] 
            micro_ori = batch_ori[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            micro_ori_inter = batch_ori_inter[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t, weights= self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            # t = torch.tensor([4, 3, 2, 1, 0, 1, 0, 2, 0, 1, 0, 3, 
            #                   3, 0, 3, 3, 2, 0, 0, 0, 2, 3, 4, 0], dtype=torch.int64).to(dist_util.dev())
            # if weights == None:
            #     weights = torch.ones(24).to(dist_util.dev())
            
            # compute_losses = self.diffusion.training_losses_time_variant 
            compute_losses = functools.partial(
                self.diffusion.training_losses_time_variant, 
                self.ddp_model,
                micro,
                micro_ori,
                micro_ori_inter,
                mask,
                t,
                model_kwargs=cond,
                pyramid=pyramid
            )
            
            
            if last_batch or not self.use_ddp: # true
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            if self.use_fp16: # false
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)
            # self.optimize_normal()

    def forward_backward_new(self, batch, batch_ori, batch_ori_inter, mask, cond): # [24, 2, 64, 64] [24, 2, 512, 512] [24, 512, 512] init_flow','src_feat'
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 64, 64] 
            micro_ori = batch_ori[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            micro_ori_inter = batch_ori_inter[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t, weights= self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if weights == None:
                weights = torch.ones(24).to(dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_losses_new, 
                self.ddp_model,
                micro,
                micro_ori,
                micro_ori_inter,
                mask,
                t,
                model_kwargs=cond,
            )

            if last_batch or not self.use_ddp: # true
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            
            
            if self.use_fp16: # false
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)

    def forward_backward(self, batch, batch_ori, mask, cond): # [24, 2, 64, 64] [24, 2, 512, 512] [24, 512, 512] init_flow','src_feat'
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 64, 64] 
            micro_ori = batch_ori[i : i + self.microbatch].to(dist_util.dev()) # [24, 2, 512, 512]
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t, weights= self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if weights == None:
                weights = torch.ones(24).to(dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_losses, 
                self.ddp_model,
                micro,
                micro_ori,
                mask,
                t,
                model_kwargs=cond,
            )

            if last_batch or not self.use_ddp: # true
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            
            
            if self.use_fp16: # false
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)


    def optimize_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad == None:
                pass
            else:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            # print("return")
            return
        # print("lr_anneal_steps",self.lr_anneal_steps)
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        if ts.dim() == 0:
            ts = ts.unsqueeze(0)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)