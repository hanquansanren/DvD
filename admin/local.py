class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = 'checkpoints'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir
        self.pre_trained_models_dir = self.workspace_dir+"/backup"
        ########################################################################################
        self.eval_dataset_name = 'docunet'  
        if self.eval_dataset_name ==  'dir300':
            self.eval_dataset = '/home/share/dir300'   
        elif self.eval_dataset_name == 'docunet':
            self.eval_dataset = '/home/share/docunet'   
        elif self.eval_dataset_name == 'anyphoto':
            self.eval_dataset = '/home/share/init_all_final/init_8'
        elif self.eval_dataset_name == 'docreal':
            self.eval_dataset = '/home/share/docreal'
        ########################################################################################
        self.dataset_name = 'doc3d'  
        if self.dataset_name == 'doc_debug':
            self.doc_debug = '/home/share/train_bug3'
            self.time_variant = False 
        elif self.dataset_name == 'aug_doc3d': 
            self.doc_debug = '/home/share/train_bug3' 
            self.time_variant = "new"  
        elif self.dataset_name == 'doc3d': 
            self.doc_debug = '/home/share/doc3d_rearrange2' 
            self.time_variant = True
        self.train_mode = 'stage_1_dit_cross' 
        self.iter = True 
        self.train_VGG = True     
        self.use_gt_mask = False 
        self.use_line_mask = True 
        self.use_init_flow = False
        self.lr = 1e-4   
        self.diffusion_steps = 3 
        self.batch_size = 10 
        self.n_threads = 4
        ######################################
        self.log_interval = 20
        self.save_interval = 4000
        self.resume_step = 0 #152000 #1390000
        self.resume_checkpoint =  None
        self.nbr_objects = 4
        self.min_area_objects = 1300
        self.compute_object_reprojection_mask = True
        self.initial_pretrained_model = None
        self.data_dir = ''
        self.schedule_sampler = 'uniform' #'uniform' 'multi' 'fixed'
        self.weight_decay = 0.0
        self.lr_anneal_steps = 0
        self.microbatch = -1
        self.ema_rate = 0.9999
        self.use_fp16 = False
        self.fp16_scale_growth = 0.001
        self.image_size = 64
        self.flow_size = (64, 64)
        self.num_channels = 128
        self.num_res_blocks = 3
        self.num_heads = 4
        self.num_heads_upsample = -1
        self.attention_resolutions = "16,8"
        self.dropout = 0.0
        self.learn_sigma = False
        self.sigma_small = False
        self.class_cond = False
        self.noise_schedule = 'cosine'
        self.use_kl = False
        self.predict_xstart = True
        self.rescale_timesteps = True
        self.rescale_learned_sigmas = True
        self.use_checkpoint = False
        self.use_scale_shift_norm = True
        self.clip_denoised = False
        self.num_samples = 10000
        self.val_batch_size = 1
        self.use_ddim = False
        self.model_path = 'checkpoints/stage_1_dit_cross_mix_2025-03-24-19-48-08-929885/model1852000.pt'
        self.seg_model_path = "checkpoints/backup/seg.pth"
        self.line_seg_model_path = 'checkpoints/backup/line_model2.pth' # 'checkpoints/backup/line_model2.pth' 'checkpoints/backup/30.pt'
        self.new_seg_model_path = 'checkpoints/backup/seg_model.pth'  
        self.timestep_respacing = ''
        self.n_batch = 2    # The number of multiple hypotheses
        self.visualize = True    # Set True, if you want qualitative results.
        self.use_sr_net = False
