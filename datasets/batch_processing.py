import torch
import torch.nn.functional as F
import torch.utils.data
from packaging import version




def pre_process_image_glunet(source_img, device, mean_vector=[0.485, 0.456, 0.406],
                             std_vector=[0.229, 0.224, 0.225]):
    """
    Image is in range [0, 255}. Creates image at 256x256, and applies imagenet weights to both.
    Args:
        source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
        device:
        mean_vector:
        std_vector:

    Returns:
        image at original and 256x256 resolution
    """
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = source_img.shape
    source_img_copy = source_img.float().to(device).div(255.0)

    # mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    # std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    # source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device).div(255.0), size=(256, 256), mode='area')

    # source_img_256 = source_img_256.float().div(255.0)
    # source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    return source_img_copy.to(device), source_img_256.to(device)

class CATsBatchPreprocessing:
    """ Class responsible for processing the mini-batch to create the desired training inputs for GLU-Net based networks.
    Particularly, from the source and target images at original resolution as well as the corresponding ground-truth
    flow field, needs to create the source, target and flow at resolution 256x256 for training the L-Net.
    """

    def __init__(self, settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False,
                 mapping=False):
        """
        Args:
            settings: settings
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target image) for loss
                                     computation?
            sparse_ground_truth: is ground-truth sparse? Important for downscaling/upscaling of the flow field
            mapping: load correspondence map instead of flow field?
        """
        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders
        self.sparse_ground_truth = sparse_ground_truth

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mapping = mapping

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The input data, should contain the following fields:
                        'source_image', 'target_image', 'correspondence_mask'
                        'flow_map' if self.mapping is False else 'correspondence_map_pyro'
                        'mask_zero_borders' if self.apply_mask_zero_borders
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image', 'source_image_256', 'target_image_256', flow_map',
                        'flow_map_256', 'mask', 'mask_256', 'correspondence_mask'
        """
        source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'], self.device)
        target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'], self.device)

        # At original resolution
        if self.sparse_ground_truth:
        
            flow_gt_original = mini_batch['flow_map'][0].to(self.device)
            flow_gt_256 = mini_batch['flow_map'][1].to(self.device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            if flow_gt_256.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_256 = flow_gt_256.permute(0, 3, 1, 2)
        else:
            if self.mapping:
                mapping_gt_original = mini_batch['correspondence_map_pyro'].to(self.device)
                # flow_gt_original = unormalise_and_convert_mapping_to_flow(mapping_gt_original.permute(0,3,1,2))
            else:
                flow_gt_original = mini_batch['flow_map'].to(self.device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape

            # now we have flow everywhere, at 256x256 resolution, b, _, 256, 256
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256), mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0/float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0/float(h_original)

        bs, _, h_original, w_original = flow_gt_original.shape
        bs, _, h_256, w_256 = flow_gt_256.shape

        # defines the mask to use during training
        mask = None
        mask_256 = None
        if self.apply_mask_zero_borders:
            # make mask to remove all black borders
            mask = mini_batch['mask_zero_borders'].to(self.device)  # bxhxw, torch.uint8

            mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).byte()  # bx256x256, rounding
            mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()
        elif self.apply_mask:
            if self.sparse_ground_truth:
                mask = mini_batch['correspondence_mask'][0].to(self.device)
                mask_256 = mini_batch['correspondence_mask'][1].to(self.device)
            else:
                mask = mini_batch['correspondence_mask'].to(self.device)  # bxhxw, torch.uint8
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                         align_corners=False).squeeze(1).byte()   # bx256x256, rounding
                mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()

        mini_batch['source_image'] = source_image
        mini_batch['target_image'] = target_image
        mini_batch['source_image_256'] = source_image_256
        mini_batch['target_image_256'] = target_image_256
        mini_batch['flow_map'] = flow_gt_original
        mini_batch['flow_map_256'] = flow_gt_256
        mini_batch['mask'] = mask
        mini_batch['mask_256'] = mask_256
        if self.sparse_ground_truth:
            mini_batch['correspondence_mask'][0] = mini_batch['correspondence_mask'][0].to(self.device)
            mini_batch['correspondence_mask'][1] = mini_batch['correspondence_mask'][1].to(self.device)
        else:
            mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
       
        
        return mini_batch

class GLUNetBatchPreprocessing:
    """ Class responsible for processing the mini-batch to create the desired training inputs for GLU-Net based networks.
    Particularly, from the source and target images at original resolution as well as the corresponding ground-truth
    flow field, needs to create the source, target and flow at resolution 256x256 for training the L-Net.
    """

    def __init__(self, settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False,
                 mapping=False, megadepth=False):
        """
        Args:
            settings: settings
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target image) for loss
                                     computation?
            sparse_ground_truth: is ground-truth sparse? Important for downscaling/upscaling of the flow field
            mapping: load correspondence map instead of flow field?
        """
        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders
        self.sparse_ground_truth = sparse_ground_truth
        self.megadepth = megadepth

        self.device = getattr(settings, 'device', None)
        # if self.device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")
        print("Using device: {}".format(self.device))
        self.mapping = mapping

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The input data, should contain the following fields:
                        'source_image', 'target_image', 'correspondence_mask'
                        'flow_map' if self.mapping is False else 'correspondence_map_pyro'
                        'mask_zero_borders' if self.apply_mask_zero_borders
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image', 'source_image_256', 'target_image_256', flow_map',
                        'flow_map_256', 'mask', 'mask_256', 'correspondence_mask'
        """
        
        
        source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'], self.device)
        target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'], self.device)
        
        # At original resolution
        if self.sparse_ground_truth:
            flow_gt_original = mini_batch['flow_map'][0].to(self.device)
            flow_gt_256 = mini_batch['flow_map'][1].to(self.device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            if flow_gt_256.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_256 = flow_gt_256.permute(0, 3, 1, 2)
        else:
            if self.mapping:
                mapping_gt_original = mini_batch['correspondence_map_pyro'][-1].to(self.device)
                # flow_gt_original = unormalise_and_convert_mapping_to_flow(mapping_gt_original.permute(0,3,1,2))
            else:
                if self.megadepth:
                    flow_gt_original = mini_batch['flow_map'][0].to(self.device)
                else:
                    flow_gt_original = mini_batch['flow_map'].to(self.device)

            src_vis = mini_batch['source_image']
            trg_vis = mini_batch['target_image']   
            flow_vis = mini_batch['flow_map'][0]     

            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape

            # now we have flow everywhere, at 256x256 resolution, b, _, 256, 256
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256), mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0/float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0/float(h_original)

        bs, _, h_original, w_original = flow_gt_original.shape
        bs, _, h_256, w_256 = flow_gt_256.shape

        # defines the mask to use during training
        mask = None
        mask_256 = None
        if self.apply_mask_zero_borders:
            # make mask to remove all black borders
            mask = mini_batch['mask_zero_borders'].to(self.device)  # bxhxw, torch.uint8

            mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).byte()  # bx256x256, rounding
            mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()
        
        elif self.apply_mask:
            if self.sparse_ground_truth:
                mask = mini_batch['correspondence_mask'][0].to(self.device)
                mask_256 = mini_batch['correspondence_mask'][1].to(self.device)
            else:
                if self.megadepth:
                    mask = mini_batch['correspondence_mask'][0].to(self.device)  # bxhxw, torch.uint8
                else:
                    mask = mini_batch['correspondence_mask'].to(self.device)  # bxhxw, torch.uint8
                    
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                         align_corners=False).squeeze(1).byte()   # bx256x256, rounding
                mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()

        mini_batch['source_image'] = source_image
        mini_batch['target_image'] = target_image
        mini_batch['source_image_256'] = source_image_256
        mini_batch['target_image_256'] = target_image_256
        mini_batch['flow_map'] = flow_gt_original
        mini_batch['flow_map_256'] = flow_gt_256
        mini_batch['mask'] = mask
        mini_batch['mask_256'] = mask_256

        if self.sparse_ground_truth:
            if self.megadepth:
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'][0].to(self.device)
            else:
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
                
        else:
            if self.megadepth:
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'][0].to(self.device)
            else:
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
       
        return mini_batch



class DocBatchPreprocessing:
    """ Class responsible for processing the mini-batch to create the desired training inputs for GLU-Net based networks.
    Particularly, from the source and target images at original resolution as well as the corresponding ground-truth
    flow field, needs to create the source, target and flow at resolution 256x256 for training the L-Net.
    """

    def __init__(self, settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False,
                 mapping=False, megadepth=False):
        """
        Args:
            settings: settings
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target image) for loss
                                     computation?
            sparse_ground_truth: is ground-truth sparse? Important for downscaling/upscaling of the flow field
            mapping: load correspondence map instead of flow field?
        """
        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders
        self.sparse_ground_truth = sparse_ground_truth
        self.megadepth = megadepth

        self.device = getattr(settings, 'device', None)
        # if self.device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")
        print("Using device: {}".format(self.device))
        self.mapping = mapping

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The input data, should contain the following fields:
                        'source_image', 'target_image', 'correspondence_mask'
                        'flow_map' if self.mapping is False else 'correspondence_map_pyro'
                        'mask_zero_borders' if self.apply_mask_zero_borders
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image', 'source_image_256', 'target_image_256', flow_map',
                        'flow_map_256', 'mask', 'mask_256', 'correspondence_mask'
        """
        
        
        source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'], self.device) # [24, 3, 512, 512],[24, 3, 256, 256])
        # target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'], self.device)
        target_image = None
        
        # At original resolution
        if self.sparse_ground_truth: # false
            flow_gt_original = mini_batch['flow_map'][0].to(self.device)
            flow_gt_256 = mini_batch['flow_map'][1].to(self.device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            if flow_gt_256.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_256 = flow_gt_256.permute(0, 3, 1, 2)
        else: # true
            if self.mapping: # false
                mapping_gt_original = mini_batch['correspondence_map_pyro'][-1].to(self.device)
                # flow_gt_original = unormalise_and_convert_mapping_to_flow(mapping_gt_original.permute(0,3,1,2))
            else: # true
                if self.megadepth: # false
                    flow_gt_original = mini_batch['flow_map'][0].to(self.device)
                elif 'flow_map' in mini_batch:# true
                    flow_gt_original = mini_batch['flow_map'].to(self.device) # [24, 2, 512, 512]
                else:
                    mini_batch['source_image'] = source_image
                    mini_batch['source_image_256'] = source_image_256
                    return mini_batch
            # src_vis = mini_batch['source_image']
            # trg_vis = mini_batch['target_image']   
            # flow_vis = mini_batch['flow_map'][0]     

            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape

            # now we have flow everywhere, at 256x256 resolution, b, _, 256, 256
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256), mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0/float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0/float(h_original)

        bs, _, h_original, w_original = flow_gt_original.shape
        bs, _, h_256, w_256 = flow_gt_256.shape

        # defines the mask to use during training
        mask = None
        mask_256 = None
        if self.apply_mask_zero_borders: # false
            # make mask to remove all black borders
            mask = mini_batch['mask_zero_borders'].to(self.device)  # bxhxw, torch.uint8

            mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).byte()  # bx256x256, rounding
            mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()
        
        elif self.apply_mask: # false
            if self.sparse_ground_truth:
                mask = mini_batch['correspondence_mask'][0].to(self.device)
                mask_256 = mini_batch['correspondence_mask'][1].to(self.device)
            else:
                if self.megadepth:
                    mask = mini_batch['correspondence_mask'][0].to(self.device)  # bxhxw, torch.uint8
                else:
                    mask = mini_batch['correspondence_mask'].to(self.device)  # bxhxw, torch.uint8
                    
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                         align_corners=False).squeeze(1).byte()   # bx256x256, rounding
                mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()

        mini_batch['source_image'] = source_image
        if 'target_image' in mini_batch: 
            mini_batch['target_image'] = target_image
        else:
            mini_batch['target_image'] = None 
        mini_batch['source_image_256'] = source_image_256
        mini_batch['target_image_256'] = None
        mini_batch['flow_map'] = flow_gt_original
        mini_batch['flow_map_256'] = flow_gt_256
        mini_batch['mask'] = mask
        mini_batch['mask_256'] = mask_256

        if self.sparse_ground_truth: # false
            if self.megadepth: # false
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'][0].to(self.device)
            else:
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
                
        else: # true
            if self.megadepth: # false
                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'][0].to(self.device)
            else: # true
                pass
                # mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
        return mini_batch

