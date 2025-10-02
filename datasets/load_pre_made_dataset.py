import glob
import os.path

from datasets.listdataset import ListDataset, Aug_ListDataset, Aug_Doc3d_ListDataset
from datasets.util import split2list

def make_doc3d_dataset_list(dir, split=None):
    dataset_list = []
    for sample_name in os.listdir(dir):
        # flow_map = os.path.join(dir, os.path.basename(flow_map))
        # root_filename = os.path.basename(flow_map)[:-9]
        img = os.path.join(dir, sample_name, 'img.png') # source image
        flow_map = os.path.join(dir, sample_name, 'bm.mat') # flow_map
        abd = os.path.join(dir, sample_name, 'recon.png') # recon
        # img2 = os.path.join(dir, root_filename + '_img_2.jpg') # target image
        dataset_list.append([img, flow_map, abd])
        
    return split2list(dataset_list, split, default_split=0.97)


def make_doc_dataset_list(dir, split=None):
    dataset_list = []
    for sample_name in os.listdir(dir):
        # flow_map = os.path.join(dir, os.path.basename(flow_map))
        # root_filename = os.path.basename(flow_map)[:-9]
        img = os.path.join(dir, sample_name, 'warped_document.png') # source image
        flow_map = os.path.join(dir, sample_name, 'warped_BM.npz') # flow_map
        abd = os.path.join(dir, sample_name, 'warped_recon.png') # recon
        # img2 = os.path.join(dir, root_filename + '_img_2.jpg') # target image
        dataset_list.append([img, flow_map, abd])
        
    return split2list(dataset_list, split, default_split=0.97)


def make_dataset(dir, get_mapping, split=None, dataset_name=None):
    """
    Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm  in folder images and
      [name]_flow.flo' in folder flow """
    images = []
    if get_mapping: # false
        flow_dir = 'mapping'
        # flow_dir is actually mapping dir in that case, it is always normalised to [-1,1]
    else:
        flow_dir = 'flow'
    image_dir = 'images'

    # Make sure that the folders exist
    if not os.path.isdir(dir):
        raise ValueError("the training directory path that you indicated does not exist ! ")
    if not os.path.isdir(os.path.join(dir, flow_dir)):
        raise ValueError("the training directory path that you indicated does not contain the flow folder ! "
                         "Check your directories.")
    if not os.path.isdir(os.path.join(dir, image_dir)):
        raise ValueError("the training directory path that you indicated does not contain the images folder ! "
                         "Check your directories.")

    for flow_map in sorted(glob.glob(os.path.join(dir, flow_dir, '*_flow.flo'))):
        flow_map = os.path.join(flow_dir, os.path.basename(flow_map))
        root_filename = os.path.basename(flow_map)[:-9]
        img1 = os.path.join(image_dir, root_filename + '_img_1.jpg') # source image
        img2 = os.path.join(image_dir, root_filename + '_img_2.jpg') # target image
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue
        if dataset_name is not None: # false
            images.append([[os.path.join(dataset_name, img1),
                            os.path.join(dataset_name, img2)],
                            os.path.join(dataset_name, flow_map)])
        else: # true
            images.append([[img1, img2], flow_map])
    return split2list(images, split, default_split=0.97)


def assign_default(default_dict, dict):
    if dict is None:
        dall = default_dict
    else:
        dall = {}
        dall.update(default_dict)
        dall.update(dict)
    return dall


def Doc_Dataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None, get_mapping=False, compute_mask_zero_borders=False,
                   add_discontinuity=False, min_nbr_perturbations=5, max_nbr_perturbations=6,
                   parameters_v2=None):
    """
    Builds a dataset from existing image pairs and corresponding ground-truth flow fields and optionally add
    some flow perturbations.
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset
        get_mapping: output mapping instead of flow in __getittem__ ?
        compute_mask_zero_borders: output mask of zero borders ?
        add_discontinuity: add discontinuity to image pairs and corresponding ground-truth flow field ?
        min_nbr_perturbations:
        max_nbr_perturbations:
        parameters_v2: parameters of v2

    Returns:
        train_dataset
        test_dataset

    """
    train_list, test_list = make_doc_dataset_list(root, split) # get list [[sample1],[sample2],...]
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                compute_mask_zero_borders=compute_mask_zero_borders)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                compute_mask_zero_borders=compute_mask_zero_borders)
    return train_dataset, test_dataset

def Aug_Doc_Dataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None, get_mapping=False, compute_mask_zero_borders=False,
                   add_discontinuity=False, min_nbr_perturbations=5, max_nbr_perturbations=6,
                   parameters_v2=None):
    train_list, test_list = make_doc_dataset_list(root, split) # get list [[sample1],[sample2],...]
    train_dataset = Aug_ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                compute_mask_zero_borders=compute_mask_zero_borders)
    test_dataset = Aug_ListDataset(root, test_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                compute_mask_zero_borders=compute_mask_zero_borders)
    return train_dataset, test_dataset

def Doc3d_Dataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None, get_mapping=False, compute_mask_zero_borders=False,
                   add_discontinuity=False, min_nbr_perturbations=5, max_nbr_perturbations=6,
                   parameters_v2=None):
    train_list, test_list = make_doc3d_dataset_list(root, split) # get list [[sample1],[sample2],...]
    train_dataset = Aug_Doc3d_ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                compute_mask_zero_borders=compute_mask_zero_borders)
    test_dataset = Aug_Doc3d_ListDataset(root, test_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                compute_mask_zero_borders=compute_mask_zero_borders)
    return train_dataset, test_dataset
