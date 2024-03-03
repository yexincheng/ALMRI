import numpy as np
from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
import os
from datasets import path
import json
import skimage
import matplotlib.pyplot as plt
import cv2

def transform_to_original(ori_zeros, mode, view ,id, pred):
    with open(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', view, mode, 'sub_index.json'), 'r') as f:
        sub_index = json.load(f)
        label_index = sub_index[f'{id}.FatImaging_W.nii.gz']

    ori_zeros[label_index] += pred
    return ori_zeros

def majority_voting(array1, array2, array3):
    # Stack the arrays along a new axis to create a 4D array
    stacked_arrays = np.stack((array1, array2, array3), axis=-1)

    # Use np.apply_along_axis to apply the majority voting function along the last axis
    result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=-1, arr=stacked_arrays)
    result = result.astype(np.uint8)
    return result

def resize2origin(pred_axial, pred_sagittal, pred_coronal, views, original_shape, mode, id):
    pred_axial_resized = np.array([cv2.resize(p, (original_shape[2], original_shape[1]), interpolation=cv2.INTER_NEAREST) for p in pred_axial], dtype=np.uint8)
    view = views[0]
    ori_zeros = np.zeros(original_shape, dtype=np.uint8)
    pred_axial_ori = transform_to_original(ori_zeros, mode, view, id, pred_axial_resized)

    pred_sagittal_resized = np.array([cv2.resize(p, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST) for p in pred_sagittal], dtype=np.uint8)
    view = views[1]
    ori_zeros = np.zeros((original_shape[2], original_shape[0], original_shape[1]), dtype=np.uint8)
    pred_sagittal_ori = transform_to_original(ori_zeros, mode, view, id, pred_sagittal_resized).transpose(1, 2, 0)

    pred_coronal_resized = np.array([cv2.resize(p, (original_shape[2], original_shape[0]), interpolation=cv2.INTER_NEAREST) for p in pred_coronal], dtype=np.uint8)
    view = views[2]
    ori_zeros = np.zeros((original_shape[1], original_shape[0], original_shape[2]), dtype=np.uint8)
    pred_coronal_ori = transform_to_original(ori_zeros, mode, view, id, pred_coronal_resized).transpose(1, 0, 2)

    return pred_axial_ori, pred_sagittal_ori, pred_coronal_ori

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=1.2)) 

def save_preds_nii(id, prefix, task, mode, preds, save_path):
    # load image and gt
    img = sitk.ReadImage(os.path.join(path.RAINE_ORGAN_IMAGES_51, f'{id}.FatImaging_W.nii.gz'))
    img_array = sitk.GetArrayFromImage(img)
    original_shape = img_array.shape
    # gt = sitk.ReadImage(os.path.join(path.RAINE_ORGAN_MANUAL_GTS_51, f'{id}.FatImaging_W.nii.gz'))
    # gt_array = sitk.GetArrayFromImage(gt)

    with open(os.path.join(prefix, task, mode, 'sub_index.json'), 'r') as f:
        sub_index = json.load(f)
    label_index = sub_index[f'{id}.FatImaging_W.nii.gz']

    ori_zeros = np.zeros(original_shape)
    # resize prediction to original size
    preds_resized = skimage.transform.resize(preds, (preds.shape[0], original_shape[1], original_shape[2]), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    # print(preds_resized.shape)
    ori_zeros[label_index] += preds_resized
    # print(ori_zeros.shape)

    # save prediction
    result_image = sitk.GetImageFromArray(ori_zeros)
    result_image.CopyInformation(img)
    # write the image
    sitk.WriteImage(result_image, os.path.join(save_path, f'{id}.FatImaging_W.nii.gz'))
    return ori_zeros, img_array

def min_max_norm_3dimg(img):
    """Min-max normalization [0, 1] for 3D image
    SAM ask for a uint image, with pixel values in [0, 255]
    Args:
        img: 3D image
    Return:
        img: normalized uint 3D image
    """
    
    img_norm = (img - np.min(img)) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None) * 255
    img_norm[img==0] = 0
    img_norm = img_norm.astype(np.uint8)
    return img_norm

def roi_bbox(gt2D, seed=2024):
    '''Returns a bounding box from a mask region of interest (ROI)'''
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    np.random.seed(seed)
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = np.array([x_min, y_min, x_max, y_max])
    return bbox

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask size'''
    bbox = np.array([0, 0, mask.shape[1], mask.shape[0]])
    return bbox

# create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, sample_pool_path):
        self.npz_files = sorted(sample_pool_path) 
        # print(self.npz_files[0])
        self.npz_data = [np.load(f) for f in self.npz_files]
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        # print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        # whole image as bbox
        bboxes = np.array([0, 0, gt2D.shape[1], gt2D.shape[0]])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()

class NpzDataset_roi(Dataset): 
    def __init__(self, sample_pool_path, seed):
        self.npz_files = sorted(sample_pool_path) 
        # print(self.npz_files[0])
        self.npz_data = [np.load(f) for f in self.npz_files]
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        # print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
        self.seed = seed
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        if self.seed:
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            np.random.seed(self.seed)
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()
    