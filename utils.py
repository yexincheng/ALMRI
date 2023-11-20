import numpy as np
from torch.utils.data import Dataset
import torch

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

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    # y_indices, x_indices = np.where(mask > 0)
    # x_min, x_max = np.min(x_indices), np.max(x_indices)
    # y_min, y_max = np.min(y_indices), np.max(y_indices)
    # # add perturbation to bounding box coordinates
    # H, W = mask.shape
    # np.random.seed(2023)
    # x_min = max(0, x_min - np.random.randint(0, 20))
    # x_max = min(W, x_max + np.random.randint(0, 20))
    # y_min = max(0, y_min - np.random.randint(0, 20))
    # y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = np.array([0, 0, mask.shape[1], mask.shape[0]])
    # return np.array([x_min, y_min, x_max, y_max])
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
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        # H, W = gt2D.shape
        # x_min = max(0, x_min - np.random.randint(0, 20))
        # x_max = min(W, x_max + np.random.randint(0, 20))
        # y_min = max(0, y_min - np.random.randint(0, 20))
        # y_max = min(H, y_max + np.random.randint(0, 20))
        # bboxes = np.array([x_min, y_min, x_max, y_max])
        # whole image as bbox
        bboxes = np.array([0, 0, gt2D.shape[1], gt2D.shape[0]])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()