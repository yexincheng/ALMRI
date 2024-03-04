import numpy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #TODO: add yml
import cv2 #TODO: add yml
import os
import random
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import time
from glob import glob
from utils import NpzDataset, NpzDataset_roi, transform_to_original, majority_voting, resize2origin, show_box
from torch.utils.data import DataLoader
import json
import pickle
import skimage
import SimpleITK as sitk
from metrics import compute_dice_coefficient
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train box regression model')
    parser.add_argument('--seed', type=int, default=2024, help='random seed for data sampling')

    args = parser.parse_args()

    axial_view = 'medsam_MRI_LeftKidney_axial_norm'
    coronal_view = 'medsam_MRI_LeftKidney_coronal_norm'
    sagittal_view = 'medsam_MRI_LeftKidney_sagittal_norm'

    training_axial = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', axial_view, 'train', '*.npz')))
    training_sagittal = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', sagittal_view, 'train', '*.npz')))
    training_coronal = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', coronal_view, 'train', '*.npz')))


    seed = args.seed
    with open(f'datasets/RAINE_organ_51/MRI_LeftKidney/DynUNet_sampling/random_100_{seed}pool.json', 'r') as f:
        training_pool = json.load(f)
    training_pool = [k.split('/')[-1].split('.')[0] for k in training_pool]  
    for i in range(1,41):
        # get the image embedding and downsample the GT
        # i = 40
        training_ids = training_pool[:i]
        # find data with ids from 3 views
        training_axial_ids = [k for k in training_axial if k.split('/')[-1].split('.')[0] in training_ids]
        training_sagittal_ids = [k for k in training_sagittal if k.split('/')[-1].split('.')[0] in training_ids]
        training_coronal_ids = [k for k in training_coronal if k.split('/')[-1].split('.')[0] in training_ids]
        training_3views = training_axial_ids + training_sagittal_ids + training_coronal_ids
        print(f'Seed {seed}, Number of training subjects: {i}', f'\tNumber of training pool: {len(training_3views)}')
        train_embeddings = []
        train_gts = []

        print('Start dowwmsampling GT...')
        t1 = time.time()

        for fname in tqdm(training_3views):
            # read data
            name = fname.split('/')[-1].split('.')[0]
            embeddings = np.load(fname)['img_embeddings']
            gts = np.load(fname)['gts']
            for emb, gt in zip(embeddings, gts):    # (256, 64, 64) (256, 256)
                emb = emb.transpose(1,2,0).reshape(-1, 256)  # (64, 64, 256)
                # print(emb.shape, gt.shape)
                train_embeddings.append(emb)

                gt = cv2.resize(gt, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
                train_gts.append(gt.flatten())
                
        train_embeddings = np.concatenate(train_embeddings)
        train_gts = np.concatenate(train_gts)

        # Create a linear regression model and fit it to the training data
        print('Start training...')
        start_time = time.time()
        model = LogisticRegression(max_iter=1000) 
        model.fit(train_embeddings, train_gts)
        end_time = time.time()
        print(f'Training time: {end_time - start_time} seconds')
        task = 'MRI_LeftKidney_3views_label'
        pickle.dump(model, open(f'checkpoints/medsam_norm_bbox_regression/{task}_{i}samples_seed{seed}.pkl', 'wb'))
        print('All done')
