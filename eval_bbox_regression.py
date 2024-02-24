import numpy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2 
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

def upsampled_vote(bbox_model, index, mode, original_shape):
    axial_view = 'MRI_LeftKidney_axial'
    sagittal_view = 'MRI_LeftKidney_sagittal'
    coronal_view = 'MRI_LeftKidney_coronal'

    axial_set = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', axial_view, mode, '*.npz')))
    sagittal_set = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', sagittal_view, mode, '*.npz')))
    coronal_set = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', coronal_view, mode, '*.npz')))

    # sagittal
    fname_sagittal = sagittal_set[index]
    id = os.path.basename(fname_sagittal).split('.')[0]
    imgs_sagittal = np.load(fname_sagittal)['imgs']
    embeddings_sagittal = np.load(fname_sagittal)['img_embeddings']
    gts_sagittal = np.load(fname_sagittal)['gts']
    y_preds_sagittal = []
    mask_preds_sagittal = []

    for img_sagittal, emb_sagittal, gt_sagittal in zip(imgs_sagittal, embeddings_sagittal,  gts_sagittal):    # (256, 64, 64) (256, 256)

        emb_sagittal = emb_sagittal.transpose(1,2,0).reshape(-1, 256)  # (64, 64, 256)e)
        y_pred_sagittal = bbox_model.predict(emb_sagittal)
        y_pred_sagittal = y_pred_sagittal.reshape((64,64))
        y_preds_sagittal.append(y_pred_sagittal)
        mask_pred_l_sagittal = cv2.resize(y_pred_sagittal, (gt_sagittal.shape[1], gt_sagittal.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_preds_sagittal.append(mask_pred_l_sagittal)
        
    # coronal
    fname_coronal = coronal_set[index]
    imgs_coronal = np.load(fname_coronal)['imgs']
    embeddings_coronal = np.load(fname_coronal)['img_embeddings']
    gts_coronal = np.load(fname_coronal)['gts']
    y_preds_coronal = []
    mask_preds_coronal = []

    for img_coronal, emb_coronal, gt_coronal in zip(imgs_coronal, embeddings_coronal, gts_coronal):    # (256, 64, 64) (256, 256)
        
        emb_coronal = emb_coronal.transpose(1,2,0).reshape(-1, 256)  # (64, 64, 256)
        y_pred_coronal = bbox_model.predict(emb_coronal)
        y_pred_coronal = y_pred_coronal.reshape((64,64))
        y_preds_coronal.append(y_pred_coronal)
        mask_pred_l_coronal = cv2.resize(y_pred_coronal, (gt_coronal.shape[1], gt_coronal.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_preds_coronal.append(mask_pred_l_coronal)
        
    # axial
    fname_axial = axial_set[index]
    imgs_axial = np.load(fname_axial)['imgs']
    embeddings_axial = np.load(fname_axial)['img_embeddings']
    gts_axial = np.load(fname_axial)['gts']
    y_preds_axial = []
    mask_preds_axial = []
    # gts_downsampled = []
    for img_axial, emb_axial, gt_axial in zip(imgs_axial, embeddings_axial, gts_axial):    # (256, 64, 64) (256, 256)

        emb_axial = emb_axial.transpose(1,2,0).reshape(-1, 256)  # (64, 64, 256)
        y_pred_axial = bbox_model.predict(emb_axial)
        y_pred_axial = y_pred_axial.reshape((64,64))
        y_preds_axial.append(y_pred_axial)
        mask_pred_l_axial = cv2.resize(y_pred_axial, (gt_axial.shape[1], gt_axial.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_preds_axial.append(mask_pred_l_axial)    

        # gts_downsampled.append(cv2.resize(gt_axial, dsize=(64, 64), interpolation=cv2.INTER_NEAREST))

    mask_preds_axial_ori, mask_preds_sagittal_ori, mask_preds_coronal_ori = resize2origin(mask_preds_axial, mask_preds_sagittal, mask_preds_coronal, original_shape, mode, id)
    start_time = time.time()
    vote_mask_preds = majority_voting(mask_preds_axial_ori, mask_preds_sagittal_ori, mask_preds_coronal_ori)
    end_time = time.time()
    print(f'{id} Voting time: {end_time - start_time}')
    return vote_mask_preds

def bbox_vote_infer(vote_mask_preds, view, mode, index, sam_model, device, seed):

    dataset = sorted(glob(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', view, mode, '*.npz')))
    id = dataset[index].split('/')[-1].split('.')[0]
    imgs = np.load(dataset[index])['imgs']
    embeddings = np.load(dataset[index])['img_embeddings']
    gts = np.load(dataset[index])['gts']
    
    with open(os.path.join('/scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51', view, mode, 'sub_index.json'), 'r') as f:
        sub_index = json.load(f)
        label_index = sub_index[f'{id}.FatImaging_W.nii.gz']
    if 'axial' in view:
        vote_mask_preds_roi = vote_mask_preds[label_index]
    elif 'coronal' in view:
        vote_mask_preds_roi = vote_mask_preds[:,label_index,:]
    elif 'sagittal' in view:
        vote_mask_preds_roi = vote_mask_preds[:,:,label_index]
    bboxes = []
    preds = []
    for j, (img, emb, gt, _) in enumerate(zip(imgs, embeddings, gts, vote_mask_preds_roi)):    # (256, 64, 64) (256, 256)
        
        if 'axial' in view:
            mask_j = vote_mask_preds_roi[j]
        elif 'coronal' in view:
            mask_j = vote_mask_preds_roi[:,j,:]
        elif 'sagittal' in view:
            mask_j = vote_mask_preds_roi[:,:,j]
        # print(mask_j.shape, img.shape, emb.shape, gt.shape)
        # print(np.unique(mask_j))
        mask = cv2.resize(mask_j, (256,256), interpolation=cv2.INTER_NEAREST)
        # print(np.unique(mask))
        # bbox 
        y_indices, x_indices = np.where(mask > 0)
        H, W = mask.shape
        if np.all(mask == 0):
            bbox = np.array([0, 0, H, W])
            medsam_seg = np.zeros((H, W), dtype=np.uint8)

        else:
            # print(y_indices, x_indices, np.unique(gt))
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            np.random.seed(seed)
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        # predictor.set_image(img)
        # masks, _, _ = predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=bbox[None, :],
        #     multimask_output=False,
        #     )
        
            with torch.no_grad():
                sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
                bbox_trans = sam_trans.apply_boxes(bbox, (H, W))
                box_torch = torch.as_tensor(bbox_trans, dtype=torch.float32, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)
                if len(mask.shape) == 2:
                    mask = mask[None, :, :]
                    mask_torch = torch.as_tensor(mask, dtype=torch.float, device=device)
                    mask_torch = mask_torch[None, :, :, :]

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            medsam_seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=torch.as_tensor(emb).to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
                )
            medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
            medsam_seg_prob = np.stack(medsam_seg_prob, axis=0)
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        bboxes.append(bbox)
        preds.append(medsam_seg)
    preds = np.array(preds)
    bboxes = np.array(bboxes)
    return preds, bboxes, imgs, gts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/dice/bbox_regression')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/SAM/sam_vit_b_01ec64.pth')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--task', type=str, default='MRI_LeftKidney_3views_label')
    parser.add_argument('--num_samples', type=int, default=40)
    parser.add_argument('--bseed', type=int, default=2023, help='random seed for bbox jitter')
    parser.add_argument('--sseed', type=int, default=2023, help='random seed for random sample')
    args = parser.parse_args()
   
    bbox_model = pickle.load(open(f'checkpoints/bbox_regression/{args.task}_{args.num_samples}samples_seed{args.sseed}.pkl', 'rb'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f'datasets/RAINE_organ_51/MRI_LeftKidney/{args.mode}/sub_index.json', 'r') as f:
        sub_index = json.load(f)
    id_set = sorted([k.split('.')[0] for k in sub_index.keys()])

    sam_model = sam_model_registry['vit_b'](checkpoint=args.checkpoint).to(device)
    sam_model.eval()
    
    dices_final = []
    dices_axial = []
    dices_sagittal = []
    dices_coronal = []
    for i in tqdm(range(len(id_set))):
        
        id = id_set[i]
        # print(id)
        original_img = sitk.ReadImage(f'datasets/RAINE_organ_51/trainingset_51/{id}.FatImaging_W.nii.gz')
        original_imgarray = sitk.GetArrayFromImage(original_img)
        original_gtarray = sitk.GetArrayFromImage(sitk.ReadImage(f'datasets/RAINE_organ_51/trainingset_51/labels/final/{id}.FatImaging_W.nii.gz'))
        original_gtarray = np.uint8(original_gtarray==1)
        original_shape = original_gtarray.shape

        if os.path.exists(os.path.join('datasets/vote_mask_preds/', f'{id}_3views_{args.num_samples}samples.npy')):
            vote_mask_preds = np.load(os.path.join('datasets/vote_mask_preds/', f'{id}_3views_{args.num_samples}samples.npy'))
        else:
            vote_mask_preds = upsampled_vote(bbox_model, i, args.mode, original_shape)
            np.save(os.path.join('datasets/vote_mask_preds/', f'{id}_3views_40samples.npy'), vote_mask_preds)

        preds_axial, bboxes_axial, imgs_axial, gts_axial = bbox_vote_infer(vote_mask_preds, 'MRI_LeftKidney_axial', args.mode, i, sam_model, device, args.bseed)
        dice_axial = compute_dice_coefficient(gts_axial, preds_axial)
        dices_axial.append(dice_axial)

        preds_sagittal, bboxes_sagittal, imgs_sagittal, gts_sagittal = bbox_vote_infer(vote_mask_preds, 'MRI_LeftKidney_sagittal', args.mode, i, sam_model, device, args.bseed)
        dice_sagittal = compute_dice_coefficient(gts_sagittal, preds_sagittal)
        dices_sagittal.append(dice_sagittal)

        preds_coronal, bboxes_coronal, imgs_coronal, gts_coronal = bbox_vote_infer(vote_mask_preds, 'MRI_LeftKidney_coronal', args.mode, i, sam_model, device, args.bseed)
        dice_coronal = compute_dice_coefficient(gts_coronal, preds_coronal)
        dices_coronal.append(dice_coronal)

        pred_axial_ori, pred_sagittal_ori, pred_coronal_ori = resize2origin(preds_axial, preds_sagittal, preds_coronal, original_shape, args.mode, id)
        pred_final = majority_voting(pred_axial_ori, pred_sagittal_ori, pred_coronal_ori)
        dice_final = compute_dice_coefficient(original_gtarray, pred_final)
        dices_final.append(dice_final)

    np.save(os.path.join(args.output_dir, f'{args.mode}_vote_upsampled_3views_emb_bseed{args.bseed}_sample{args.num_samples}_sseed{args.sseed}.npy'), dices_final)   
    np.save(os.path.join(args.output_dir, f'{args.mode}_vote_upsampled_axial_emb_bseed{args.bseed}_sample{args.num_samples}_sseed{args.sseed}.npy'), dices_axial)
    np.save(os.path.join(args.output_dir, f'{args.mode}_vote_upsampled_sagittal_emb_bseed{args.bseed}_sample{args.num_samples}_sseed{args.sseed}.npy'), dices_sagittal)
    np.save(os.path.join(args.output_dir, f'{args.mode}_vote_upsampled_coronal_emb_bseed{args.bseed}_sample{args.num_samples}_sseed{args.sseed}.npy'), dices_coronal)

