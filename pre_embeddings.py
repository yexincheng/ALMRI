import os 
from glob import glob
import argparse
import SimpleITK as sitk
import numpy as np
import datasets.path as path
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from skimage import transform, segmentation, io
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import min_max_norm_3dimg
import json

def get_slice(array, index, i):
    """    Get the slice of the 3D array from different viewa"""
    x, y, z = array.shape
    if index == x:
        return array[i,:,:]
    elif index == y:
        return array[:,i,:]
    elif index == z:
        return array[:,:,i]

# preprocess the dataset
def sam_preprocess(filter, view, img, gt, image_size, sam_model, device, post_norm):
    """Resize image and ground truth to 256*256, and get the image embedding
    Args:
        img: 3D image
        gt: 3D ground truth
        image_size: image size
        sam_model: sam model
        device: device
    Return:
        img: 2D image list
        gt: 2D ground truth list
        img_embeddings: image embeddings
    """
    imgs = []
    gts = []
    img_embeddings = []
    index = img.shape[view]
    label_index = []
    for i in range(index):
        gt_slice_i = get_slice(gt, index, i)
        gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        if np.sum(gt_slice_i)>=filter: # select slice containing organ
            img_slice_i = get_slice(img, index, i)
            label_index.append(i)
            # resize img_slice_i to 256x256
            img_slice_i = transform.resize(img_slice_i, (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            # convert to three channels
            img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
            imgs.append(img_slice_i)
            # print('img_slice_i', img_slice_i.shape)
            # assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
            gts.append(gt_slice_i)

            if sam_model is not None:
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_slice_i)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                # model input: (1, 3, 1024, 1024)
                input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
                if post_norm:
                    print('post normalization')
                    input_image = input_image.cpu().numpy()
                    input_image = (input_image - input_image.min()) / np.clip(input_image.max() - input_image.min(), a_min=1e-8, a_max=None)
                    input_image = torch.as_tensor(input_image).to(device)
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])

    # print('label_index', label_index)
    if sam_model is not None:
        return imgs, gts, img_embeddings, label_index
    else:
        return imgs, gts, label_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='./datasets/RAINE_organ_51', help='dataset path prefix')
    parser.add_argument('--task', type=str, default='MRI_Pancreas', help='task name')
    parser.add_argument('--label_id', type=int, default=3, help='label id')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/SAM/sam_vit_b_01ec64.pth', help='SAM checkpoint for fine tunning')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--filter', type=int, default=3, help='filter for selecting slice containing organ')
    parser.add_argument('--view', type=int, default=0, help='0 for axial, 1 for coronal, 2 for sagittal')
    parser.add_argument('--post_norm', action='store_true', help='post normalization for sam model')
    args = parser.parse_args()
    # label id
    # left kidney: 1,
    # right kidney: 2,
    # pancreas: 3,
    # background: 0,

    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data paths
    images = sorted(glob(os.path.join(path.RAINE_ORGAN_IMAGES_51, '*.nii.gz')))
    gts = sorted(glob(os.path.join(path.RAINE_ORGAN_MANUAL_GTS_51, '*.nii.gz')))
    names = [os.path.basename(image).split('/')[-1] for image in images]
    # split data into train, test
    np.random.seed(2023)
    np.random.shuffle(names)
    train_names = sorted(names[:int(len(names)*0.8)]) # 40
    test_names = sorted(names[int(len(names)*0.8):]) # 11
    if args.mode == 'train':
        ds_names = train_names
    elif args.mode == 'test':
        ds_names = test_names
    # prepare the save path
    save_path = os.path.join(args.prefix, args.task, args.mode)
    os.makedirs(save_path, exist_ok=True)

    # set up the model
    sam_model = sam_model_registry[args.sam_model_type](checkpoint=args.checkpoint).to(device)
    sub_index = {}

    # preprocess the dataset
    # outliers: sub-12400, sub-14870, sub-16880, sub-22770, sub-41810, sub-52220
    for name in tqdm(ds_names):
        img_path = os.path.join(path.RAINE_ORGAN_IMAGES_51, name)
        gt_path = os.path.join(path.RAINE_ORGAN_MANUAL_GTS_51, name)
        # load image and gt
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        gt = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt)
        gt_array = np.uint8(gt_array==args.label_id)

        # normalize image
        img_norm = min_max_norm_3dimg(img_array) 

        # preprocess the image and gt for SAM
        imgs, gts, img_embeddings, label_index = sam_preprocess(args.filter, args.view, img_norm, gt_array, args.image_size, sam_model, device, args.post_norm)
        sub_index[name] = label_index
        # stack the list to array then save to npz file
        if len(imgs)>1:
            imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
            gts = np.stack(gts, axis=0) # (n, 256, 256)
            img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
            print(name, 'imgs shape', imgs.shape, '\tgts shape', gts.shape, '\timg_embeddings shape', img_embeddings.shape)
            np.savez_compressed(os.path.join(save_path, name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts, img_embeddings=img_embeddings)
            # save an example image for sanity check
            idx = np.random.randint(0, imgs.shape[0])
            img_idx = imgs[idx,:,:,:]
            gt_idx = gts[idx,:,:]
            bd = segmentation.find_boundaries(gt_idx, mode='inner')
            img_idx[bd, :] = [255, 0, 0]
            io.imsave(save_path + '.png', img_idx, check_contrast=False)

    with open(os.path.join(save_path, 'sub_index.json'), 'w') as f:
        json.dump(sub_index, f, indent=4)
