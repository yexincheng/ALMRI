import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

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

def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum
 
def infer(imgs, ckp_path, model_type, device):
    # infer on fine-tuned sam model
    medsam_segs = []
    bboxes = []
    for img in imgs:
        # bbox = get_bbox_from_mask(gt)
        bbox = np.array([0 ,0, img.shape[0], img.shape[1]])
        bboxes.append(bbox)
        
        # predict the segmentation mask using the fine-tuned model
        sam_model = sam_model_registry[model_type](checkpoint=ckp_path).to(device)

        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        H, W = img.shape[:2]
        resize_img = sam_trans.apply_image(img)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
            # convert box to 1024x1024 grid
            bbox = sam_trans.apply_boxes(bbox, (H, W))
            box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            medsam_seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
                )
            medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            medsam_segs.append(medsam_seg)
    return np.stack(medsam_segs, axis=0), bboxes


prefix = './datasets/RAINE_organ_51'
task = 'MRI_Pancreas'
#SAM MODEL TYPE 
sam_model_type = 'vit_b'
# sample strategy
strategy = 'random'
# active learning fine tunning checkpoint
base_model = 'SAM'
save_path_ckp = os.path.join('./checkpoints/', base_model + '_' + task + '_' + strategy)
# device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckp_paths = sorted(glob(os.path.join(save_path_ckp, '*best.pth')))
testing_pool_path = os.path.join(prefix, task, 'test')
testing_pool = glob(os.path.join(testing_pool_path, '*.npz'))
dice_test = []

for p in tqdm(ckp_paths):
    avg_dice = []
    for t in testing_pool:
        imgs, gts = np.load(t)['imgs'], np.load(t)['gts']
        pre, _ = infer(imgs, p, sam_model_type, device)
        dice = compute_dice_coefficient(gts, pre)
        avg_dice.append(dice)
    dice_test.append(np.mean(avg_dice))

# x = list(range(len(dice_test)))
plt.plot(dice_test)
plt.xlabel('Number of samples')
plt.ylabel('Dice coefficient')
plt.title(f'{base_model} {task} {strategy} sampling')
plt.savefig(f'{base_model}_{task}_{strategy}_sampling.png')
plt.close()
