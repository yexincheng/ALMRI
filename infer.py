import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils import compute_dice_coefficient, get_bbox_from_mask
from metrics import compute_dice_coefficient
import wandb

# batch inference
def infer(gts, embeddings, ckp_path, model_type, device):
    # infer on fine-tuned sam model
    bboxes = []
    # bbox = get_bbox_from_mask(gts)
    bbox = np.array([0 ,0, gts.shape[-1], gts.shape[-2]])
    bboxes.append(bbox)
    
    # predict the segmentation mask using the fine-tuned model
    sam_model = sam_model_registry[model_type](checkpoint=ckp_path).to(device)

    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    H, W = gts.shape[2:]
    with torch.no_grad():
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
            image_embeddings=embeddings.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg_prob = np.stack(medsam_seg_prob, axis=0)
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg, bboxes


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
num_epochs = 100
label_id = 3
ckp_paths = sorted(glob(os.path.join(save_path_ckp, '*latest.pth')))
testing_pool_path = os.path.join(prefix, task, 'test')
testing_pool = glob(os.path.join(testing_pool_path, '*.npz'))
dice_test = []

#
#wandb.logi(key='4aaa2e71cdec13a78a42c6ceac38dd0c7235a131', relogin=True)
#wandb.init(
    # Set the project where this run will be logged
#    project="SAM-Activelearning", 
#    group="random",
#    name=f'{base_model}_{sam_model_type}_{task}_{strategy}_epoch{num_epochs}',
#    config = {
#        "task": task,
#        'label_id': label_id,
#        'base_model': base_model,
#        "model": sam_model_type,
#        "strategy": strategy,
#        "num_epochs": num_epochs
#    }
#)
#table = wandb.Table(columns=["image", "pred", "gt", 'id', 'dice'])


for p in tqdm(ckp_paths):
    avg_dice = []
    for t in testing_pool:
        imgs, gts = np.load(t)['imgs'], np.load(t)['gts']
        pre, _ = infer(imgs, p, sam_model_type, device)
        dice = compute_dice_coefficient(gts, pre)
        avg_dice.append(dice)
    print(f'Dice for Subject {t} on Model {p}: ', dice)
    dice_test.append(np.mean(avg_dice))
    print(f'Average Dice for Model {p}: ', np.mean(avg_dice))

# x = list(range(len(dice_test)))
plt.plot(dice_test)
plt.xlabel('Number of samples')
plt.ylabel('Dice coefficient')
plt.title(f'{base_model} {task} {strategy} epochs{num_epochs} sampling')
plt.savefig(f'{base_model}_{task}_{strategy}_epochs{num_epochs}_sampling.png')
plt.close()
