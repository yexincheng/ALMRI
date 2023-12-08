import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from metrics import compute_dice_coefficient
import wandb
from utils import NpzDataset
from torch.utils.data import DataLoader
import argparse

# batch inference
def infer(imgs, embeddings, ckp_path, model_type, device):
    # infer on fine-tuned sam model
    bboxes = []
    # bbox = get_bbox_from_mask(imgs)
    bbox = np.array([0 ,0, imgs.shape[2], imgs.shape[1]])
    bboxes.append(bbox)
    
    # predict the segmentation mask using the fine-tuned model
    sam_model = sam_model_registry[model_type](checkpoint=ckp_path).to(device)

    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    H, W = imgs.shape[2], imgs.shape[1] # (B, H, W, C)
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

def log_image_table(images, preds, gts, id, dice, table):
    """Log a wandb.Table with (img, pred, gts, scores)
    Args:
        images: images (B, H, W, 3)
        preds: predicted masks (B, H, W)
        gts: ground truth masks (B, H, W)
    """
    # print(f'{id}: {dice}')
    index = len(images) // 2 # log the middle slice
    table.add_data(wandb.Image(images[index][:, :, 0]), wandb.Image(preds[index, :, :,]), wandb.Image(gts[index, :, :,]), id, dice)
    


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='./datasets/RAINE_organ_51', help='dataset path prefix')
    parser.add_argument('--task', type=str, default='MRI_Pancreas', help='task name')
    parser.add_argument('--label_id', type=int, default=3, help='label id')
    parser.add_argument('--strategy', type=str, default='random', help='sampling strategy')
    parser.add_argument('--base_model', type=str, default='SAM', help='base model name')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--wandb', type=bool, default=False, help='log to wandb')
    parser.add_argument('--mode', type=str, default='eval', help='mode: eval or train')
    args = parser.parse_args()

    save_path_ckp = os.path.join('./checkpoints/', args.base_model + '_' + args.task + '_' + args.strategy)
    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckp_paths = sorted(glob(os.path.join(save_path_ckp, f'epochs{args.num_epochs}', '*latest.pth')))
    testing_pool_path = os.path.join(args.prefix, args.task, 'test')
    testing_pool = glob(os.path.join(testing_pool_path, '*.npz'))
    dice_test = []

    if args.wandb:
        wandb.login(key='4aaa2e71cdec13a78a42c6ceac38dd0c7235a131', relogin=True)
        print('wandb login')
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            # reinit=True,
            # Set the project where this run will be logged
            project="SAM-Activelearning", 
            group=f'{args.strategy}',
            name=f'{args.base_model}_{args.sam_model_type}_{args.task}_{args.strategy}_epochs{args.num_epochs}',
            config = {
                "task": args.task,
                'label_id': args.label_id,
                'base_model': args.base_model,
                "model": args.sam_model_type,
                "strategy": args.strategy,
                "num_epochs": args.num_epochs
            }
        )
        table = wandb.Table(columns=["image", "pred", "gt", 'id', 'dice'])

    # eval on test set for each checkpoint
    print('Evaluating on test set for {} checkpoints'.format(len(ckp_paths)))
    for p in tqdm(ckp_paths):
        avg_dice = []
        for t in tqdm(testing_pool):
            id = os.path.basename(t).split('.')[0]
            imgs = np.load(t)['imgs']

            dataset = NpzDataset([t])
            batch_size = len(dataset)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # make sure batch size is larger than sample slices
            for embeddings, gts, _ in dataloader:
                preds, _ = infer(imgs, embeddings, p, args.sam_model_type, device)
            gts = gts[:, 0, :, :].numpy() # convert tensor to numpy (B, H, W)
            dice = compute_dice_coefficient(gts, preds)
            avg_dice.append(dice)
            if '40' in p and args.wandb:
                preds_int = preds * 1 # convert boolean to int
                log_image_table(imgs, preds_int, gts, id, dice, table)
        dice_test.append(np.mean(avg_dice))
        if args.wandb:
            wandb.log({f"Eval/Dice {args.strategy} epoch{args.num_epochs}": np.mean(avg_dice)}, commit=True)
    if args.wandb:
        wandb.log({f"predictions_table {args.strategy} epoch{args.num_epochs}":table}, commit=True)
        wandb.finish()

    # plot dice curve
    # x = list(range(len(dice_test)))
    plt.plot(dice_test)
    plt.xlabel('Number of samples')
    plt.ylabel('Dice coefficient')
    plt.title(f'{args.base_model} {args.task} {args.strategy} epochs{args.num_epochs} sampling')
    plt.savefig(os.path.join(f'figures/{args.mode}', f'{args.base_model}_{args.task}_{args.strategy}_epochs{args.num_epochs}_sampling.png'))
    plt.close()
