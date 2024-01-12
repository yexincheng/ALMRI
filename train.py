import datasets.path as path
# import nibabel as nib
from glob import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import monai
# from utils.SurfaceDice import compute_dice_coefficient
from tqdm import tqdm
import json
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import argparse
from utils import NpzDataset
import wandb
from strategy import random_sampling



if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='./datasets/RAINE_organ_51', help='dataset path prefix')
    parser.add_argument('--task', type=str, default='MRI_Pancreas', help='task name')
    parser.add_argument('--label_id', type=int, default=3, help='label id')
    parser.add_argument('--strategy', type=str, default='random', help='sampling strategy')
    parser.add_argument('--base_model', type=str, default='SAM', help='base model name')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/SAM/sam_vit_b_01ec64.pth', help='SAM checkpoint for fine tunning')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for fine tunning')
    parser.add_argument('--wandb', type=bool, default=False, help='log to wandb')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    args = parser.parse_args()
    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # active learning fine tunning checkpoint
    save_path_ckp = os.path.join('./checkpoints/', args.base_model + '_' + args.task + '_' + args.strategy)
    os.makedirs(save_path_ckp, exist_ok=True)
    # sampling dataset path
    sampling_datapath = os.path.join(args.prefix, args.task, args.base_model+'_sampling')
    os.makedirs(sampling_datapath, exist_ok=True)

    training_pool_path = os.path.join(args.prefix, args.task, 'train')
    training_pool = glob(os.path.join(training_pool_path, '*.npz'))
    sample_pool = []
    batch_size = 16 # make sure batch size is smaller than sample slices
    losses = []
    best_loss = 1e10
    sam_model = sam_model_registry[args.sam_model_type](checkpoint=args.checkpoint).to(device)
    sam_model.train()
    # Set up the optimizer, hyperparameter tuning will improve performance here
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    save_path_ckp_seed_epoch = os.path.join(save_path_ckp, f'seed{args.seed}', f'epochs{args.num_epochs}')
    os.makedirs(save_path_ckp_seed_epoch, exist_ok=True)

    # wandb
    if args.wandb:
        wandb.login(key='4aaa2e71cdec13a78a42c6ceac38dd0c7235a131', relogin=True, settings=wandb.Settings(_service_wait=300))
        wandb.init(
            project="SAM-Activelearning", 
            group="random",
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

    for i in range(len(training_pool)):
        sample_pool, training_pool = random_sampling(training_pool, sample_pool, args.seed)

        num_samples = len(sample_pool)
        if num_samples > 2 and num_samples <= 5:
            batch_size = 32
        elif num_samples > 5:
            batch_size = 64
        
        print('Number of samples: ', num_samples, '\tBatch size: ', batch_size)
        sample_dataset = NpzDataset(sample_pool)
        sample_dataloader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(sample_dataloader)):
                # img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4)
                # print(f"{image_embedding.shape=}, {gt2D.shape=}, {boxes.shape=}")
                with torch.no_grad():
                    box_np = boxes.numpy() # [0, 0, 256, 256]
                    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
                    box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                    if len(box_torch.shape) == 2:
                        box_torch = box_torch[:, None, :] # (B, 1, 4)
                    # get prompt embeddings 
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )
                # predicted masks
                # print(f"{image_embedding.shape=}, {sparse_embeddings.shape=}, {dense_embeddings.shape=}")
                mask_predictions, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                    image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                    multimask_output=False,
                )

                loss = seg_loss(mask_predictions, gt2D.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= step
            losses.append(epoch_loss)
            print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
            if args.wandb:
                wandb.log({f"Train/loss of {num_samples} num sample": epoch_loss})
            # save the latest model checkpoint
            torch.save(sam_model.state_dict(), os.path.join(save_path_ckp_seed_epoch, f'{args.base_model}_{args.sam_model_type}_{num_samples:02d}_latest.pth'))
            # save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(sam_model.state_dict(), os.path.join(save_path_ckp_seed_epoch, f'{args.base_model}_{args.sam_model_type}_{num_samples:02d}_best.pth'))
        
        # save loss
        sace_path_loss = os.makedirs(os.path.join('results/loss', f'epochs{args.epochs}'), exist_ok=True)
        np.save(os.path.join(sace_path_loss, f'{args.strategy}_{args.base_model}_{args.sam_model_type}_{num_samples:02d}_{args.seed}_train_loss.npy'), np.array(losses))
        # plot loss
        plt.plot(losses)
        plt.title('Dice + Cross Entropy Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.show() # comment this line if you are running on a server
        plt.savefig(os.path.join(save_path_ckp_seed_epoch, f'{args.base_model}_{args.sam_model_type}_{num_samples:02d}_train_loss.png'))
        plt.close()

    with open(os.path.join(sampling_datapath, f'{args.strategy}_{args.num_epochs}_{args.seed}pool.json'), 'w') as f:
        json.dump(sample_pool, f, indent=4)

    if args.wandb:
        # wandb.save(os.path.join(sampling_datapath, f'{args.strategy}_pool.json'))
        wandb.finish()