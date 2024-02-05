# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import distutils.util
import logging
import os
import sys
import time
from glob import glob

import torch
import torch.distributed as dist
from monai.apps.deepedit.interaction import Interaction

from monai.apps.deepedit.transforms import (
    AddGuidanceSignalDeepEditd,
    AddRandomGuidanceDeepEditd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    SplitPredsLabeld,
)
from monai.data import partition_dataset, list_data_collate, Dataset,pad_list_data_collate
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset
from monai.engines import SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    from_engine,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    ToNumpyd,
    ToTensord,
    CenterSpatialCropd,
)
from monai.utils import set_determinism
import json
from strategy import random_sampling
import numpy as np
import matplotlib.pyplot as plt
# from ignite.engine import _prepare_batch


def get_network(labels):
    # Network
    network = DynUNet(
        spatial_dims=3,
        in_channels=len(labels) + 1,
        out_channels=len(labels),
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )
    return network


def get_pre_transforms(labels, spatial_size):
    t = [
        LoadImaged(keys=("image", "label"), reader="ITKReader"),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),

        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        CenterSpatialCropd(keys=["image", "label"],roi_size=spatial_size),
        # Transforms for click simulation
        # FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),	
        # AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        # AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)

def get_click_transforms():
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        ToNumpyd(keys=("image", "label", "pred")),
        # Transforms for click simulation
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditd(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)


def get_post_transforms(labels):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)


def get_loaders(args, nii_pool, pre_transforms):


    # data path datasets/RAINE_organ_51/trainingset_51/
    imagelist = [os.path.join(args.prefix, 'trainingset_51', image_name) for image_name in nii_pool]
    labellist = [os.path.join(args.prefix, 'trainingset_51', 'labels', 'final', image_name) for image_name in nii_pool]
    # imagelist = glob(os.path.join(args.prefix, 'trainingset_51', '*.nii.gz'))
    # labellist = glob(os.path.join(args.prefix, 'trainingset_51', 'labels', 'final', '*.nii.gz'))
    print('len(imagelist)', len(imagelist), 'len(labellist)', len(labellist))
    datalist = [{"image": image_name, "label": label_name} for image_name, label_name in zip(imagelist, labellist)]
    print(datalist[-1])
    total_l = len(datalist)
    # if len(datalist) <= 3:
    #     batch_size = len(datalist)
    # else:
    #     batch_size = 3

    # Check the length of the elements in datalist
    # for data in datalist:
    #     print(len(data["image"]), len(data["label"]))

    # # Check the output of pre_transforms
    # for data in datalist:
    #     transformed = pre_transforms(data)
    #     print(transformed["image"].shape, transformed["label"].shape, '\n label value', np.unique(transformed["label"]))

    # print('len(nii_pool)', len(nii_pool), '\tbatch_size', batch_size)
    train_ds = Dataset(datalist, pre_transforms)
    print('dataset length', len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    for batch in train_loader:
        print('batch', batch["image"].shape, batch["label"].shape)
        
    logging.info("Total Records used for Training is: {}/{}".format(len(train_ds), total_l))

    return train_loader

def prepare_batch(batch, device=None, non_blocking=False):
    return _prepare_batch((batch["image"], batch["label"]), device, non_blocking)

def create_trainer(args, sample_pool_nii):

    set_determinism(seed=args.seed)

    device = torch.device("cuda" if args.use_gpu else "cpu")

    pre_transforms = get_pre_transforms(args.labels, args.spatial_size)
    click_transforms = get_click_transforms()
    post_transform = get_post_transforms(args.labels)

    train_loader = get_loaders(args, sample_pool_nii, pre_transforms)

    # define training components
    network = get_network(args.labels).to(device)
    if args.resume:
        logging.info("Loading Network...")
        # map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        network.load_state_dict(torch.load(args.model_filepath))

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        # ValidationHandler(validator=evaluator, interval=args.val_freq, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),

    ]

    all_train_metrics = dict()
    all_train_metrics["train_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    for key_label in args.labels:
        if key_label != "background":
            all_train_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.num_epochs,
        train_data_loader=train_loader,
        network=network,
        # iteration_update=Interaction(
        #     deepgrow_probability=args.deepgrow_probability_train,
        #     transforms=click_transforms,
        #     click_probability_key="probability",
        #     train=True,
        #     label_names=args.labels,
        # ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )
    return trainer


def run(args):
    

    training_pool_path = os.path.join(args.prefix, args.task, 'train')
    training_pool = glob(os.path.join(training_pool_path, '*.npz'))
    sample_pool = []
    
    for i in range(len(training_pool)):
        sample_pool, training_pool = random_sampling(training_pool, sample_pool, args.seed)
        num_samples = len(sample_pool)
        # base name sample list
        sample_pool_nii = [(os.path.basename(sample).split('.npz')[0]+'.nii.gz') for sample in sample_pool]

       
        trainer = create_trainer(args, sample_pool_nii)

        start_time = time.time()
        torch.cuda.empty_cache()

        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        trainer.run()
        end_time = time.time()

        logging.info("Total Training Time {}".format(end_time - start_time))
        save_path_ckp_seed_epoch =  os.path.join('./checkpoints/', args.base_model + '_' + args.task + '_' + args.strategy, f'seed{args.seed}', f'epochs{args.num_epochs}')
        os.makedirs(save_path_ckp_seed_epoch, exist_ok=True)
        torch.save(
            trainer.network.state_dict(), os.path.join(save_path_ckp_seed_epoch, f'{args.base_model}_{num_samples:02d}_latest.pth')
        )
        if not args.multi_gpu:
            model_ts = torch.jit.script(trainer.network)
            torch.jit.save(model_ts, os.path.join(save_path_ckp_seed_epoch, f'{args.base_model}_{num_samples:02d}_latest.ts'))

        if args.multi_gpu:
            dist.destroy_process_group()
        break

        # save loss
        # print(trainer.state.metrics.keys())
        # print(len(trainer.state.metrics), trainer.state.metrics['train_loss'])
        # losses = [stat["train_loss"] for stat in trainer.state.metrics]
        # save_path_loss = os.path.join('results/loss', f'epochs{args.num_epochs}')
        # os.makedirs(save_path_loss, exist_ok=True)
        # np.save(os.path.join(save_path_loss, f'{args.strategy}_{args.base_model}_{num_samples:02d}_{args.seed}_train_loss.npy'), np.array(losses))
        # # plot loss
        # plt.plot(losses)
        # plt.title('Dice + Cross Entropy Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # # plt.show() # comment this line if you are running on a server
        # plt.savefig(os.path.join(save_path_ckp_seed_epoch, f'{args.base_model}_{num_samples:02d}_train_loss.png'))
        # plt.close()

    # save sample pool as json
    sampling_datapath = os.path.join(args.prefix, args.task, args.base_model+'_sampling')
    os.makedirs(sampling_datapath, exist_ok=True)
    with open(os.path.join(sampling_datapath, f'{args.strategy}_{args.num_epochs}_{args.seed}pool.json'), 'w') as f:
        json.dump(sample_pool, f, indent=4)


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument("-a", "--amp", type=strtobool, default="false")

    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("-r", "--resume", type=strtobool, default="false")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    # parser.add_argument("-it", "--max_train_interactions", type=int, default=15)
    # parser.add_argument("-iv", "--max_val_interactions", type=int, default=5)

    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=0.4)
    # parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)

    parser.add_argument("--multi_gpu", type=strtobool, default="false")
    parser.add_argument("--size", nargs=3, type=int)
    
    parser.add_argument('--prefix', type=str, default='./datasets/RAINE_organ_51', help='dataset path prefix')
    parser.add_argument('--task', type=str, default='MRI_Pancreas', help='task name')
    parser.add_argument('--strategy', type=str, default='random', help='sampling strategy')
    parser.add_argument('--base_model', type=str, default='DynUNet', help='base model name')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    args = parser.parse_args()
    args.spatial_size = (args.size)
    # For single label using one of the Medical Segmentation Decathlon
#     args.labels = {"spleen": 1, "background": 0}

    # # For multiple label using the BTCV dataset (https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
    # # For this, remember to update accordingly the function 'get_loaders' in lines 151-152
    args.labels = {
        "left kidney": 1,
        # "right kidney": 2,
        # "pancreas": 3,
        "background": 0,

        }
    if args.resume:
        logging.info("Resuming Training...")
        args.model_filepath = None
    
    run(args)


if __name__ == "__main__":
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


    main()

