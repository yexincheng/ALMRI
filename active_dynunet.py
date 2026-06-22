"""
Unified DynUNet training for active learning and sequential finetuning.

Modes (subcommands):
  active      Active learning on trainingset_45 (random or epistemic entropy).
              No training until 5 subjects; 100 epochs at 5 subjects, then 30 per
              new subject. Milestone checkpoints at 35, 40, and 45 subjects.

  finetune49  Finetune from the 45-subject entropy-active checkpoint on
              trainingset_49. Adds four subjects in fixed order; 30 epochs each.
              Milestone checkpoints at 46, 47, 48, and 49 subjects.

  finetune51  Finetune from the 49-subject finetune checkpoint on trainingset_51.
              Adds two subjects in fixed order; 30 epochs each.
              Milestone checkpoints at 50 and 51 subjects.
"""
import argparse
import distutils.util
import json
import logging
import os
import sys
import time
from glob import glob
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    NormalizeLabelsInDatasetd,
)
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
)
from monai.utils import set_determinism

from strategy import entropy_sampling, random_sampling, sequential_sampling

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_SUFFIX = ".FatImaging_W.nii.gz"

LABELS = {
    "left kidney": 1,
    "right kidney": 2,
    "pancreas": 3,
    "background": 0,
}
NUMBER_INTENSITY_CH = 1
IN_CHANNELS = NUMBER_INTENSITY_CH + len(LABELS)
OUT_CHANNELS = len(LABELS)

INITIAL_SUBJECTS = 5
INITIAL_EPOCHS = 100
INCREMENTAL_EPOCHS = 30

ACTIVE_DEFAULTS = {
    "image_dir": os.path.join(SCRIPT_DIR, "trainingset_45"),
    "label_dir": os.path.join(SCRIPT_DIR, "trainingset_45", "labels", "final"),
    "ckpt_dir_random": os.path.join(
        SCRIPT_DIR, "checkpoints", "DynUNet_trainingset45_random_active"
    ),
    "ckpt_dir_entropy": os.path.join(
        SCRIPT_DIR, "checkpoints", "DynUNet_trainingset45_entropy_active"
    ),
    "milestone_saves": (35, 40, 45),
    "loss_subdir": "trainingset45_active",
    "sampling_subdir": "trainingset45_active",
}

FINETUNE49_DEFAULTS = {
    "image_dir": os.path.join(SCRIPT_DIR, "datasets", "trainingset_49"),
    "label_dir": os.path.join(SCRIPT_DIR, "datasets", "trainingset_49", "labels", "final"),
    "checkpoint": os.path.join(
        SCRIPT_DIR,
        "checkpoints",
        "DynUNet_trainingset45_entropy_active",
        "seed2025",
        "init100_inc30",
        "DynUNet_45subjects_seed2025.pt",
    ),
    "ckpt_dir": os.path.join(
        SCRIPT_DIR, "checkpoints", "DynUNet_trainingset49_entropy_finetune"
    ),
    "new_subject_ids": ("sub-41810", "sub-16880", "sub-12400", "sub-14870"),
    "base_subjects": 45,
    "milestone_saves": (46, 47, 48, 49),
    "loss_subdir": "trainingset49_finetune",
    "sampling_subdir": "trainingset49_finetune",
}

FINETUNE51_DEFAULTS = {
    "image_dir": os.path.join(SCRIPT_DIR, "datasets", "trainingset_51"),
    "label_dir": os.path.join(SCRIPT_DIR, "datasets", "trainingset_51", "labels", "final"),
    "checkpoint": os.path.join(
        SCRIPT_DIR,
        "checkpoints",
        "DynUNet_trainingset49_entropy_finetune",
        "seed2025",
        "inc30",
        "DynUNet_49subjects_seed2025.pt",
    ),
    "ckpt_dir": os.path.join(
        SCRIPT_DIR, "checkpoints", "DynUNet_trainingset51_entropy_finetune"
    ),
    "new_subject_ids": ("sub-52220", "sub-50710"),
    "base_subjects": 49,
    "milestone_saves": (50, 51),
    "loss_subdir": "trainingset51_finetune",
    "sampling_subdir": "trainingset51_finetune",
}


def epochs_for_pool_size(
    num_subjects: int, initial_subjects: int, initial_epochs: int, incremental_epochs: int
) -> int:
    if num_subjects < initial_subjects:
        return 0
    if num_subjects == initial_subjects:
        return initial_epochs
    return incremental_epochs


def get_network():
    return DynUNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )


def get_train_transforms(labels, spatial_size):
    return Compose(
        [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeLabelsInDatasetd(keys="label", label_names=labels),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            AddGuidanceFromPointsDeepEditd(
                ref_image="image", guidance="guidance", label_names=labels
            ),
            CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
            AddGuidanceSignalDeepEditd(
                keys="image",
                guidance="guidance",
                number_intensity_ch=NUMBER_INTENSITY_CH,
            ),
            ToTensord(keys=("image", "label")),
        ]
    )


def get_infer_transforms(labels, spatial_size):
    """Inference transforms for epistemic entropy scoring (image only)."""
    return Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            AddGuidanceFromPointsDeepEditd(
                ref_image="image", guidance="guidance", label_names=labels
            ),
            CenterSpatialCropd(keys="image", roi_size=spatial_size),
            AddGuidanceSignalDeepEditd(
                keys="image",
                guidance="guidance",
                number_intensity_ch=NUMBER_INTENSITY_CH,
            ),
            ToTensord(keys="image"),
        ]
    )


def get_all_subjects(image_dir: str):
    subjects = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
    if not subjects:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return subjects


def get_train_loader(label_dir, subject_paths, transforms):
    datalist = [
        {
            "image": image_path,
            "label": os.path.join(label_dir, os.path.basename(image_path)),
        }
        for image_path in subject_paths
    ]
    train_ds = Dataset(data=datalist, transform=transforms)
    return DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def train_epochs(network, train_loader, optimizer, loss_fn, device, num_epochs):
    losses = []
    network.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in train_loader:
            imgs = data["image"].to(device)
            gts = data["label"].to(device)
            if imgs.shape[1] != IN_CHANNELS:
                raise RuntimeError(
                    f"Expected {IN_CHANNELS} input channels, got {imgs.shape[1]}."
                )
            optimizer.zero_grad()
            outputs = network(imgs)
            loss = loss_fn(outputs, gts)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(train_loader), 1)
        losses.append(epoch_loss)
        logging.info("Epoch %d loss: %.4f", epoch + 1, epoch_loss)
    return losses


def save_checkpoint(network, ckpt_dir, num_subjects, seed):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir, f"DynUNet_{num_subjects}subjects_seed{seed}.pt"
    )
    torch.save(network.state_dict(), ckpt_path)
    logging.info("Saved checkpoint: %s", ckpt_path)
    return ckpt_path


def setup_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logging.info("Created directory: %s", path)


def subject_id_to_path(image_dir: str, subject_id: str) -> str:
    case = (
        subject_id
        if subject_id.endswith(".nii.gz")
        else f"{subject_id}{CASE_SUFFIX}"
    )
    path = os.path.join(image_dir, case)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def build_initial_pool(
    image_dir: str, new_subject_ids: Sequence[str], base_subjects: int
) -> list[str]:
    new_cases = {
        subject_id if subject_id.endswith(".nii.gz") else f"{subject_id}{CASE_SUFFIX}"
        for subject_id in new_subject_ids
    }
    subjects = sorted(
        path
        for path in glob(os.path.join(image_dir, "*.nii.gz"))
        if os.path.basename(path) not in new_cases
    )
    if len(subjects) != base_subjects:
        raise RuntimeError(
            f"Expected {base_subjects} base subjects in {image_dir}, "
            f"found {len(subjects)}."
        )
    return subjects


def build_new_subject_paths(image_dir: str, new_subject_ids: Sequence[str]) -> list[str]:
    return [subject_id_to_path(image_dir, subject_id) for subject_id in new_subject_ids]


def run_active(args):
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not os.path.isdir(args.label_dir):
        raise FileNotFoundError(f"Label directory not found: {args.label_dir}")

    strategy_tag = args.strategy
    ckpt_root = args.ckpt_dir or (
        ACTIVE_DEFAULTS["ckpt_dir_entropy"]
        if args.strategy == "entropy"
        else ACTIVE_DEFAULTS["ckpt_dir_random"]
    )
    ckpt_dir = os.path.join(
        ckpt_root,
        f"seed{args.seed}",
        f"init{args.initial_epochs}_inc{args.incremental_epochs}",
    )
    loss_dir = os.path.join(SCRIPT_DIR, "results", "loss", ACTIVE_DEFAULTS["loss_subdir"], strategy_tag)
    sampling_dir = os.path.join(
        SCRIPT_DIR, "results", "sampling", ACTIVE_DEFAULTS["sampling_subdir"], strategy_tag
    )
    setup_dirs(ckpt_dir, loss_dir, sampling_dir)

    all_subjects = get_all_subjects(args.image_dir)
    training_pool = sorted(list(all_subjects))
    sample_pool = []

    train_transforms = get_train_transforms(LABELS, args.spatial_size)
    infer_transforms = get_infer_transforms(LABELS, args.spatial_size)

    def prepare_infer_batch(image_path: str):
        sample = infer_transforms({"image": image_path})
        if not isinstance(sample["image"], torch.Tensor):
            sample["image"] = torch.as_tensor(sample["image"])
        if sample["image"].ndim == 4:
            sample["image"] = sample["image"].unsqueeze(0)
        return sample

    network = get_network().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)

    logging.info(
        "Active learning | strategy=%s | %d subjects | initial=%d@%dep | incremental=%dep",
        args.strategy,
        len(all_subjects),
        args.initial_subjects,
        args.initial_epochs,
        args.incremental_epochs,
    )

    milestone_saves = ACTIVE_DEFAULTS["milestone_saves"]
    for step_idx in range(len(all_subjects)):
        if args.strategy == "random":
            sample_pool, training_pool = random_sampling(
                training_pool, sample_pool, args.seed
            )
        elif args.strategy == "entropy":
            if len(sample_pool) < args.initial_subjects:
                logging.info(
                    "Entropy bootstrap (%d/%d): sequential subject order.",
                    len(sample_pool) + 1,
                    args.initial_subjects,
                )
                sample_pool, training_pool = sequential_sampling(
                    training_pool, sample_pool
                )
            else:
                logging.info(
                    "Scoring unlabeled pool with MONAI Label epistemic entropy..."
                )
                sample_pool, training_pool, entropy_scores = entropy_sampling(
                    training_pool,
                    sample_pool,
                    network,
                    device,
                    prepare_infer_batch,
                    num_mc_samples=args.num_mc_samples,
                )
                scores_path = os.path.join(
                    sampling_dir,
                    f"entropy_scores_step{step_idx + 1:02d}.json",
                )
                with open(scores_path, "w") as f:
                    json.dump(
                        {os.path.basename(k): v for k, v in entropy_scores.items()},
                        f,
                        indent=2,
                    )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        num_subjects = len(sample_pool)
        subject_path = sample_pool[-1]
        logging.info(
            "=== Step %d/%d | %d subject(s) | added: %s ===",
            step_idx + 1,
            len(all_subjects),
            num_subjects,
            os.path.basename(subject_path),
        )

        num_epochs = epochs_for_pool_size(
            num_subjects,
            args.initial_subjects,
            args.initial_epochs,
            args.incremental_epochs,
        )
        if num_epochs == 0:
            logging.info(
                "Skipping training (%d/%d); %d epochs at %d subjects.",
                num_subjects,
                args.initial_subjects,
                args.initial_epochs,
                args.initial_subjects,
            )
            continue

        logging.info("Training %d epochs on %d subjects.", num_epochs, num_subjects)
        train_loader = get_train_loader(args.label_dir, sample_pool, train_transforms)

        start_time = time.time()
        losses = train_epochs(
            network, train_loader, optimizer, loss_fn, device, num_epochs
        )
        logging.info("Training time: %.1fs", time.time() - start_time)

        pool_path = os.path.join(
            sampling_dir,
            f"{strategy_tag}_seed{args.seed}_pool_{num_subjects:02d}.json",
        )
        with open(pool_path, "w") as f:
            json.dump([os.path.basename(p) for p in sample_pool], f, indent=2)

        np.save(
            os.path.join(
                loss_dir,
                f"{strategy_tag}_seed{args.seed}_{num_epochs}ep_{num_subjects:02d}subjects_loss.npy",
            ),
            np.array(losses),
        )

        if num_subjects in milestone_saves:
            save_checkpoint(network, ckpt_dir, num_subjects, args.seed)
            plt.figure()
            plt.plot(losses)
            plt.title(f"Loss ({num_subjects} subjects, {strategy_tag})")
            plt.xlabel("Epoch")
            plt.ylabel("DiceCE loss")
            plt.savefig(
                os.path.join(
                    ckpt_dir,
                    f"DynUNet_{num_subjects}subjects_seed{args.seed}_loss.png",
                )
            )
            plt.close()

    order_path = os.path.join(
        sampling_dir, f"{strategy_tag}_seed{args.seed}_acquisition_order.json"
    )
    with open(order_path, "w") as f:
        json.dump([os.path.basename(p) for p in sample_pool], f, indent=2)
    logging.info("Acquisition order saved to %s", order_path)
    logging.info("Finished. Checkpoints in %s", ckpt_dir)


def run_finetune(args, config: dict):
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")
    if not os.path.isdir(args.label_dir):
        raise FileNotFoundError(f"Label dir not found: {args.label_dir}")

    ckpt_dir = os.path.join(
        args.ckpt_dir,
        f"seed{args.seed}",
        f"inc{args.incremental_epochs}",
    )
    loss_dir = os.path.join(SCRIPT_DIR, "results", "loss", config["loss_subdir"])
    meta_dir = os.path.join(SCRIPT_DIR, "results", "sampling", config["sampling_subdir"])
    setup_dirs(ckpt_dir, loss_dir, meta_dir)

    new_subject_ids = config["new_subject_ids"]
    sample_pool = build_initial_pool(
        args.image_dir, new_subject_ids, config["base_subjects"]
    )
    new_subject_paths = build_new_subject_paths(args.image_dir, new_subject_ids)

    for subject_path in sample_pool + new_subject_paths:
        label_path = os.path.join(args.label_dir, os.path.basename(subject_path))
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")

    train_transforms = get_train_transforms(LABELS, args.spatial_size)
    network = get_network().to(device)
    network.load_state_dict(torch.load(args.checkpoint, map_location=device))
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)

    logging.info("Loaded checkpoint: %s", args.checkpoint)
    logging.info("Base pool: %d subjects from %s", len(sample_pool), args.image_dir)
    logging.info(
        "Adding %d subjects in order: %s",
        len(new_subject_paths),
        ", ".join(new_subject_ids),
    )
    logging.info(
        "%d epochs per new subject (%d epochs total)",
        args.incremental_epochs,
        args.incremental_epochs * len(new_subject_paths),
    )

    milestone_saves = config["milestone_saves"]
    for step_idx, subject_path in enumerate(new_subject_paths, start=1):
        sample_pool.append(subject_path)
        num_subjects = len(sample_pool)
        logging.info(
            "=== Step %d/%d | %d subjects | added: %s ===",
            step_idx,
            len(new_subject_paths),
            num_subjects,
            os.path.basename(subject_path),
        )

        train_loader = get_train_loader(args.label_dir, sample_pool, train_transforms)

        start_time = time.time()
        losses = train_epochs(
            network,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.incremental_epochs,
        )
        logging.info("Training time: %.1fs", time.time() - start_time)

        pool_path = os.path.join(
            meta_dir, f"finetune_seed{args.seed}_pool_{num_subjects:02d}.json"
        )
        with open(pool_path, "w") as f:
            json.dump([os.path.basename(p) for p in sample_pool], f, indent=2)

        np.save(
            os.path.join(
                loss_dir,
                f"finetune_seed{args.seed}_{args.incremental_epochs}ep_{num_subjects:02d}subjects_loss.npy",
            ),
            np.array(losses),
        )

        if num_subjects in milestone_saves:
            save_checkpoint(network, ckpt_dir, num_subjects, args.seed)
            plt.figure()
            plt.plot(losses)
            plt.title(f"Loss ({num_subjects} subjects, finetune)")
            plt.xlabel("Epoch")
            plt.ylabel("DiceCE loss")
            plt.savefig(
                os.path.join(
                    ckpt_dir,
                    f"DynUNet_{num_subjects}subjects_seed{args.seed}_loss.png",
                )
            )
            plt.close()

    order_path = os.path.join(meta_dir, f"finetune_seed{args.seed}_acquisition_order.json")
    with open(order_path, "w") as f:
        json.dump([os.path.basename(p) for p in sample_pool], f, indent=2)
    logging.info("Final pool saved to %s", order_path)
    logging.info("Finished. Checkpoints in %s", ckpt_dir)


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def add_common_args(parser):
    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--spatial_size", nargs=3, type=int, default=[192, 192, 96])
    parser.add_argument("--seed", type=int, default=2025)


def main():
    parser = argparse.ArgumentParser(
        description="DynUNet active learning and finetune training."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    active_parser = subparsers.add_parser(
        "active",
        help="Active learning on trainingset_45 (random or entropy)",
    )
    add_common_args(active_parser)
    active_parser.add_argument(
        "--strategy",
        type=str,
        default="entropy",
        choices=("entropy", "random"),
        help="after initial subjects: entropy (MONAI Label epistemic) or random",
    )
    active_parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=5,
        help="Monte Carlo samples for epistemic entropy (MONAI Label default)",
    )
    active_parser.add_argument("--initial_subjects", type=int, default=INITIAL_SUBJECTS)
    active_parser.add_argument("--initial_epochs", type=int, default=INITIAL_EPOCHS)
    active_parser.add_argument("--incremental_epochs", type=int, default=INCREMENTAL_EPOCHS)
    active_parser.add_argument("--image_dir", type=str, default=ACTIVE_DEFAULTS["image_dir"])
    active_parser.add_argument("--label_dir", type=str, default=ACTIVE_DEFAULTS["label_dir"])
    active_parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="override checkpoint root (default: random or entropy specific dir)",
    )

    finetune49_parser = subparsers.add_parser(
        "finetune49",
        help="Finetune from 45-subject checkpoint onto trainingset_49",
    )
    add_common_args(finetune49_parser)
    finetune49_parser.add_argument(
        "--checkpoint",
        type=str,
        default=FINETUNE49_DEFAULTS["checkpoint"],
        help="45-subject starting checkpoint",
    )
    finetune49_parser.add_argument(
        "--image_dir",
        type=str,
        default=FINETUNE49_DEFAULTS["image_dir"],
    )
    finetune49_parser.add_argument(
        "--label_dir",
        type=str,
        default=FINETUNE49_DEFAULTS["label_dir"],
    )
    finetune49_parser.add_argument(
        "--incremental_epochs",
        type=int,
        default=INCREMENTAL_EPOCHS,
        help="epochs per newly added subject (default: 30)",
    )
    finetune49_parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=FINETUNE49_DEFAULTS["ckpt_dir"],
        help="output checkpoint root",
    )

    finetune51_parser = subparsers.add_parser(
        "finetune51",
        help="Finetune from 49-subject checkpoint onto trainingset_51",
    )
    add_common_args(finetune51_parser)
    finetune51_parser.add_argument(
        "--checkpoint",
        type=str,
        default=FINETUNE51_DEFAULTS["checkpoint"],
        help="49-subject starting checkpoint",
    )
    finetune51_parser.add_argument(
        "--image_dir",
        type=str,
        default=FINETUNE51_DEFAULTS["image_dir"],
    )
    finetune51_parser.add_argument(
        "--label_dir",
        type=str,
        default=FINETUNE51_DEFAULTS["label_dir"],
    )
    finetune51_parser.add_argument(
        "--incremental_epochs",
        type=int,
        default=INCREMENTAL_EPOCHS,
        help="epochs per newly added subject (default: 30)",
    )
    finetune51_parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=FINETUNE51_DEFAULTS["ckpt_dir"],
        help="output checkpoint root",
    )

    args = parser.parse_args()
    args.spatial_size = tuple(args.spatial_size)

    if args.mode == "active":
        run_active(args)
    elif args.mode == "finetune49":
        run_finetune(args, FINETUNE49_DEFAULTS)
    elif args.mode == "finetune51":
        run_finetune(args, FINETUNE51_DEFAULTS)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
