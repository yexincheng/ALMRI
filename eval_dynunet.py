"""
Unified DynUNet evaluation and inference.

Modes (subcommands):
  curve   Sweep active-learning checkpoints (*latest.pth) and plot Dice vs pool size.
          Uses an npz subject pool under {prefix}/{task}/{mode}/ and evaluates one
          organ via --label_id.

  infer   Run a single checkpoint on a directory of NIfTI images. Computes per-organ
          and mean Dice when --label_dir is set; saves label maps and one-hot npz
          predictions.
"""
import argparse
import distutils.util
import json
import os
import warnings
from glob import glob
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
)
from monai.config import KeysCollection
from monai.data import DataLoader, Dataset, MetaTensor
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    MapTransform,
    Orientationd,
    Resize,
)
from monai.utils import InterpolateMode, ensure_tuple_rep
from tqdm import tqdm

from metrics import compute_dice_coefficient

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ORGAN_LABEL_IDS = (1, 2, 3)
ORGAN_NAMES = {1: "left_kidney", 2: "right_kidney", 3: "pancreas"}
FULL_LABELS = {
    "left kidney": 1,
    "right kidney": 2,
    "pancreas": 3,
    "background": 0,
}
NUMBER_INTENSITY_CH = 1

INFER_DEFAULTS = {
    "checkpoint": os.path.join(
        SCRIPT_DIR,
        "checkpoints",
        "DynUNet_trainingset45_entropy_active",
        "seed2025",
        "init100_inc30",
        "DynUNet_45subjects_seed2025.pt",
    ),
    "image_dir": os.path.join(SCRIPT_DIR, "datasets", "trainingset_51"),
    "pred_dir": os.path.join(SCRIPT_DIR, "inference_outputs", "pred_trainingset_51"),
    "dice_dir": os.path.join(SCRIPT_DIR, "results", "dice", "trainingset_51"),
}


class Restored(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        has_channel: bool = True,
        mode: str = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(keys)
        self.ref_image = ref_image
        self.has_channel = has_channel
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        meta_dict = (
            d[self.ref_image].meta
            if d.get(self.ref_image) is not None and isinstance(d[self.ref_image], MetaTensor)
            else d.get(f"{self.ref_image}_{self.meta_key_postfix}", {})
        )

        for idx, key in enumerate(self.keys):
            result = d[key]
            current_size = result.shape[1:] if self.has_channel else result.shape
            spatial_shape = meta_dict.get("spatial_shape", current_size)
            spatial_size = spatial_shape[-len(current_size) :]
            if torch.any(np.not_equal(current_size, spatial_size)):
                resizer = Resize(spatial_size=spatial_size, mode=self.mode[idx])
                result = resizer(
                    result, mode=self.mode[idx], align_corners=self.align_corners[idx]
                )
            d[key] = (
                result
                if len(result.shape) <= 3
                else result[0]
                if result.shape[0] == 1
                else result
            )

            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["affine"] = meta_dict.get("original_affine")
        return d


def get_post_transforms(pre_transforms):
    return Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            Invertd(keys="pred", transform=pre_transforms, orig_keys="image"),
            Restored(keys="pred", ref_image="image", mode="nearest"),
        ]
    )


def get_pre_transforms_curve(labels, spatial_size):
    """Legacy curve mode: guidance points only (input channels doubled at inference)."""
    return Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            AddGuidanceFromPointsDeepEditd(
                ref_image="image", guidance="guidance", label_names=labels
            ),
            CenterSpatialCropd(keys="image", roi_size=spatial_size),
            EnsureTyped(keys="image"),
        ]
    )


def get_pre_transforms_infer(labels, spatial_size, number_intensity_ch=NUMBER_INTENSITY_CH):
    """Infer mode: full 5-channel guidance signal matching active training."""
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
                number_intensity_ch=number_intensity_ch,
            ),
            EnsureTyped(keys="image"),
        ]
    )


def get_network(in_channels: int, out_channels: int):
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )


def read_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "network" in ckpt:
            return ckpt["network"]
        return ckpt
    return ckpt


def get_in_channels_from_state_dict(state_dict) -> int:
    for key, tensor in state_dict.items():
        if key.endswith("input_block.conv1.conv.weight"):
            return int(tensor.shape[1])
    raise KeyError("Could not infer in_channels from checkpoint state_dict")


def data_load_curve(args, pre_transforms):
    testing_pool_path = os.path.join(args.prefix, args.task, args.mode)
    testing_pool = glob(os.path.join(testing_pool_path, "*.npz"))
    testing_pool_nii = [
        os.path.basename(sample).split(".npz")[0] + ".nii.gz" for sample in testing_pool
    ]
    imagelist = sorted(
        [
            os.path.join(args.image_dir, image_name)
            for image_name in testing_pool_nii
        ]
    )
    images_ds = Dataset(data=[{"image": i} for i in imagelist], transform=pre_transforms)
    images_loader = DataLoader(images_ds, batch_size=1, shuffle=False)
    return images_loader, imagelist


def data_load_infer(image_dir, image_list, pre_transforms):
    if image_list:
        with open(image_list) as f:
            names = [line.strip() for line in f if line.strip()]
        imagelist = sorted(
            [
                os.path.join(
                    image_dir,
                    n if n.endswith(".nii.gz") else f"{n}.FatImaging_W.nii.gz",
                )
                for n in names
            ]
        )
    else:
        imagelist = sorted(glob(os.path.join(image_dir, "*.nii.gz")))

    images_ds = Dataset(data=[{"image": i} for i in imagelist], transform=pre_transforms)
    images_loader = DataLoader(images_ds, batch_size=1, shuffle=False)
    return images_loader, imagelist


def load_gt_volume(gt_path):
    gt = sitk.ReadImage(gt_path)
    gt_array = sitk.GetArrayFromImage(gt)
    return np.transpose(gt_array, (2, 1, 0))


def label_to_onehot(volume: np.ndarray, label_ids=ORGAN_LABEL_IDS) -> np.ndarray:
    return np.stack([np.uint8(volume == label_id) for label_id in label_ids], axis=0)


def compute_organ_dice_onehot(
    gt_volume: np.ndarray, pred_volume: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    gt_onehot = label_to_onehot(gt_volume)
    pred_onehot = label_to_onehot(pred_volume)

    dice = {}
    for channel_idx, label_id in enumerate(ORGAN_LABEL_IDS):
        organ = ORGAN_NAMES[label_id]
        dice[organ] = float(
            compute_dice_coefficient(
                gt_onehot[channel_idx].astype(bool),
                pred_onehot[channel_idx].astype(bool),
            )
        )
    dice["mean"] = float(np.nanmean([dice[ORGAN_NAMES[i]] for i in ORGAN_LABEL_IDS]))
    return dice, gt_onehot, pred_onehot


def setup_output_dirs(pred_dir: str, dice_dir: Optional[str], save_predictions: bool) -> None:
    dirs_to_create = []
    if dice_dir is not None:
        dirs_to_create.append(dice_dir)
    if save_predictions:
        dirs_to_create.append(pred_dir)

    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def save_case_outputs(
    case_name: str,
    pred_volume: np.ndarray,
    pred_onehot: np.ndarray,
    affine: np.ndarray,
    pred_dir: str,
):
    os.makedirs(pred_dir, exist_ok=True)
    stem = case_name.replace(".nii.gz", "")
    np.savez_compressed(
        os.path.join(pred_dir, f"{stem}_onehot.npz"),
        **{ORGAN_NAMES[label_id]: pred_onehot[i] for i, label_id in enumerate(ORGAN_LABEL_IDS)},
    )
    nib.save(
        nib.Nifti1Image(pred_volume.astype(np.uint8), affine),
        os.path.join(pred_dir, case_name),
    )


def run_curve(args):
    save_path_ckp = os.path.join(
        args.root,
        "checkpoints",
        f"{args.base_model}_{args.task}_{args.strategy}",
        f"seed{args.seed}",
        f"epochs{args.num_epochs}",
    )
    ckp_pool = sorted(glob(os.path.join(save_path_ckp, "*latest.pth")))
    print(save_path_ckp)
    print("Number of checkpoints:", len(ckp_pool))

    save_nii_path = os.path.join(
        "datasets",
        "predictions",
        f"{args.task}_{args.base_model}_{args.strategy}",
        f"seed{args.seed}",
        f"epochs{args.num_epochs}",
        args.mode,
    )
    os.makedirs(save_nii_path, exist_ok=True)

    save_path_dice = os.path.join("results", "dice", f"epochs{args.num_epochs}")
    os.makedirs(save_path_dice, exist_ok=True)

    device = torch.device("cuda" if args.use_gpu else "cpu")
    pre_transforms = get_pre_transforms_curve(args.labels, args.spatial_size)
    post_transforms = get_post_transforms(pre_transforms)
    dataloader, imagelist = data_load_curve(args, pre_transforms)

    dice_test = []
    network = get_network(len(args.labels), len(args.labels))
    for j, checkpoint_path in enumerate(tqdm(ckp_pool)):
        print("running checkpoint", checkpoint_path)
        avg_dice = []
        network.load_state_dict(torch.load(checkpoint_path, map_location=device))
        network.to(device)
        network.eval()
        with torch.no_grad():
            for ii, batch in enumerate(dataloader):
                original_affine = batch["image_meta_dict"]["affine"][0, :, :].numpy()
                name = os.path.split(imagelist[ii])[1]
                print(name)

                gt_path = os.path.join(args.label_dir, name)
                gt = sitk.ReadImage(gt_path)
                gt_array = sitk.GetArrayFromImage(gt)
                gt_array = np.transpose(np.uint8(gt_array == args.label_id), (2, 1, 0))

                model_input = batch["image"].repeat(1, 2, 1, 1, 1).to(device)
                batch["pred"] = network(model_input)[0]
                batch["image"] = batch["image"][0]
                batch = post_transforms(batch)

                dice = compute_dice_coefficient(
                    gt_array.astype(bool), batch["pred"].astype(np.uint8).astype(bool)
                )
                avg_dice.append(dice)
                if "40" in checkpoint_path:
                    nib.save(
                        nib.Nifti1Image(
                            batch["pred"].astype(np.uint8), original_affine
                        ),
                        os.path.join(save_nii_path, name),
                    )

            np.save(
                os.path.join(
                    save_path_dice,
                    f"{args.strategy}_{args.task}_{args.base_model}_epochs{args.num_epochs}_"
                    f"seed{args.seed}_{args.mode}_{j + 1}samples_dice.npy",
                ),
                avg_dice,
            )
            dice_test.append(np.mean(avg_dice))

    np.save(
        os.path.join(
            save_path_dice,
            f"{args.strategy}_{args.task}_{args.base_model}_epochs{args.num_epochs}_"
            f"seed{args.seed}_{args.mode}_dice.npy",
        ),
        dice_test,
    )

    figure_dir = os.path.join("figures", args.mode)
    os.makedirs(figure_dir, exist_ok=True)
    plt.plot(dice_test)
    plt.xlabel("Number of samples")
    plt.ylabel("Dice coefficient")
    plt.title(
        f"{args.base_model} {args.task} {args.strategy} "
        f"epochs{args.num_epochs} seed{args.seed} sampling"
    )
    plt.savefig(
        os.path.join(
            figure_dir,
            f"{args.base_model}_{args.task}_{args.strategy}_epochs{args.num_epochs}_"
            f"seed{args.seed}_sampling.png",
        )
    )
    plt.close()


def run_infer(args):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    has_labels = args.label_dir is not None
    if has_labels:
        if not os.path.isdir(args.label_dir):
            raise FileNotFoundError(f"Label directory not found: {args.label_dir}")
        if args.dice_dir is None:
            args.dice_dir = INFER_DEFAULTS["dice_dir"]

    print("Checkpoint:", args.checkpoint)
    print("Image dir:", args.image_dir)
    print("Label dir:", args.label_dir)
    print("Prediction dir:", args.pred_dir)
    print("Dice dir:", args.dice_dir)

    setup_output_dirs(args.pred_dir, args.dice_dir if has_labels else None, args.save_predictions)

    device = torch.device("cuda" if args.use_gpu else "cpu")
    state_dict = read_checkpoint(args.checkpoint, device)
    in_channels = get_in_channels_from_state_dict(state_dict)
    out_channels = len(FULL_LABELS)
    expected_in = NUMBER_INTENSITY_CH + len(FULL_LABELS)
    print(
        f"Checkpoint in_channels={in_channels}, out_channels={out_channels} "
        f"(expected in={expected_in})"
    )
    if in_channels != expected_in:
        print(
            f"Warning: checkpoint expects {in_channels} input channels, "
            f"but guidance pipeline produces {expected_in}."
        )

    pre_transforms = get_pre_transforms_infer(FULL_LABELS, args.spatial_size)
    post_transforms = get_post_transforms(pre_transforms)
    dataloader, imagelist = data_load_infer(args.image_dir, args.image_list, pre_transforms)

    network = get_network(in_channels, out_channels)
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()

    per_case_dice = []
    with torch.no_grad():
        for ii, batch in enumerate(tqdm(dataloader, desc="cases")):
            original_affine = batch["image_meta_dict"]["affine"][0, :, :].numpy()
            name = os.path.split(imagelist[ii])[1]
            print(name)

            batch_on_device = dict(batch)
            model_input = batch["image"].to(device)
            if model_input.shape[1] != in_channels:
                raise RuntimeError(
                    f"Input has {model_input.shape[1]} channels but checkpoint expects {in_channels}."
                )
            batch_on_device["image"] = model_input
            batch_on_device["pred"] = network(model_input)[0]
            batch_on_device["image"] = batch["image"][0]
            batch_on_device = post_transforms(batch_on_device)

            pred_volume = batch_on_device["pred"].astype(np.uint8)
            pred_onehot = label_to_onehot(pred_volume)

            if has_labels:
                gt_path = os.path.join(args.label_dir, name)
                gt_volume = load_gt_volume(gt_path)
                case_dice, _, _ = compute_organ_dice_onehot(gt_volume, pred_volume)
                case_dice["case"] = name
                per_case_dice.append(case_dice)

            if args.save_predictions:
                save_case_outputs(
                    name, pred_volume, pred_onehot, original_affine, args.pred_dir
                )

    if has_labels:
        mean_per_organ = {
            organ: float(np.nanmean([d[organ] for d in per_case_dice]))
            for organ in ORGAN_NAMES.values()
        }
        mean_all_organs = float(np.nanmean([d["mean"] for d in per_case_dice]))
        summary = {
            "checkpoint": args.checkpoint,
            "per_case": per_case_dice,
            "mean_per_organ": mean_per_organ,
            "mean_all_organs": mean_all_organs,
        }

        os.makedirs(args.dice_dir, exist_ok=True)
        dice_json_path = os.path.join(args.dice_dir, "dice.json")
        with open(dice_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        np.save(
            os.path.join(args.dice_dir, "dice_per_case.npy"),
            per_case_dice,
            allow_pickle=True,
        )
        np.save(os.path.join(args.dice_dir, "dice_mean_per_organ.npy"), mean_per_organ)
        np.save(
            os.path.join(args.dice_dir, "dice_mean_all_organs.npy"),
            np.array(mean_all_organs),
        )

        print("Dice saved to", args.dice_dir)
        print("Mean per organ:", mean_per_organ)
        print("Mean all organs:", mean_all_organs)
    elif args.save_predictions:
        print("Predictions saved to", args.pred_dir)


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def add_common_args(parser):
    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument(
        "--spatial_size",
        nargs=3,
        type=int,
        default=[192, 192, 96],
        help="center-crop size (H, W, D)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="DynUNet evaluation: checkpoint curve or single-checkpoint inference."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    curve_parser = subparsers.add_parser(
        "curve",
        help="Evaluate Dice curve across active-learning checkpoints",
    )
    add_common_args(curve_parser)
    curve_parser.add_argument(
        "--root",
        type=str,
        default="./",
        help="root path for checkpoints",
    )
    curve_parser.add_argument(
        "--prefix",
        type=str,
        default=os.path.join(SCRIPT_DIR, "datasets", "RAINE_organ_51"),
        help="dataset path prefix containing npz subject pool",
    )
    curve_parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "datasets", "trainingset_51"),
        help="directory with input NIfTI images",
    )
    curve_parser.add_argument(
        "--label_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "datasets", "trainingset_51", "labels", "final"),
        help="ground-truth label directory",
    )
    curve_parser.add_argument("--task", type=str, default="MRI_LeftKidney")
    curve_parser.add_argument("--strategy", type=str, default="random")
    curve_parser.add_argument("--base_model", type=str, default="DynUNet")
    curve_parser.add_argument("--label_id", type=int, default=1)
    curve_parser.add_argument("--seed", type=int, default=2024)
    curve_parser.add_argument("--num_epochs", type=int, default=100)
    curve_parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="npz pool subdirectory name under {prefix}/{task}/",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Single-checkpoint inference with per-organ Dice on trainingset_51",
    )
    add_common_args(infer_parser)
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        default=INFER_DEFAULTS["checkpoint"],
        help="path to model checkpoint (.pt or .pth)",
    )
    infer_parser.add_argument(
        "--image_dir",
        type=str,
        default=INFER_DEFAULTS["image_dir"],
        help="directory with input *.nii.gz images",
    )
    infer_parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="ground-truth label directory; if omitted, dice is skipped",
    )
    infer_parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="optional text file with subject ids (one per line)",
    )
    infer_parser.add_argument(
        "--pred_dir",
        type=str,
        default=INFER_DEFAULTS["pred_dir"],
        help="directory for saved predictions (label map + one-hot npz)",
    )
    infer_parser.add_argument(
        "--dice_dir",
        type=str,
        default=None,
        help="directory for per-organ Dice results (requires --label_dir)",
    )
    infer_parser.add_argument(
        "--save_predictions",
        type=strtobool,
        default="true",
        help="write predictions under pred_dir/",
    )

    args = parser.parse_args()
    args.spatial_size = tuple(args.spatial_size)

    if args.command == "curve":
        args.labels = {
            "left kidney": 1,
            "background": 0,
        }
        run_curve(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        parser.error(f"Unknown command: {args.command}")

    print("Finished")


if __name__ == "__main__":
    main()
