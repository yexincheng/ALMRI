"""
Active learning sampling strategies.

Entropy sampling follows MONAI Label epistemic active learning:
- Scoring: monailabel.tasks.scoring.epistemic.EpistemicScoring.entropy_3d_volume
- Selection: monailabel.tasks.activelearning.epistemic.Epistemic (highest epistemic_entropy)
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

def random_sampling(training_pool:list, sample_pool:list, seed:int):
    """Random sampling strategy
    Args:
        training_pool: list of training data path pool, each element is a npz file path containing 
            image, ground truth, and image embedding
        sample_pool: list of sample pool, initially empty
        seed: random seed
    Return:
        sample_pool: updated sample pool
        training_pool: updated training pool"""
    np.random.seed(seed)
    next_sample = np.random.choice(training_pool)
    sample_pool.append(next_sample)
    training_pool.remove(next_sample)
    return sample_pool, training_pool


def sequential_sampling(training_pool: list, sample_pool: list):
    """
    Deterministic bootstrap: add the lexicographically first remaining subject.
    Used only to build the initial pool before entropy scoring (not random AL).
    """
    training_pool.sort()
    next_sample = training_pool[0]
    sample_pool.append(next_sample)
    training_pool.pop(0)
    return sample_pool, training_pool


def entropy_3d_volume(vol_input: np.ndarray) -> np.ndarray:
    """
    Mean entropy across classes for Monte Carlo predictions.
    Adapted from MONAI Label EpistemicScoring.entropy_3d_volume.
    """
    vol_input = vol_input.astype("float32")
    dims = vol_input.shape
    reps = dims[0]
    entropy = np.zeros(dims[2:], dtype="float32")
    threshold = 0.00005
    vol_input = vol_input.copy()
    vol_input[vol_input <= 0] = threshold

    if len(dims) == 5:
        for channel in range(dims[1]):
            t_vol = np.squeeze(vol_input[:, channel, :, :, :])
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            entropy = entropy + (-np.multiply(t_avg, t_log))
    else:
        t_vol = np.squeeze(vol_input)
        t_sum = np.sum(t_vol, axis=0)
        t_avg = np.divide(t_sum, reps)
        t_log = np.log(t_avg)
        entropy = entropy + (-np.multiply(t_avg, t_log))
    return entropy


def compute_epistemic_entropy_score(
    network,
    image_tensor,
    device,
    num_samples: int = 10,
) -> float:
    """
    Epistemic entropy via MC dropout (model.train()), MONAI Label style.
    """
    import torch

    if num_samples < 2:
        num_samples = 2

    network.train()
    accum_outputs = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = network(image_tensor.to(device))
            soft_preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            accum_outputs.append(soft_preds)

    accum_numpy = np.stack(accum_outputs)
    accum_numpy = np.squeeze(accum_numpy)
    if accum_numpy.ndim > 4:
        accum_numpy = accum_numpy[:, 1:, :, :, :]

    entropy_map = entropy_3d_volume_with_monailabel(accum_numpy)
    return float(np.nanmean(entropy_map))


def select_by_epistemic_entropy(
    scores: Dict[str, float],
    strategy_key: str = "epistemic_entropy",
) -> str:
    """
    Pick the unlabeled image with highest entropy score.
    Mirrors monailabel.tasks.activelearning.epistemic.Epistemic selection.
    """
    if not scores:
        raise ValueError("No entropy scores available for selection.")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    image_path = ranked[0][0]
    logger.info(
        "%s: selected %s (score=%.6f)",
        strategy_key,
        os.path.basename(image_path),
        ranked[0][1],
    )
    return image_path


def entropy_sampling(
    training_pool: List[str],
    sample_pool: List[str],
    network,
    device,
    prepare_batch_fn: Callable[[str], dict],
    num_mc_samples: int = 10,
) -> Tuple[List[str], List[str], Dict[str, float]]:
    """
    Score remaining subjects with epistemic entropy and add the highest-scoring one.
    """
    scores: Dict[str, float] = {}
    for image_path in training_pool:
        batch = prepare_batch_fn(image_path)
        image_tensor = batch["image"]
        if hasattr(image_tensor, "unsqueeze") and image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(0)
        scores[image_path] = compute_epistemic_entropy_score(
            network, image_tensor, device, num_samples=num_mc_samples
        )
        logger.info(
            "Epistemic entropy %s: %.6f",
            os.path.basename(image_path),
            scores[image_path],
        )

    next_path = select_by_epistemic_entropy(scores)
    sample_pool.append(next_path)
    training_pool.remove(next_path)
    return sample_pool, training_pool, scores


def entropy_3d_volume_with_monailabel(vol_input: np.ndarray) -> np.ndarray:
    """Use MONAI Label implementation when the package is installed."""
    try:
        from monailabel.tasks.scoring.epistemic import EpistemicScoring

        return EpistemicScoring.entropy_3d_volume(vol_input)
    except ImportError:
        return entropy_3d_volume(vol_input)
