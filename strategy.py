import logging
import random
import time
import numpy as np
from utils import NpzDataset
from torch.utils.data import DataLoader
from eval import infer

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

