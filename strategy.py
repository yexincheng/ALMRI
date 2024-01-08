import logging
import random
import time
import numpy as np
from utils import NpzDataset
from torch.utils.data import DataLoader
from eval import infer

def random_sampling(training_pool, sample_pool, seed):
    np.random.seed(seed)
    next_sample = np.random.choice(training_pool)
    sample_pool.append(next_sample)
    training_pool.remove(next_sample)
    return sample_pool, training_pool

