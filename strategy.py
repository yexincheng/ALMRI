import logging
import random
import time
import numpy as np

def random_sampling(training_pool, sample_pool):
    np.random.seed(2023)
    next_sample = np.random.choice(training_pool)
    sample_pool.append(next_sample)
    training_pool.remove(next_sample)
    return sample_pool, training_pool

