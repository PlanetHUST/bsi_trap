import numpy as np


def make_sample_seed(base_seed: int, model_index: int, sample_index: int) -> int:
    return int(base_seed + 100_000 * model_index + sample_index)


def make_rng(
    base_seed: int, model_index: int, sample_index: int
) -> np.random.Generator:
    seed = make_sample_seed(base_seed, model_index, sample_index)
    return np.random.default_rng(seed)
