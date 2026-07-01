from dataclasses import dataclass
import secrets

import numpy as np
from numpy.random import Generator, SeedSequence


@dataclass
class RNGStreams:
    "for different types of RNG"

    agn: Generator
    nsc: Generator
    bbh: Generator
    spin: Generator
    pair_up: Generator
    dynamics: Generator
    observation: Generator
    resampling: Generator


def resolve_seed(seed: int | None = None) -> int:
    if seed is None:
        return secrets.randbits(128)

    return seed


def make_rng_streams(master_seed: int) -> RNGStreams:

    parent_sequence = SeedSequence(master_seed)

    (
        agn_seq,
        nsc_seq,
        bbh_seq,
        spin_seq,
        pair_up_seq,
        dynamics_seq,
        observation_seq,
        resampling_seq,
    ) = parent_sequence.spawn(8)

    return RNGStreams(
        agn=np.random.default_rng(agn_seq),
        nsc=np.random.default_rng(nsc_seq),
        bbh=np.random.default_rng(bbh_seq),
        spin=np.random.default_rng(spin_seq),
        pair_up=np.random.default_rng(pair_up_seq),
        dynamics=np.random.default_rng(dynamics_seq),
        observation=np.random.default_rng(observation_seq),
        resampling=np.random.default_rng(resampling_seq),
    )
