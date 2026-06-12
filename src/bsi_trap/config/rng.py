from dataclasses import dataclass
import secrets

import numpy as np
from numpy.random import Generator, SeedSequence


@dataclass
class RNGStreams:
    """Named random-number streams for one simulation workflow."""

    agn: Generator
    nsc: Generator
    bbh: Generator
    spin: Generator
    pair_up: Generator
    dynamics: Generator
    observation: Generator
    resampling: Generator


def resolve_seed(seed: int | None) -> int:
    """
    Resolve a possibly-None seed into an actual integer seed.

    If seed is None, a fresh 128-bit seed is generated.
    The returned seed should be logged for reproducibility.
    """
    if seed is None:
        return secrets.randbits(128)

    return seed


def make_rng_streams(master_seed: int) -> RNGStreams:
    """
    Create independent RNG streams from one master seed.

    The stream order must be stable. Do not reorder these entries casually,
    because it changes the stochastic workflow.
    """
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
