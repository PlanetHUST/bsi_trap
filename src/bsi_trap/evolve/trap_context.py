"""Construction of local trap geometry used by pairing dynamics."""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import DiskProfile, TrapContext
from bsi_trap.config.model_config import ModelConfig


def build_trap_context(disk: DiskProfile, model_config: ModelConfig) -> TrapContext:
    """Build the finite annular trap volume from explicit model parameters.

    The trap is represented as an annulus with
    ``V = 4 pi R_trap Delta_R_trap H_BH``. Both the radial width and compact
    object scale height are controlled by ``ModelConfig`` in units of the
    local gas scale height, keeping the density assumption visible at the
    workflow boundary.
    """
    gas_scale_height_cm = disk.get_scale_height_cm(disk.r_trap_cm)
    width_cm = model_config.trap_width_scale_heights * gas_scale_height_cm
    compact_object_height_cm = (
        model_config.compact_object_scale_height_fraction * gas_scale_height_cm
    )
    volume_cm3 = (
        4.0
        * np.pi
        * disk.r_trap_cm
        * width_cm
        * compact_object_height_cm
    )
    return TrapContext(
        width_cm=width_cm,
        scale_height_cm=compact_object_height_cm,
        volume_cm3=volume_cm3,
    )
