"""Some cool functions"""

# flake8: noqa
from vapoursynth import __version__ as vs_version  # type: ignore[attr-defined]

from . import aa, deband, mask, misc, noise, placebo, scale, sharp, types, util

drm = mask.Difference().rescale
dcm = mask.Difference().creditless
lcm = mask.luma_credit_mask


def __check_vs_version() -> None:
    if vs_version.release_major < 56:
        raise ImportError('"vardautomation" only supports Vapoursynth R56 and above!')


__check_vs_version()
