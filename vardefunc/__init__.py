"""Some cool functions"""

# flake8: noqa

from . import deband, mask, misc, noise, placebo, scale, sharp, types, util

drm = mask.Difference().rescale
dcm = mask.Difference().creditless
lcm = mask.luma_credit_mask
