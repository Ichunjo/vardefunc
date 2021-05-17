"""Some cool functions"""

# flake8: noqa

from . import deband, mask, misc, noise, placebo, scale, sharp, util

drm = mask.diff_rescale_mask
dcm = mask.diff_creditless_mask
lcm = mask.luma_credit_mask
gk = misc.generate_keyframes
