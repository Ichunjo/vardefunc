"""Some cool functions"""

# flake8: noqa
from . import aa, deband, mask, misc, noise, placebo, scale, sharp, types, util
from .aa import *
from .mask import *
from .misc import *
from .noise import *
from .ocr import *
from .scale import *
from .types import *
from .util import *

drm = mask.Difference().rescale
dcm = mask.Difference().creditless
lcm = mask.luma_credit_mask
