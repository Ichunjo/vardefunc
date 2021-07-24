"""Various functions used for debanding."""
import warnings
from typing import Any, Dict, List, Optional, Union

import vapoursynth as vs
from debandshit import SAMPLEMODE
from debandshit import dumb3kdb as d3kdb
from debandshit import f3kbilateral as f3kbil
from debandshit import lfdeband as lfdb

core = vs.core



def dumb3kdb(clip: vs.VideoNode, radius: int = 16,
             threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
             sample_mode: SAMPLEMODE = 2, use_neo: bool = False, **kwargs: Any) -> vs.VideoNode:
    warnings.warn('dumb3kdb: use debandshit.dumb3kdb instead', DeprecationWarning)
    return d3kdb(clip, radius, threshold, grain, sample_mode, use_neo, **kwargs)


def f3kbilateral(clip: vs.VideoNode, radius: int = 16,
                 threshold: Union[int, List[int]] = 65, grain: Union[int, List[int]] = 0,
                 f3kdb_args: Optional[Dict[str, Any]] = None,
                 limflt_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    warnings.warn('f3kbilateral: use debandshit.f3kbilateral instead', DeprecationWarning)
    return f3kbil(clip, radius, threshold, grain, f3kdb_args, limflt_args)


def lfdeband(clip: vs.VideoNode) -> vs.VideoNode:
    warnings.warn('lfdeband: use debandshit.lfdeband instead', DeprecationWarning)
    return lfdb(clip)
