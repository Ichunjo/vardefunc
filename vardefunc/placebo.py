"""Placebo wrapper"""
import warnings
from typing import List, Union

import vapoursynth as vs
from debandshit import placebo_deband

from .scale import placebo_shader

core = vs.core


def deband(clip: vs.VideoNode, radius: float = 16.0,
           threshold: Union[float, List[float]] = 4.0, iterations: int = 1,
           grain: Union[float, List[float]] = 6.0, chroma: bool = True, **kwargs) -> vs.VideoNode:
    warnings.warn('placebo.deband: use debandshit.placebo_deband instead', DeprecationWarning)
    return placebo_deband(clip, radius, threshold, iterations, grain, **kwargs)



def shader(clip: vs.VideoNode, width: int, height: int, shader_file: str, luma_only: bool = True, **kwargs) -> vs.VideoNode:
    warnings.warn('placebo.shader: use vardefunc.scale.placebo_shader instead', DeprecationWarning)
    return placebo_shader(clip, width, height, shader_file, luma_only, **kwargs)
