"""Noising/denoising functions"""
from functools import partial
from typing import List, Tuple, Union, cast

import lvsfunc
from vsutil import (Dither, Range, depth, disallow_variable_format,
                    disallow_variable_resolution, get_y, split)

import vapoursynth as vs

from .mask import FDOG

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def decsiz(clip: vs.VideoNode, sigmaS: float = 10.0, sigmaR: float = 0.009,
           min_in: Union[int, float] = None, max_in: Union[int, float] = None, gamma: float = 1.0,
           protect_mask: vs.VideoNode = None, prefilter: bool = True,
           planes: List[int] = None, show_mask: bool = False) -> vs.VideoNode:
    """Denoising function using Bilateral intended to decrease the filesize
       by just blurring the invisible grain above max_in and keeping all of it
       below min_in. The range in between is progressive.

    Args:
        clip (vs.VideoNode): Source clip.

        sigmaS (float, optional): Bilateral parameter.
            Sigma of Gaussian function to calculate spatial weight. Defaults to 10.0.

        sigmaR (float, optional): Bilateral parameter.
            Sigma of Gaussian function to calculate range weight. Defaults to 0.009.

        min_in (Union[int, float], optional):
            Minimum pixel value below which the grain is kept. Defaults to None.

        max_in (Union[int, float], optional):
            Maximum pixel value above which the grain is blurred. Defaults to None.

        gamma (float, optional):
            Controls the degree of non-linearity of the conversion. Defaults to 1.0.

        protect_mask (vs.VideoNode, optional):
            Mask that includes all the details that should not be blurred.
            If None, it uses the default one.

        prefilter (bool, optional):
            Blurs the luma as reference or not. Defaults to True.

        planes (List[int], optional): Defaults to all planes.

        show_mask (bool, optional): Returns the mask.

    Returns:
        vs.VideoNode: Denoised clip.

    Example:
        import vardefunc as vdf

        clip = depth(clip, 16)
        clip = vdf.decsiz(clip, min_in=128<<8, max_in=200<<8)
    """

    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1
    gamma = 1 / gamma
    if clip.format.color_family == vs.GRAY:
        planes = [0]
    else:
        planes = [0, 1, 2] if not planes else planes


    if not protect_mask:
        clip16 = depth(clip, 16)
        masks = split(
            # partial(lvsfunc.mask.range_mask, rad=3, radc=2)(clip16).resize.Bilinear(format=vs.YUV444P16)
            core.std.BlankClip(clip, width=1, height=1)
        ) + [
            FDOG().get_mask(get_y(clip16)).std.Maximum().std.Minimum()
        ]
        protect_mask = core.std.Expr(masks, 'x y max z max 3250 < 0 65535 ? a max 8192 < 0 65535 ?') \
            .std.BoxBlur(hradius=1, vradius=1, hpasses=2, vpasses=2)


    clip_y = get_y(clip)
    if prefilter:
        pre = clip_y.std.BoxBlur(hradius=2, vradius=2, hpasses=4, vpasses=4)
    else:
        pre = clip_y

    denoise_mask = core.std.Expr(pre, f'x {min_in} max {max_in} min {min_in} - {max_in} {min_in} - / {gamma} pow 0 max 1 min {peak} *')
    mask = core.std.Expr(
        [depth(protect_mask, bits, range=Range.FULL, range_in=Range.FULL, dither_type=Dither.NONE), denoise_mask],
        'y x -'
    )

    if show_mask:
        return mask


    denoise = core.bilateral.Bilateral(clip, sigmaS=sigmaS, sigmaR=sigmaR, planes=planes, algorithm=0)

    return core.std.MaskedMerge(clip, denoise, mask, planes)


def adaptative_regrain(denoised: vs.VideoNode, new_grained: vs.VideoNode, original_grained: vs.VideoNode,
                       range_avg: Tuple[float, float] = (0.5, 0.4), luma_scaling: int = 28) -> vs.VideoNode:
    """Merge back the original grain below the lower range_avg value,
       apply the new grain clip above the higher range_avg value
       and weight both of them between the range_avg values for a smooth merge.
       Intended for use in applying a static grain in higher PlaneStatsAverage values
       to decrease the file size since we can't see a dynamic grain on that level.
       However, in dark scenes, it's more noticeable so we apply the original grain.

    Args:
        denoised (vs.VideoNode): The denoised clip.
        new_grained (vs.VideoNode): The new regrained clip.
        original_grained (vs.VideoNode): The original regrained clip.
        range_avg (Tuple[float, float], optional): Range used in PlaneStatsAverage. Defaults to (0.5, 0.4).
        luma_scaling (int, optional): Parameter in adg.Mask. Defaults to 28.

    Returns:
        vs.VideoNode: The new adaptative grained clip.

    Example:
        import vardefunc as vdf

        denoise = denoise_filter(src, ...)
        diff = core.std.MakeDiff(src, denoise)
        ...
        some filters
        ...
        new_grained = core.neo_f3kdb.Deband(last, preset='depth', grainy=32, grainc=32)
        original_grained = core.std.MergeDiff(last, diff)
        adapt_regrain = vdf.adaptative_regrain(last, new_grained, original_grained, range_avg=(0.5, 0.4), luma_scaling=28)
    """

    avg = core.std.PlaneStats(denoised)
    adapt_mask = core.adg.Mask(get_y(avg), luma_scaling)
    adapt_grained = core.std.MaskedMerge(new_grained, original_grained, adapt_mask)

    avg_max = max(range_avg)
    avg_min = min(range_avg)

    def _diff(n: int, f: vs.VideoFrame, avg_max: float, avg_min: float,
              new: vs.VideoNode, adapt: vs.VideoNode) -> vs.VideoNode:
        psa = cast(float, f.props['PlaneStatsAverage'])
        if psa > avg_max:
            clip = new
        elif psa < avg_min:
            clip = adapt
        else:
            weight = (psa - avg_min) / (avg_max - avg_min)
            clip = core.std.Merge(adapt, new, [weight])
        return clip

    diff_function = partial(_diff, avg_max=avg_max, avg_min=avg_min, new=new_grained, adapt=adapt_grained)

    return core.std.FrameEval(denoised, diff_function, [avg])
