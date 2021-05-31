"""Debanding functions"""
from typing import Any, Dict, List, Union

import vapoursynth as vs

from .util import FormatError

core = vs.core


def dumb3kdb(clip: vs.VideoNode, radius: int = 16,
             threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
             sample_mode: int = 2, use_neo: bool = False, **kwargs) -> vs.VideoNode:
    """
        "f3kdb but better".
        Both f3kdb and neo_f3kdb actually change strength at 1 + 16 * n for sample_mode=2
        and 1 + 32 * n for sample_mode=1, 3 or 4. This function is aiming to average n and n + 1 strength
        for a better accuracy.
        Original function written by Z4ST1N, modified by Vardë.
        https://f3kdb.readthedocs.io/en/latest/index.html
        https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb

    Args:
        clip (vs.VideoNode): Source clip.

        radius (int, optional):
            Banding detection range. Defaults to 16.

        threshold (Union[int, List[int]], optional):
            Banding detection threshold(s) for planes.
            If difference between current pixel and reference pixel is less than threshold,
            it will be considered as banded. Defaults to 30.

        grain (Union[int, List[int]], optional):
            Specifies amount of grains added in the last debanding stage. Defaults to 0.

        sample_mode (int, optional):
            Valid modes are:
                – 1: Take 2 pixels as reference pixel. Reference pixels are in the same column of current pixel.
                – 2: Take 4 pixels as reference pixel. Reference pixels are in the square around current pixel.
                – 3: Take 2 pixels as reference pixel. Reference pixels are in the same row of current pixel.
                – 4: Arithmetic mean of 1 and 3.
            Reference points are randomly picked within the range. Defaults to 2.

        use_neo (bool, optional): Use neo_f3kdb.Deband. Defaults to False.

    Returns:
        vs.VideoNode: Debanded clip.
    """

    # neo_f3kdb nukes frame props
    def _trf(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:  # noqa: PLC0103, PLW0613
        (fout := f[0].copy()).props.update(f[1].props)
        return fout

    def _pick_f3kdb(neo: bool, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return core.neo_f3kdb.Deband(*args, **kwargs) if neo else core.f3kdb.Deband(*args, **kwargs)

    if sample_mode > 2 and not use_neo:
        raise ValueError('dumb3kdb: "sample_mode" argument should be less or equal to 2 when "use_neo" is false.')

    if clip.format is None:
        raise FormatError('dumb3kdb: Variable format not allowed!')


    thy, thcb, thcr = [threshold] * 3 if isinstance(threshold, int) else threshold + [threshold[-1]] * (3 - len(threshold))
    gry, grc = [grain] * 2 if isinstance(grain, int) else grain + [grain[-1]] * (2 - len(grain))

    thy, thcb, thcr = [max(1, x) for x in [thy, thcb, thcr]]


    step = 16 if sample_mode == 2 else 32

    f3kdb_args: Dict[str, Any] = dict(keep_tv_range=True, output_depth=16)
    f3kdb_args.update(kwargs)

    if thy % step == 1 and thcb % step == 1 and thcr % step == 1:
        deband = _pick_f3kdb(use_neo, clip, radius, thy, thcb, thcr, gry, grc, sample_mode, **f3kdb_args)
    else:
        loy, locb, locr = [(th - 1) // step * step + 1 for th in [thy, thcb, thcr]]
        hiy, hicb, hicr = [lo + step for lo in [loy, locb, locr]]

        lo_clip = _pick_f3kdb(use_neo, clip, radius, loy, locb, locr, gry, grc, sample_mode, **f3kdb_args)
        hi_clip = _pick_f3kdb(use_neo, clip, radius, hiy, hicb, hicr, gry, grc, sample_mode, **f3kdb_args)

        if clip.format.color_family == vs.GRAY:
            weight = [(thy - loy) / step]
        else:
            weight = [(thy - loy) / step, (thcb - locb) / step, (thcr - locr) / step]

        deband = core.std.Merge(lo_clip, hi_clip, weight)

    if use_neo:
        deband = core.std.ModifyFrame(deband, [deband, clip], selector=_trf)

    return deband
