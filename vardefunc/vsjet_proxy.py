from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator, Literal, Sequence, cast, overload

import numpy as np
import vsaa
import vsmasktools
import vsscale
import vstools
from jetpytools import KwargsT
from vskernels import BorderHandling, Hermite, KernelLike, LeftShift, ScalerLike, TopShift
from vstools import (
    ConstantFormatVideoNode,
    FieldBasedT,
    FrameRangeN,
    FrameRangesN,
    VSFunctionNoArgs,
    set_output,
    vs,
)

from .types import AnyInt, NDArray
from .types import VNumpy as vnp
from .util import normalise_ranges, ranges_to_indices, select_frames

__all__ = [
    "is_preview",
    "set_output",
    "replace_ranges",
    "based_aa",
]


@lru_cache
def is_preview() -> bool:
    try:
        import vspreview.api
    except ImportError:
        is_preview = False
    else:
        is_preview = vspreview.api.is_preview()
    return is_preview


vstools.replace_ranges.exclusive = True


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN,
    /,
    *,
    exclusive: bool = True,
    mismatch: bool = False,
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: vstools.utils.ranges._RangesCallBack,
    /,
    *,
    mismatch: bool = False,
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: vstools.utils.ranges._RangesCallBackF[vs.VideoFrame]
    | vstools.utils.ranges._RangesCallBackNF[vs.VideoFrame],
    /,
    *,
    mismatch: bool = False,
    prop_src: vs.VideoNode,
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: vstools.utils.ranges._RangesCallBackF[Sequence[vs.VideoFrame]]
    | vstools.utils.ranges._RangesCallBackNF[Sequence[vs.VideoFrame]],
    /,
    *,
    mismatch: bool = False,
    prop_src: list[vs.VideoNode],
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    *clip_b: tuple[vs.VideoNode, FrameRangeN | FrameRangesN | vstools.utils.ranges._RangesCallBack],
    mismatch: bool = False,
) -> vs.VideoNode: ...


def replace_ranges(
    clip_a: vs.VideoNode,
    *args: Any,
    exclusive: bool = True,
    mismatch: bool = False,
    prop_src: vs.VideoNode | list[vs.VideoNode] | None = None,
) -> vs.VideoNode:
    """
    Replaces frames in a clip, either with pre-calculated indices or on-the-fly with a callback.
    Frame ranges are by default exclusive. This behaviour can be changed by setting `exclusive=False`.

    Examples with clips ``black`` and ``white`` of equal length:
        * ``replace_ranges(black, white, [(0, 1)])``: replace frames 0 with ``white``
        * ``replace_ranges(black, white, [(0, 2)])``: replace frames 0 and 1 with ``white``
        * ``replace_ranges(black, white, [(None, None)])``: replace the entire clip with ``white``
        * ``replace_ranges(black, white, [(0, None)])``: same as previous
        * ``replace_ranges(black, white, [(200, None)])``: replace 200 until the end with ``white``
        * ``replace_ranges(black, white, [(200, -1)])``: replace 200 until the end with ``white``,
                                                         leaving 1 frame of ``black``

    Optional Dependencies:
        * Either of the following two plugins:
            * `VS Julek Plugin <https://github.com/dnjulek/vapoursynth-julek-plugin>`_ (recommended!)
            * `VSRemapFrames <https://github.com/Irrational-Encoding-Wizardry/Vapoursynth-RemapFrames>`_

    :param clip_a:      Original clip.
    :param clip_b:      Replacement clip.
    :param ranges:      Ranges to replace clip_a (original clip) with clip_b (replacement clip).
                        Integer values in the list indicate single frames,
                        Tuple values indicate inclusive ranges.
                        Callbacks must return true to replace a with b.
                        Negative integer values will be wrapped around based on clip_b's length.
                        None values are context dependent:
                            * None provided as sole value to ranges: no-op
                            * Single None value in list: Last frame in clip_b
                            * None as first value of tuple: 0
                            * None as second value of tuple: Last frame in clip_b
    :param exclusive:   Use exclusive ranges (Default: True).
    :param mismatch:    Accept format or resolution mismatch between clips.

    :return:            Clip with ranges from clip_a replaced with clip_b.
    """
    if len(args) == 0:
        return clip_a

    if isinstance(clip_b := args[0], vs.VideoNode):
        ranges: FrameRangeN | FrameRangesN | vstools.utils.ranges._RangesCallBackLike | None = args[1]

        if exclusive and not callable(ranges):
            ranges = normalise_ranges(clip_b, ranges, norm_dups=True)

        return vstools.replace_ranges(clip_a, clip_b, ranges, exclusive, mismatch, prop_src=prop_src)

    if not exclusive:
        raise NotImplementedError

    rclips: tuple[tuple[vs.VideoNode, FrameRangeN | FrameRangesN | vstools.utils.ranges._RangesCallBack], ...] = args

    if len(rclips) <= 10:
        for c, r in rclips:
            clip_a = replace_ranges(clip_a, c, r, mismatch=mismatch)
        return clip_a

    ref_indices = np.zeros(clip_a.num_frames, np.uint32)

    rrclips = [
        (c, np.fromiter(ranges_to_indices.gen_indices(c, r, (0, i)), np.uint32, c.num_frames))
        for (i, (c, r)) in enumerate(rclips, 1)
    ]

    clips, indices_iter = cast(tuple[Iterator[vs.VideoNode], Iterator[NDArray[AnyInt]]], zip(*rrclips))

    indices = list[NDArray[AnyInt]]()

    for i in indices_iter:
        if (isize := i.size) < (rsize := ref_indices.size):
            i = np.pad(i, (0, rsize - isize))
        elif isize > rsize:
            i = i[:rsize]

        indices.append(i)

    nindices = np.max([ref_indices, *indices], axis=0, out=ref_indices)

    return select_frames(
        [clip_a, *clips], vnp.zip_arrays(nindices, np.arange(clip_a.num_frames, dtype=np.uint32)), mismatch=mismatch
    )


def based_aa(
    clip: vs.VideoNode,
    rfactor: float = 2.0,
    mask: vs.VideoNode | vsmasktools.EdgeDetectT | Literal[False] = vsmasktools.Prewitt,
    mask_thr: int = 60,
    pscale: float = 0.0,
    downscaler: ScalerLike | None = None,
    supersampler: ScalerLike | Literal[False] = vsscale.ArtCNN,
    antialiaser: vsaa.AntiAliaser | None = None,
    prefilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | Literal[False] = False,
    postfilter: VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | Literal[False] | KwargsT | None = None,
    show_mask: bool = False,
    **aa_kwargs: Any,
) -> vs.VideoNode:
    """vsaa.based_aa with use_mclip=True and opt=3"""

    return vsaa.based_aa(
        clip,
        rfactor,
        mask,
        mask_thr,
        pscale,
        downscaler,
        supersampler,
        antialiaser,
        prefilter,
        postfilter,
        show_mask,
        **{"use_mclip": True, "opt": 3} | aa_kwargs,
    )


class Rescale(vsscale.Rescale):
    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: int | float,
        kernel: KernelLike,
        upscaler: ScalerLike = vsscale.ArtCNN,
        downscaler: ScalerLike = Hermite(linear=True),
        width: int | float | None = None,
        base_height: int | None = None,
        base_width: int | None = None,
        crop: tuple[
            vsscale.helpers.LeftCrop, vsscale.helpers.RightCrop, vsscale.helpers.TopCrop, vsscale.helpers.BottomCrop
        ] = vsscale.helpers.CropRel(),
        shift: tuple[TopShift, LeftShift] = (0, 0),
        field_based: FieldBasedT | bool | None = None,
        border_handling: int | BorderHandling = BorderHandling.MIRROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            clip,
            height,
            kernel,
            upscaler,
            downscaler,
            width,
            base_height,
            base_width,
            crop,
            shift,
            field_based,
            border_handling,
            **{"_add_props": is_preview()} | kwargs,
        )
