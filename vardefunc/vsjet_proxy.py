from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator, Sequence, cast, overload

import numpy as np
import vsscale
import vstools
from vskernels import BorderHandling, ComplexKernelLike, Hermite, LeftShift, ScalerLike, TopShift
from vstools import FieldBasedT, FrameRangeN, FrameRangesN, vs

from .types import AnyInt, VNumpy
from .util import normalise_ranges, ranges_to_indices, select_frames

__all__ = ["Rescale", "is_preview", "replace_ranges"]


@lru_cache
def is_preview() -> bool:
    import importlib

    try:
        module = importlib.import_module("vspreview.api")
        return module.is_preview()
    except ImportError:
        return False


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
    ranges: vstools.functions.ranges._RangesCallBack,
    /,
    *,
    mismatch: bool = False,
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: vstools.functions.ranges._RangesCallBackF[vs.VideoFrame]
    | vstools.functions.ranges._RangesCallBackNF[vs.VideoFrame],
    /,
    *,
    mismatch: bool = False,
    prop_src: vs.VideoNode,
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: vstools.functions.ranges._RangesCallBackF[Sequence[vs.VideoFrame]]
    | vstools.functions.ranges._RangesCallBackNF[Sequence[vs.VideoFrame]],
    /,
    *,
    mismatch: bool = False,
    prop_src: list[vs.VideoNode],
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    *clip_b: tuple[vs.VideoNode, FrameRangeN | FrameRangesN | vstools.functions.ranges._RangesCallBack],
    mismatch: bool = False,
) -> vs.VideoNode: ...


def replace_ranges(
    clip_a: vs.VideoNode,
    *args: Any,
    exclusive: bool = True,
    mismatch: bool = False,
    prop_src: vs.VideoNode | list[vs.VideoNode] | None = None,
) -> vs.VideoNode:
    if len(args) == 0:
        return clip_a

    if isinstance(clip_b := args[0], vs.VideoNode):
        ranges: FrameRangeN | FrameRangesN | vstools.functions.ranges._RangesCallBackLike | None = args[1]

        if exclusive and not callable(ranges):
            ranges = normalise_ranges(clip_b, ranges, norm_dups=True)

        return vstools.replace_ranges(clip_a, clip_b, ranges, exclusive, mismatch, prop_src=prop_src)

    if not exclusive:
        raise NotImplementedError

    rclips: tuple[tuple[vs.VideoNode, FrameRangeN | FrameRangesN | vstools.functions.ranges._RangesCallBack], ...] = (
        args
    )

    if len(rclips) <= 10:
        for c, r in rclips:
            clip_a = replace_ranges(clip_a, c, r, mismatch=mismatch)
        return clip_a

    ref_indices = np.zeros(clip_a.num_frames, np.uint32)

    rrclips = [
        (c, np.fromiter(ranges_to_indices.gen_indices(c, r, (0, i)), np.uint32, c.num_frames))
        for (i, (c, r)) in enumerate(rclips, 1)
    ]

    clips, indices_iter = cast(tuple[Iterator[vs.VideoNode], Iterator[np.typing.NDArray[AnyInt]]], zip(*rrclips))

    indices = list[np.typing.NDArray[AnyInt]]()

    for i in indices_iter:
        if (isize := i.size) < (rsize := ref_indices.size):
            i = np.pad(i, (0, rsize - isize))
        elif isize > rsize:
            i = i[:rsize]

        indices.append(i)

    nindices = np.max([ref_indices, *indices], axis=0, out=ref_indices)

    return select_frames(
        [clip_a, *clips], VNumpy.zip_arrays(nindices, np.arange(clip_a.num_frames, dtype=np.uint32)), mismatch=mismatch
    )


class Rescale(vsscale.Rescale):
    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: int | float,
        kernel: ComplexKernelLike,
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
