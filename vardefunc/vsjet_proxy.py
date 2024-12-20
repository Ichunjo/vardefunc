from __future__ import annotations

import logging

from functools import lru_cache
from typing import Any, Iterator, Literal, Sequence, SupportsFloat, cast, overload

import numpy as np
import vsdenoise
import vsmasktools
import vstools

from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import Catrom, KernelT, Point
from vsmasktools import GenericMaskT, Morpho, SobelStd, XxpandMode, normalize_mask
from vsrgtools.util import mean_matrix
from vstools import (
    ColorRange, CustomStrEnum, FrameRangeN, FrameRangesN, MatrixT, copy_signature, core,
    scale_value, set_output, vs
)

from .types import AnyInt, NDArray, RangesCallBack, RangesCallBackF, RangesCallBackNF, RangesCallBackT
from .types import VNumpy as vnp
from .util import normalise_ranges, ranges_to_indices, to_incl_incl, select_frames

__all__ = [
    "is_preview", "set_output", "replace_ranges", "BestestSource",
    "DeferredMask", "HardsubASS", "HardsubLine", "HardsubLineFade", "HardsubMask",
    "HardsubSign", "HardsubSignFades",
    "replace_squaremask",
    "dpir"
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


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN,
    /, *,
    exclusive: bool = True, mismatch: bool = False,
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: RangesCallBack,
    /, *,
    mismatch: bool = False,
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: RangesCallBackF[vs.VideoFrame] | RangesCallBackNF[vs.VideoFrame],
    /, *,
    mismatch: bool = False,
    prop_src: vs.VideoNode
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: RangesCallBackF[list[vs.VideoFrame]] | RangesCallBackNF[list[vs.VideoFrame]],
    /, *,
    mismatch: bool = False,
    prop_src: list[vs.VideoNode]
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, *clip_b: tuple[vs.VideoNode, FrameRangeN | FrameRangesN | RangesCallBack],
    mismatch: bool = False,
) -> vs.VideoNode:
    ...

def replace_ranges(
    clip_a: vs.VideoNode, *args: Any,
    exclusive: bool = True, mismatch: bool = False,
    prop_src: vs.VideoNode | list[vs.VideoNode] | None = None
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
    if isinstance(clip_b := args[0], vs.VideoNode):
        ranges: FrameRangeN | FrameRangesN | RangesCallBackT | None = args[1]

        if exclusive and not callable(ranges):
            ranges = normalise_ranges(clip_b, ranges, norm_dups=True)

        return vstools.replace_ranges(clip_a, clip_b, ranges, exclusive, mismatch, prop_src=prop_src)

    rclips: tuple[tuple[vs.VideoNode, FrameRangeN | FrameRangesN | RangesCallBack], ...] = args

    if not exclusive:
        raise NotImplementedError

    if len(rclips) == 0:
        return clip_a

    if len(rclips) == 1:
        (c, r), = rclips
        return replace_ranges(clip_a, c, r, mismatch=mismatch)

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
        [clip_a, *clips],
        vnp.zip_arrays(nindices, np.arange(clip_a.num_frames, dtype=np.uint32)),
        mismatch=mismatch
    )


try:
    from vssource import BestSource
except ImportError:
    pass
else:
    class BestestSource(BestSource):
        def __init__(self, *, force: bool = True, **kwargs: Any) -> None:
            kwargs.setdefault("showprogress", True)
            kwargs.setdefault("cachemode", 3)
            super().__init__(force=force, **kwargs)

            def handler_func(m_type: vs.MessageType, msg: str) -> None:
                if all([
                    m_type == vs.MESSAGE_TYPE_INFORMATION,
                    msg.startswith(("VideoSource ", "AudioSource ")),
                    logging.getLogger().level <= logging.WARNING,
                    is_preview()
                ]):
                    print(msg, end="\r")

            self._log_handle = core.add_log_handler(handler_func)

        def __del__(self) -> None:
            core.remove_log_handler(self._log_handle)


class DeferredMask(vsmasktools.DeferredMask):
    _incl_excl_ranges: FrameRangesN

    @property
    def ranges(self) -> FrameRangesN:
        return [
            (s, (e - 1) if e is not None else e)
            for (s, e) in normalise_ranges(None, self._incl_excl_ranges, norm_dups=True)
        ]

    @ranges.setter
    def ranges(self, value: FrameRangesN) -> None:
        self._incl_excl_ranges = value


class HardsubMask(vsmasktools.HardsubMask, DeferredMask): ...


# Waiting for https://github.com/Jaded-Encoding-Thaumaturgy/vs-masktools/pull/29
class HardsubSignFades(vsmasktools.HardsubSignFades, HardsubMask):
    def __init__(
        self, *args: Any, highpass: float = 0.0763, expand: int = 8, edgemask: GenericMaskT = SobelStd,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, highpass=highpass, expand=expand, edgemask=edgemask, **kwargs)
        self.expand_mode = expand_mode

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clipedge, refedge = (
            normalize_mask(self.edgemask, x, **kwargs).std.Convolution(mean_matrix)
            for x in (clip, ref)
        )

        highpass = scale_value(self.highpass, 32, clip, ColorRange.FULL)

        mask = norm_expr(
            [clipedge, refedge], f'x y - {highpass} < 0 {ExprToken.RangeMax} ?'
        ).std.Median()

        return Morpho.inflate(Morpho.expand(mask, self.expand, mode=self.expand_mode), iterations=4)


# Waiting for https://github.com/Jaded-Encoding-Thaumaturgy/vs-masktools/pull/29
class HardsubSign(vsmasktools.HardsubSign, HardsubMask):
    def __init__(
        self, *args: Any, thr: float = 0.06, minimum: int = 1, expand: int = 8, inflate: int = 7,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, thr=thr, minimum=minimum, expand=expand, inflate=inflate, **kwargs)
        self.expand_mode = expand_mode

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        hsmf = norm_expr([clip, ref], 'x y - abs')
        hsmf = Point.resample(hsmf, clip.format.replace(subsampling_w=0, subsampling_h=0))  # type: ignore

        hsmf = ExprOp.MAX(hsmf, split_planes=True)

        hsmf = Morpho.binarize(hsmf, self.thr)
        hsmf = Morpho.minimum(hsmf, iterations=self.minimum)
        hsmf = Morpho.expand(hsmf, self.expand, mode=self.expand_mode)
        hsmf = Morpho.inflate(hsmf, iterations=self.inflate)

        return hsmf.std.Limiter()


class HardsubLine(vsmasktools.HardsubLine, HardsubMask): ...


class HardsubLineFade(vsmasktools.HardsubLineFade, HardsubMask): ...


class HardsubASS(vsmasktools.HardsubASS, HardsubMask): ...


@copy_signature(vsmasktools.replace_squaremask)
def replace_squaremask(*args: Any, **kwargs: Any) -> Any:
    argsl = list(args)
    argsl[3] = to_incl_incl(normalise_ranges(
        kwargs.get("clipa", argsl[0]), kwargs.pop("ranges", argsl[3]), norm_dups=True
    ))

    return vsmasktools.replace_squaremask(*argsl, **kwargs)


StrengthT = SupportsFloat | vs.VideoNode | None


class _dpir(CustomStrEnum):
    DEBLOCK: _dpir = 'deblock'  # type: ignore
    DENOISE: _dpir = 'denoise'  # type: ignore

    def __call__(
        self, clip: vs.VideoNode, strength: StrengthT | tuple[StrengthT, StrengthT] = 10,
        matrix: MatrixT | None = None, cuda: bool | Literal['trt'] | None = None, i444: bool = False,
        tiles: int | tuple[int, int] | None = None, overlap: int | tuple[int, int] | None = 8,
        zones: Sequence[tuple[FrameRangeN | FrameRangesN | None, StrengthT]] | None = None,
        fp16: bool | None = None, num_streams: int | None = None, device_id: int = 0, kernel: KernelT = Catrom,
        **kwargs: Any
    ) -> vs.VideoNode:
        if zones:
            zones = [(to_incl_incl(normalise_ranges(clip, r, norm_dups=True)), stre) for r, stre in zones]

        return vsdenoise.deblock._dpir(self.value)(
            clip, strength, matrix, cuda, i444, tiles, overlap,
            zones, fp16, num_streams, device_id, kernel, **kwargs  # type: ignore
        )


dpir = _dpir.DEBLOCK
