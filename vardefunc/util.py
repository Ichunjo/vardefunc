"""Helper functions for the main functions in this module"""
from __future__ import annotations

__all__ = [
    'select_frames', 'normalise_ranges', 'ranges_to_indices',
    'adjust_clip_frames', 'adjust_audio_frames',
    'remap_rfs'
]

import math
import warnings

from fractions import Fraction
from functools import partial
from string import ascii_lowercase
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, cast, overload

import numpy as np
import vapoursynth as vs
from vstools import FrameRangeN, FrameRangesN

from pytimeconv import Convert

from .types import AnyInt
from .types import DuplicateFrame as DF
from .types import NDArray, Range, Trim
from .types import VNumpy as vnp

core = vs.core


def load_operators_expr() -> List[str]:
    """Returns clip loads operators for std.Expr as a list of string."""
    abcd = list(ascii_lowercase)
    return abcd[-3:] + abcd[:-3]


def mae_expr(gray_only: bool = True) -> str:
    """Mean Absolute Error string to be integrated in std.Expr.

    Args:
        gray_only (bool, optional):
            If both actual observation and prediction are one plane each.
            Defaults to True.

    Returns:
        str: Expression.
    """
    return 'x y - abs' if gray_only else 'x a - abs y b - abs max z c - abs max'


def max_expr(n: int) -> str:
    """Dynamic variable max string to be integrated in std.Expr.

    Args:
        n (int): Number of elements.

    Returns:
        str: Expression.
    """
    return 'x y max ' + ' max '.join(
        load_operators_expr()[i] for i in range(2, n)
    ) + ' max'


def select_frames(
    clips: vs.VideoNode | Sequence[vs.VideoNode],
    indices: NDArray[AnyInt] | List[int] | List[Tuple[int, int]],
    *, mismatch: bool = False
) -> vs.VideoNode:
    """
    Select frames from one or more clips at specified indices.
    Written by EoE. Modified by me.

    Args:
        clips (Union[vs.VideoNode, Sequence[vs.VideoNode]]):
            A clip or a list of clips to select the frames from

        indices (Union[NDArray[AnyInt], List[int], List[Tuple[int, int]]]):
            Indices of frames to select. Provide a list of indices for a single clip,
            or for multiple clips, a list of tuples in the form ``(clip_index, frame_index)``

        mismatch (bool, optional):
            Splicing clips with different formats or dimensions is considered an error
            unless mismatch is true. Defaults to False.

    Returns:
        vs.VideoNode: The selected frames in a single clip
    """

    clips = (clips, ) if isinstance(clips, vs.VideoNode) else clips
    indices = vnp.array(indices) if isinstance(indices, list) else indices

    if indices.ndim == 1:
        indices = vnp.zip_arrays(np.zeros(len(indices), np.uint32), indices)
    elif indices.ndim == 2:
        pass
    else:
        raise ValueError('select_frames: only 1D and 2D array is allowed!')

    plh = clips[0].std.BlankClip(length=len(indices))

    if mismatch:
        plh = plh.std.BlankClip(varsize=True, varformat=True)

    def _select_func(n: int, clips: Sequence[vs.VideoNode], indices: NDArray[AnyInt]) -> vs.VideoNode:
        # index: NDArray[AnyInt] = indices[n]  # Get the index / num_frame pair
        # i_clip = int(index[0])  # Get the index
        # num = int(index[1])  # Get the num_frame
        # nclip = clips[i_clip]  # Select the clip to be returned
        # tclip = nclip[num]  # Slice the clip
        # return tclip
        return clips[int(indices[n][0])][int(indices[n][1])]

    return core.std.FrameEval(plh, partial(_select_func, clips=clips, indices=indices))


@overload
def normalise_ranges(
    clip: vs.VideoNode, ranges: FrameRangeN | FrameRangesN, *, norm_dups: bool = False
) -> list[Range]:
    ...


@overload
def normalise_ranges(
    clip: vs.AudioNode, ranges: FrameRangeN | FrameRangesN, *, norm_dups: bool = False, ref_fps: Fraction | None = None
) -> list[Range]:
    ...


@overload
def normalise_ranges(
    clip: None, ranges: FrameRangeN | FrameRangesN, *, norm_dups: bool = False,
) -> list[tuple[int, int | None]]:
    ...

def normalise_ranges(
    clip: vs.VideoNode | vs.AudioNode | None, ranges: FrameRangeN | FrameRangesN,
    *, norm_dups: bool = False, ref_fps: Fraction | None = None
) -> list[Range] | list[tuple[int, int | None]]:
    """Modified version of lvsfunc.util.normalize_ranges following python slicing syntax"""
    if isinstance(clip, vs.VideoNode):
        num_frames = clip.num_frames
    elif isinstance(clip, vs.AudioNode):
        if ref_fps is not None:
            num_frames = clip.num_samples
        else:
            num_frames = clip.num_frames
    else:
        num_frames = None

    if ranges is None:
        return [(0, num_frames)]

    def _resolve_ranges_type(
        rngs: int | tuple[int | None, int | None] | FrameRangesN
    ) -> Sequence[int | tuple[int | None, int | None] | None]:
        if isinstance(rngs, int):
            return [rngs]
        if isinstance(rngs, tuple) and len(rngs) == 2:
            if isinstance(rngs[0], int) or rngs[0] is None and isinstance(rngs[1], int) or rngs[1] is None:
                return [cast(tuple[int | None, int | None], rngs)]
            else:
                raise
        rngs = cast(FrameRangesN, rngs)
        return rngs

    ranges = _resolve_ranges_type(ranges)

    nranges = set[tuple[int, int | None]]()
    f2s = Convert.f2samples

    for r in ranges:

        if r is None:
            r = (None, None)

        if isinstance(r, tuple):
            start, end = r
            if start is None:
                start = 0
            if end is None:
                end = num_frames
        else:
            start = r
            end = r + 1

        if isinstance(clip, vs.AudioNode) and ref_fps is not None:
            if start != 0:
                start = f2s(start, ref_fps, clip.sample_rate)
            if end != num_frames and end:
                end = f2s(end, ref_fps, clip.sample_rate)
    
        if start < 0 and num_frames is not None:
            start += num_frames
        if end is not None and end <= 0 and num_frames is not None:
            end += num_frames

        if end is not None:
            if start > end:
                warnings.warn(f'normalise_ranges: start frame "{start}" is higher than end frame "{end}"')
    
            if num_frames is not None:
                if start >= num_frames or end > num_frames:
                    warnings.warn(f'normalise_ranges: {r} out of range')

        if num_frames is not None:
            start = min(start, num_frames - 1)
            if end is not None:
                end = min(end, num_frames)

        nranges.add((start, end))

    out = sorted(nranges)

    if norm_dups:
        nranges_d = dict(out)
        nranges_rev = sorted(nranges_d.items(), reverse=True)

        for (start1, end1), (start2, end2) in zip(nranges_rev, nranges_rev[1:]):
            if end1 is None or end2 is None:
                continue

            if start2 < start1 <= end2 < end1:
                nranges_d[start2] = max(end1, nranges_d[start1], key=lambda x: x if x is not None else math.inf)
                del nranges_d[start1]

            if start2 < start1 and end1 <= end2:
                del nranges_d[start1]

        out = list(nranges_d.items())

    return out


def ranges_to_indices(ref: vs.VideoNode, ranges: FrameRangeN | FrameRangesN) -> NDArray[AnyInt]:
    nranges = normalise_ranges(ref, ranges)

    def _gen_indices(nranges: list[tuple[int, int]]) -> Iterable[int]:
        for f in range(ref.num_frames):
            i = 0
            for start, end in nranges:
                if start <= f < end:
                    i = 1
                    break
            yield i

    return vnp.zip_arrays(
        np.fromiter(_gen_indices(nranges), np.uint32, ref.num_frames),
        np.arange(ref.num_frames, dtype=np.uint32)
    )


def adjust_clip_frames(clip: vs.VideoNode, trims_or_dfs: List[Trim | DF] | Trim) -> vs.VideoNode:
    """Trims and/or duplicates frames"""
    trims_or_dfs = [trims_or_dfs] if isinstance(trims_or_dfs, tuple) else trims_or_dfs
    indices: List[int] = []
    for trim_or_df in trims_or_dfs:
        if isinstance(trim_or_df, tuple):
            ntrim = normalise_ranges(clip, trim_or_df).pop()
            indices.extend(range(*ntrim))
        else:
            df = trim_or_df
            indices.extend([df.numerator] * df.dup)
    return select_frames(clip, indices)


def adjust_audio_frames(audio: vs.AudioNode, trims_or_dfs: List[Trim | DF] | Trim, *, ref_fps: Optional[Fraction] = None) -> vs.AudioNode:
    audios: List[vs.AudioNode] = []
    trims_or_dfs = [trims_or_dfs] if isinstance(trims_or_dfs, tuple) else trims_or_dfs
    for trim_or_df in trims_or_dfs:
        if isinstance(trim_or_df, tuple):
            ntrim = normalise_ranges(audio, trim_or_df, ref_fps=ref_fps).pop()
            audios.append(audio[slice(*ntrim)])
        else:
            df = trim_or_df
            if ref_fps:
                df = df.to_samples(ref_fps, audio.sample_rate)
            audios.append(audio[int(df)] * df.dup)
    return core.std.AudioSplice(audios)


def remap_rfs(clip_a: vs.VideoNode, clip_b: vs.VideoNode,ranges: FrameRangeN | FrameRangesN) -> vs.VideoNode:
    """Replace ranges function using remap plugin"""
    return core.remap.ReplaceFramesSimple(
        clip_a, clip_b,
        mappings=' '.join(f'[{s} {e-1}]' for s, e in normalise_ranges(clip_a, ranges, norm_dups=True))
    )


def pick_px_op(
    use_expr: bool,
    operations: Tuple[str, Sequence[int] | Sequence[float] | int | float | Callable[..., Any]]
) -> Callable[..., vs.VideoNode]:
    """Pick either std.Lut or std.Expr"""
    expr, lut = operations
    if use_expr:
        func = partial(core.std.Expr, expr=expr)
    else:
        if callable(lut):
            func = partial(core.std.Lut, function=lut)
        elif isinstance(lut, Sequence):
            if all(isinstance(x, int) for x in lut):
                func = partial(core.std.Lut, lut=lut)  # type: ignore
            elif all(isinstance(x, float) for x in lut):
                func = partial(core.std.Lut, lutf=lut)
            else:
                raise ValueError('pick_px_operation: operations[1] is not a valid type!')
        elif isinstance(lut, int):
            func = partial(core.std.Lut, lut=lut)
        elif isinstance(lut, float):
            func = partial(core.std.Lut, lutf=lut)
        else:
            raise ValueError('pick_px_operation: operations[1] is not a valid type!')
    return func


def rmse_expr(gray_only: bool = True) -> str:
    """Root Mean Squared Error string to be integrated in std.Expr.

    Args:
        gray_only (bool, optional):
            If both actual observation and prediction are one plane each.
            Defaults to True.

    Returns:
        str: Expression.
    """
    return 'x y - dup * sqrt' if gray_only else 'x a - dup * sqrt y b - dup * sqrt max z c - dup * sqrt max'
