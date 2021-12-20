"""Helper functions for the main functions in this module"""
from __future__ import annotations

import inspect
import warnings
from fractions import Fraction
from functools import partial, wraps
from string import ascii_lowercase
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence, Set,
                    Tuple, cast, overload)

import numpy as np
import vapoursynth as vs
from pytimeconv import Convert
from vsutil import depth

from .types import (CHROMA_LOCATION, COLOUR_RANGE, F_VN, MATRIX, PRIMARIES,
                    TRANSFER, AnyInt)
from .types import DuplicateFrame as DF
from .types import F, NDArray, Range, Trim
from .types import VNumpy as vnp
from .types import format_not_none

core = vs.core


def finalise_clip(clip: vs.VideoNode, bits: int = 10, clamp_tv_range: bool = True) -> vs.VideoNode:
    """
    Converts bitdepth and optionally clamps the pixel values in TV range

    Args:
        clip (vs.VideoNode):
            Source clip

        bits (int, optional):
            Target bitdepth. Defaults to 10.

        clamp_tv_range (bool, optional):
            Clamp in TV range or not. Defaults to True.

    Returns:
        vs.VideoNode: Finalised clip
    """
    out = depth(clip, bits)
    if clamp_tv_range:
        out = out.std.Expr([f'x {16 << (bits - 8)} max {235 << (bits - 8)} min',
                            f'x {16 << (bits - 8)} max {240 << (bits - 8)} min'])
    return out


@overload
def finalise_output(*, bits: int = 10, clamp_tv_range: bool = True) -> Callable[[F_VN], F_VN]:
    ...


@overload
def finalise_output(func: Optional[F_VN], /) -> F_VN:
    ...


def finalise_output(func: Optional[F_VN] = None, /, *, bits: int = 10, clamp_tv_range: bool = True
                    ) -> Callable[[F_VN], F_VN] | F_VN:
    """
    Decorator implementation of ``finalise_clip``
    """
    if func is None:
        return cast(
            Callable[[F_VN], F_VN],
            partial(finalise_output, bits=bits, clamp_tv_range=clamp_tv_range)
        )

    @wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert func
        return finalise_clip(func(*args, **kwargs), bits, clamp_tv_range)

    return cast(F_VN, _wrapper)


def initialise_clip(
    clip: vs.VideoNode, bits: int = 16,
    matrix: vs.MatrixCoefficients | MATRIX = vs.MATRIX_BT709,
    transfer: vs.TransferCharacteristics | TRANSFER = vs.TRANSFER_BT709,
    primaries: vs.ColorPrimaries | PRIMARIES = vs.PRIMARIES_BT709,
    chroma_location: vs.ChromaLocation | CHROMA_LOCATION = vs.CHROMA_LEFT,
    colour_range: vs.ColorRange | COLOUR_RANGE = vs.RANGE_LIMITED
) -> vs.VideoNode:
    """
    Initialise a clip by converting its bitdepth and setting its VideoProps

    Args:
        clip (vs.VideoNode):
            Source clip

        bits (int, optional):
            Target bitdepth. Defaults to 16.

        matrix (vs.MatrixCoefficients, optional):
            Matrix coefficients. Defaults to vs.MATRIX_BT709.

        transfer (vs.TransferCharacteristics, optional):
            Transfer characteristics. Defaults to vs.TRANSFER_BT709.

        primaries (vs.ColorPrimaries, optional):
            Colour primaries. Defaults to vs.PRIMARIES_BT709.

        chroma_location (vs.ChromaLocation, optional):
            Chroma location. Defaults to vs.CHROMA_LEFT.

        colour_range (vs.ColorRange, optional):
            Colour range. Defaults to vs.RANGE_LIMITED.

    Returns:
        vs.VideoNode:
            Initialised clip
    """
    return depth(
        clip.std.SetFrameProps(
            _Matrix=matrix, _Transfer=transfer, _Primaries=primaries,
            _ChromaLocation=chroma_location, _ColorRange=colour_range
        ),
        bits
    )


@overload
def initialise_input(
    *, bits: int = ...,
    matrix: vs.MatrixCoefficients | MATRIX = ...,
    transfer: vs.TransferCharacteristics | TRANSFER = ...,
    primaries: vs.ColorPrimaries | PRIMARIES = ...,
    chroma_location: vs.ChromaLocation | CHROMA_LOCATION = ...
) -> Callable[[F_VN], F_VN]:
    ...


@overload
def initialise_input(func: Optional[F_VN], /) -> F_VN:
    ...


def initialise_input(
    func: Optional[F_VN] = None, /, *, bits: int = 16,
    matrix: vs.MatrixCoefficients | MATRIX = vs.MATRIX_BT709,
    transfer: vs.TransferCharacteristics | TRANSFER = vs.TRANSFER_BT709,
    primaries: vs.ColorPrimaries | PRIMARIES = vs.PRIMARIES_BT709,
    chroma_location: vs.ChromaLocation | CHROMA_LOCATION = vs.CHROMA_LEFT,
    colour_range: vs.ColorRange | COLOUR_RANGE = vs.RANGE_LIMITED
) -> Callable[[F_VN], F_VN] | F_VN:
    """
    Decorator implementation of ``initialise_clip``
    """
    if func is None:
        return cast(
            Callable[[F_VN], F_VN],
            partial(initialise_input, bits=bits, matrix=matrix, transfer=transfer, primaries=primaries)
        )

    init_args: Dict[str, Any] = dict(
        bits=bits,
        matrix=matrix, transfer=transfer, primaries=primaries,
        chroma_location=chroma_location, colour_range=colour_range
    )

    @wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert func

        args_l = list(args)
        for i, obj in enumerate(args_l):
            if isinstance(obj, vs.VideoNode):
                args_l[i] = initialise_clip(obj, **init_args)
                return func(*args_l, **kwargs)

        kwargs2 = kwargs.copy()
        for name, obj in kwargs2.items():
            if isinstance(obj, vs.VideoNode):
                kwargs2[name] = initialise_clip(obj, **init_args)
                return func(*args, **kwargs2)

        for name, param in inspect.signature(func).parameters.items():
            if param.default is not inspect.Parameter.empty and isinstance(param.default, vs.VideoNode):
                return func(*args, **kwargs2 | {name: initialise_clip(param.default, **init_args)})

        raise ValueError(
            'initialise_input: None VideoNode found in positional, keyword nor default arguments!'
        )

    return cast(F_VN, _wrapper)


def get_sample_type(clip: vs.VideoNode) -> vs.SampleType:
    """Returns the sample type of a VideoNode as an SampleType."""
    return format_not_none(clip).format.sample_type


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


def select_frames(clips: vs.VideoNode | Sequence[vs.VideoNode],
                  indices: NDArray[AnyInt] | List[int] | List[Tuple[int, int]],
                  *, mismatch: bool = False) -> vs.VideoNode:
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
        indices = vnp.zip_arrays(
            np.zeros(len(indices), np.uint32),
            indices
        )
    elif indices.ndim == 2:
        pass
    else:
        raise ValueError('select_frames: only 1D and 2D array is allowed!')

    plh = clips[0].std.BlankClip(length=len(indices))

    if mismatch:
        if plh.format and plh.format.id == vs.GRAY8:
            ph_fmt = vs.GRAY16
        else:
            ph_fmt = vs.GRAY8
        plh = core.std.Splice(
            [plh[:-1], plh.std.BlankClip(plh.width+1, plh.height+1, format=ph_fmt, length=1)],
            True
        )

    def _select_func(n: int, clips: Sequence[vs.VideoNode], indices: NDArray[AnyInt]) -> vs.VideoNode:
        # index: NDArray[AnyInt] = indices[n]  # Get the index / num_frame pair
        # i_clip = int(index[0])  # Get the index
        # num = int(index[1])  # Get the num_frame
        # nclip = clips[i_clip]  # Select the clip to be returned
        # tclip = nclip[num]  # Slice the clip
        # return tclip
        return clips[int(indices[n][0])][int(indices[n][1])]

    return core.std.FrameEval(plh, partial(_select_func, clips=clips, indices=indices))


def normalise_ranges(clip: vs.VideoNode | vs.AudioNode, ranges: Range | List[Range] | Trim | List[Trim],
                     *, norm_dups: bool = False, ref_fps: Optional[Fraction] = None) -> List[Tuple[int, int]]:
    """Modified version of lvsfunc.util.normalize_ranges following python slicing syntax"""
    if isinstance(clip, vs.VideoNode):
        num_frames = clip.num_frames
    else:
        if ref_fps is not None:
            num_frames = clip.num_samples
        else:
            num_frames = clip.num_frames

    ranges = ranges if isinstance(ranges, list) else [ranges]

    nranges: Set[Tuple[int, int]] = set()
    f2s = Convert.f2samples
    for r in ranges:
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
            start = f2s(start, ref_fps, clip.sample_rate)
            end = f2s(end, ref_fps, clip.sample_rate)
        if start < 0:
            start += num_frames
        if end <= 0:
            end += num_frames

        if start >= num_frames or end > num_frames:
            core.log_message(vs.MESSAGE_TYPE_WARNING, f'normalise_ranges: "{r}" out of range')
            warnings.warn(f'normalise_ranges: {r} out of range')

        start, end = min(start, num_frames - 1), min(end, num_frames)
        nranges.add((start, end))

    out = sorted(nranges)

    if norm_dups:
        nranges_d = dict(out)
        nranges_rev = sorted(nranges_d.items(), reverse=True)

        for (start1, end1), (start2, end2) in zip(nranges_rev, nranges_rev[1:]):
            if start2 < start1 <= end2 < end1:
                nranges_d[start2] = max(end1, nranges_d[start1])
                del nranges_d[start1]
            if start2 < start1 and end1 <= end2:
                del nranges_d[start1]

        out = list(nranges_d.items())

    return out


def replace_ranges(clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: Range | List[Range],
                   *, mismatch: bool = False) -> vs.VideoNode:
    """Modified version of lvsfunc.util.replace_ranges following python slicing syntax"""
    num_frames = clip_a.num_frames
    nranges = normalise_ranges(clip_a, ranges)

    def _gen_indices(nranges: List[Tuple[int, int]]) -> Iterable[int]:
        for f in range(num_frames):
            i = 0
            for start, end in nranges:
                if start <= f < end:
                    i = 1
                    break
            yield i

    indices = vnp.zip_arrays(
        np.fromiter(_gen_indices(nranges), np.uint32, num_frames),
        np.arange(num_frames, dtype=np.uint32)
    )

    return select_frames([clip_a, clip_b], indices, mismatch=mismatch)


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


def remap_rfs(clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: Range | List[Range]) -> vs.VideoNode:
    """Replace ranges function using remap plugin"""
    return core.remap.ReplaceFramesSimple(
        clip_a, clip_b,
        mappings=' '.join(f'[{s} {e-1}]' for s, e in normalise_ranges(clip_a, ranges, norm_dups=True))
    )


def pick_px_op(
    use_expr: bool,
    operations: Tuple[str, Sequence[int] | Sequence[float] | int | float | F]
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
                func = partial(core.std.Lut, lut=lut)
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
