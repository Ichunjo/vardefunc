"""Helper functions for the main functions in this module"""
from functools import partial, wraps
from string import ascii_lowercase
from typing import (Any, Callable, List, Optional, Sequence, Tuple, Union,
                    cast, overload)

import vapoursynth as vs
from vsutil import Range as CRange
from vsutil import depth

from .types import FF
from .types import DuplicateFrame as DF
from .types import Range, Trim

core = vs.core


class FormatError(Exception):
    """Raised when a format of VideoNode object is not allowed."""


@overload
def finalise_output(*, bits: int = 10, clamp_range: bool = True) -> FF:  # type: ignore
    ...


@overload
def finalise_output(func: Optional[FF] = None, /) -> FF:
    ...


@overload
def finalise_output(func: Optional[FF] = None, /, *, bits: int = 10, clamp_range: bool = True) -> FF:
    ...


def finalise_output(func: Optional[FF] = None, /, *, bits: int = 10, clamp_range: bool = True) -> FF:
    """Decorator to dither down the final output clip and clamp range to legal values"""
    if func is None:
        return cast(FF, partial(finalise_output, bits=bits, clamp_range=clamp_range))

    @wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert func
        out = func(*args, **kwargs)
        rng = CRange.FULL if get_colour_range(out) == 0 else CRange.LIMITED
        out = depth(out, bits, range=rng, range_in=rng)
        if rng == CRange.LIMITED:
            out = out.std.Limiter(16 << (bits - 8), [235 << (bits - 8), 240 << (bits - 8)], [0, 1, 2])
        return out

    return cast(FF, _wrapper)


def get_colour_range(clip: vs.VideoNode) -> int:
    """Get the colour range from the VideoProps"""
    return cast(int, clip.get_frame(0).props['_ColorRange'])


def get_sample_type(clip: vs.VideoNode) -> vs.SampleType:
    """Returns the sample type of a VideoNode as an SampleType."""
    if clip.format is None:
        raise FormatError('Variable format not allowed!')
    return clip.format.sample_type


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


def normalise_ranges(clip: vs.VideoNode, ranges: Union[Range, List[Range], Trim, List[Trim]]) -> List[Tuple[int, int]]:
    """Modified version of lvsfunc.util.normalize_ranges following python slicing syntax"""
    ranges = ranges if isinstance(ranges, list) else [ranges]

    out: List[Tuple[int, int]] = []
    for r in ranges:
        if isinstance(r, tuple):
            start, end = r
            if start is None:
                start = 0
            if end is None:
                end = clip.num_frames
        else:
            start = r
            end = r + 1
        if start < 0:
            start += clip.num_frames
        if end <= 0:
            end += clip.num_frames
        out.append((start, end))

    return out


def replace_ranges(clip_a: vs.VideoNode, clip_b: vs.VideoNode,
                   ranges: Union[Range, List[Range]], *, mismatch: bool = False) -> vs.VideoNode:
    """Modified version of lvsfunc.util.replace_ranges following python slicing syntax"""
    out = clip_a

    nranges = normalise_ranges(clip_b, ranges)

    for start, end in nranges:
        tmp = clip_b[start:end]
        if start != 0:
            tmp = core.std.Splice([out[:start], tmp], mismatch=mismatch)
        if end < out.num_frames:
            tmp = core.std.Splice([tmp, out[end:]], mismatch=mismatch)
        out = tmp

    return out


def adjust_clip_frames(clip: vs.VideoNode, trims_or_dfs: List[Union[Trim, DF]]) -> vs.VideoNode:
    """Trims and/or duplicates frames"""
    clips: List[vs.VideoNode] = []
    for trim_or_df in trims_or_dfs:
        if isinstance(trim_or_df, tuple):
            start, end = normalise_ranges(clip, trim_or_df)[0]
            clips.append(clip[start:end])
        else:
            df = trim_or_df
            clips.append(clip[df] * df.dup)
    return core.std.Splice(clips)


def pick_px_op(use_expr: bool,
               operations: Tuple[str, Union[Sequence[int], Sequence[float], int, float, Callable[..., Any]]]
               ):
    """Pick either std.Lut or std.Expr
       Returns partial[VideoNode]"""
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
