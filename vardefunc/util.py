"""Helper functions for the main functions in this module"""
from functools import partial, wraps
from string import ascii_lowercase
from typing import Any, Callable, List, Sequence, Tuple, Union

import vapoursynth as vs

from .types import DuplicateFrame as DF
from .types import Range, Trim

core = vs.core


class FormatError(Exception):
    """Raised when a format of VideoNode object is not allowed."""


def copy_docstring_from(source: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator intended to copy the docstring from an other function

    Args:
        source (Callable[..., Any]): Source function.

    Returns:
        Callable[..., Any]: Function decorated
    """
    @wraps(source)
    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        func.__doc__ = source.__doc__
        return func
    return wrapper


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


def normalise_ranges(clip: vs.VideoNode, ranges: Union[Range, List[Range]]) -> List[Tuple[int, int]]:
    """Normalise `Range` to a list of an inclusive/exclusive couple positive integer ranges.

    Args:
        clip (vs.VideoNode):
            Reference clip used for length.

        ranges (Union[Range, List[Range]]):
            Single `Range` or list of `Range`.

    Returns:
        List[Tuple[int, int]]: List of inclusive/exclusive ranges.
    """
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
