"""Helper functions for the main functions in this module"""
import inspect
import warnings
from functools import partial, wraps
from string import ascii_lowercase
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Set,
                    Tuple, Union, cast, overload)

import vapoursynth as vs
from vsutil import Range as CRange
from vsutil import depth

from .types import F_VN, MATRIX, PRIMARIES, TRANSFER
from .types import DuplicateFrame as DF
from .types import F, PropsVal, Range, Trim
from .types import Zimg, format_not_none

core = vs.core


@overload
def finalise_output(func: Optional[F_VN], /) -> F_VN:
    ...


@overload
def finalise_output(*, bits: int = 10, clamp_tv_range: bool = True) -> Callable[[F_VN], F_VN]:
    ...


def finalise_output(func: Optional[F_VN] = None, /, *, bits: int = 10, clamp_tv_range: bool = True
                    ) -> Union[Callable[[F_VN], F_VN], F_VN]:
    """
    Function decorator that dither down the final output clip and clamp range to legal values.

    Decorated `func`'s output must be of type `vapoursynth.VideoNode`.
    """
    if func is None:
        return cast(
            Callable[[F_VN], F_VN],
            partial(finalise_output, bits=bits, clamp=clamp_tv_range)
        )

    @wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert func
        out = func(*args, **kwargs)
        out = depth(out, bits)
        if clamp_tv_range:
            out = out.std.Limiter(16 << (bits - 8), [235 << (bits - 8), 240 << (bits - 8)], [0, 1, 2])
        return out

    return cast(F_VN, _wrapper)


@overload
def initialise_input(func: Optional[F_VN], /) -> F_VN:
    ...


@overload
def initialise_input(
    *, bits: int = ...,
    matrix: Union[Zimg.Matrix, MATRIX] = ...,
    transfer: Union[Zimg.Transfer, TRANSFER] = ...,
    primaries: Union[Zimg.Primaries, PRIMARIES] = ...
) -> Callable[[F_VN], F_VN]:
    ...


def initialise_input(
    func: Optional[F_VN] = None, /, *, bits: int = 16,
    matrix: Union[Zimg.Matrix, MATRIX] = Zimg.Matrix.BT709,
    transfer: Union[Zimg.Transfer, TRANSFER] = Zimg.Transfer.BT709,
    primaries: Union[Zimg.Primaries, PRIMARIES] = Zimg.Primaries.BT709
) -> Union[Callable[[F_VN], F_VN], F_VN]:
    """
    Function decorator that dither up the input clip and set matrix, transfer and primaries.
    """
    if func is None:
        return cast(
            Callable[[F_VN], F_VN],
            partial(initialise_input, bits=bits, matrix=matrix, transfer=transfer, primaries=primaries)
        )

    @wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert func

        i, j = 0, 0
        args_l, kwargs_l = list(args), list(kwargs.items())
        in_kwargs = False
        while True:
            # Checks positional arguments
            try:
                obj = args_l[i]
                if isinstance(obj, vs.VideoNode):
                    clip = obj
                    break
                else:
                    i += 1
            except IndexError:
                # Check keyword arguments
                in_kwargs = True
                try:
                    name, obj = kwargs_l[i]
                    if isinstance(obj, vs.VideoNode):
                        clip = obj
                        break
                    else:
                        j += 1
                except IndexError:
                    # Check default arguments
                    signature = inspect.signature(func)
                    default_args = {
                        k: v.default
                        for k, v in signature.parameters.items()
                        if v.default is not inspect.Parameter.empty and isinstance(v.default, vs.VideoNode)
                    }
                    if default_args:
                        name, clip = list(default_args.items())[0]
                    else:
                        raise ValueError(
                            'initialise_input: None VideoNode found in positional, keyword nor default arguments!'
                        )
                    break

        if bits:
            clip = depth(clip, bits)
        for prop, val in zip(('_Matrix', '_Transfer', '_Primaries'), (matrix, transfer, primaries)):
            clip = clip.std.SetFrameProp(prop, intval=val)

        if in_kwargs:
            kwargs[name] = clip  # type: ignore
        else:
            args_l[i] = clip

        return func(*args_l, **kwargs)

    return cast(F_VN, _wrapper)


def get_colour_range(clip: vs.VideoNode) -> PropsVal.ColorRange:
    """Get the colour range from the VideoProps"""
    return PropsVal.ColorRange[
        {crange.value: crange.name for crange in PropsVal.ColorRange}[cast(int, clip.get_frame(0).props['_ColorRange'])]
    ]


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



def normalise_ranges(clip: vs.VideoNode, ranges: Union[Range, List[Range], Trim, List[Trim]],
                     *, norm_dups: bool = False) -> List[Tuple[int, int]]:
    """Modified version of lvsfunc.util.normalize_ranges following python slicing syntax"""
    ranges = ranges if isinstance(ranges, list) else [ranges]

    nranges: Set[Tuple[int, int]] = set()
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

        if start >= clip.num_frames or end > clip.num_frames:
            warnings.warn(f'normalise_ranges: {r} out of range')
        else:
            start, end = min(start, clip.num_frames - 1), min(end, clip.num_frames)
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
            start, end = normalise_ranges(clip, trim_or_df).pop()
            clips.append(clip[start:end])
        else:
            df = trim_or_df
            clips.append(clip[df] * df.dup)
    return core.std.Splice(clips)


def pick_px_op(
    use_expr: bool,
    operations: Tuple[str, Union[Sequence[int], Sequence[float], int, float, F]]
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
