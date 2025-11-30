"""Helper functions for the main functions in this module"""

from __future__ import annotations

from contextlib import suppress
from fractions import Fraction
from functools import cached_property, partial
from itertools import groupby, pairwise
from math import inf
from types import NoneType
from typing import Any, Callable, Iterable, MutableSequence, Optional, Self, Sequence, TypeGuard, cast, overload

import numpy as np
from pytimeconv import Convert
from vstools import ClipsCache, FrameRangeN, FrameRangesN, core, vs
from vstools.functions.ranges import _RangesCallBack

from .types import AnyInt, DuplicateFrame, Range, Trim, VNumpy

__all__ = [
    "MutableVideoNode",
    "adjust_audio_frames",
    "adjust_clip_frames",
    "normalise_ranges",
    "ranges_to_indices",
    "select_frames",
    "to_incl_excl",
    "to_incl_incl",
]


def select_frames(
    clips: vs.VideoNode | Sequence[vs.VideoNode],
    indices: np.typing.NDArray[AnyInt] | Sequence[int] | Sequence[tuple[int, int]],
    *,
    mismatch: bool = False,
) -> vs.VideoNode:
    """
    Select frames from one or more clips at specified indices.

    Passing one clip will perform as frame remap just like vstools.remap_frames.
    Passing two or more clips will perform as a mix of remap and replace_ranges function.

    Original idea from EoE.

    :param clips:       A clip or a sequence of clips to select the frames from
    :param indices:     Indices of frames to select.
                        Provide a sequence of indices for a single clip, or for multiple clips,
                        a sequence of tuples in the form ``(clip_index, frame_index)``
    :param mismatch:    Splicing clips with different formats or dimensions is considered an error
                        unless mismatch is true. Defaults to False.
    :return:            The selected frames in a single clip.
    """
    clips = clips if isinstance(clips, Sequence) else [clips]
    indices = VNumpy.array(indices) if isinstance(indices, Sequence) else indices

    if indices.ndim == 1:
        indices = VNumpy.zip_arrays(np.zeros(len(indices), np.uint32), indices)
    elif indices.ndim == 2:
        pass
    else:
        raise ValueError("select_frames: only 1D and 2D array is allowed!")

    base = (
        clips[0].std.BlankClip(length=len(indices))
        if not mismatch
        else clips[0].std.BlankClip(length=len(indices), varsize=True, varformat=True)
    )

    def _select_func(n: int, clips: Sequence[vs.VideoNode], indices: np.typing.NDArray[AnyInt]) -> vs.VideoNode:
        # index: NDArray[AnyInt] = indices[n]  # Get the index / num_frame pair
        # i_clip = int(index[0])  # Get the index
        # num = int(index[1])  # Get the num_frame
        # nclip = clips[i_clip]  # Select the clip to be returned
        # tclip = nclip[num]  # Slice the clip
        # return tclip
        return clips[int(indices[n][0])][int(indices[n][1])]

    return core.std.FrameEval(base, partial(_select_func, clips=clips, indices=indices))


@overload
def normalise_ranges(
    clip: vs.VideoNode, ranges: FrameRangeN | FrameRangesN | _RangesCallBack, *, norm_dups: bool = True
) -> list[Range]: ...


@overload
def normalise_ranges(
    clip: vs.AudioNode,
    ranges: FrameRangeN | FrameRangesN | _RangesCallBack,
    *,
    norm_dups: bool = True,
    ref_fps: Fraction | None = None,
) -> list[Range]: ...


@overload
def normalise_ranges(
    clip: None,
    ranges: FrameRangeN | FrameRangesN,
    *,
    norm_dups: bool = True,
) -> list[tuple[int, int | None]]: ...


def normalise_ranges(
    clip: vs.VideoNode | vs.AudioNode | None,
    ranges: FrameRangeN | FrameRangesN | _RangesCallBack,
    *,
    norm_dups: bool = True,
    ref_fps: Fraction | None = None,
) -> list[Range] | list[tuple[int, int | None]]:
    """
    Normalise ranges to a list of positive ranges following python slicing syntax `(inclusive, exclusive)`

    :param clip:        Input clip.
    :param ranges:      Frame range list of frame ranges, or range callbacks.
    :param norm_dups:   Normalise duplicated, defaults to True
    :param ref_fps:     FPS reference when passing an AudioNode, defaults to None
    :return:            Normalised ranges
    """
    if isinstance(clip, vs.VideoNode):
        num_frames = clip.num_frames
    elif isinstance(clip, vs.AudioNode):
        num_frames = clip.num_samples if ref_fps is not None else clip.num_frames
    else:
        num_frames = None

    if ranges is None:
        return [(0, num_frames)]

    def _resolve_ranges_type(
        rngs: int | tuple[int | None, int | None] | FrameRangesN | _RangesCallBack,
    ) -> Sequence[int | tuple[int | None, int | None] | None]:
        if isinstance(rngs, int):
            return [rngs]

        if isinstance(rngs, tuple) and len(rngs) == 2:
            if isinstance(rngs[0], int) or (rngs[0] is None and isinstance(rngs[1], int)) or rngs[1] is None:
                return [cast(tuple[int | None, int | None], rngs)]
            else:
                raise ValueError

        if callable(rngs):
            if not num_frames:
                raise ValueError

            cb_rngs = list[tuple[int, int]]()
            r = 0

            for i, j in groupby(rngs(n) for n in range(num_frames)):
                step = len(list(j))
                if i:
                    cb_rngs.append((r, r + step))
                r += step
            return cb_rngs

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
            import warnings

            if start > end:
                warnings.warn(f'normalise_ranges: start frame "{start}" is higher than end frame "{end}"')

            if num_frames is not None and (start >= num_frames or end > num_frames):
                warnings.warn(f"normalise_ranges: {r} out of range")

        if num_frames is not None:
            start = min(start, num_frames - 1)
            if end is not None:
                end = min(end, num_frames)

        nranges.add((start, end))

    out = sorted(nranges)

    if norm_dups:
        nranges_d = dict(out)
        nranges_rev = sorted(nranges_d.items(), reverse=True)

        for (start1, end1), (start2, end2) in pairwise(nranges_rev):
            if end1 is None or end2 is None:
                continue

            if start2 < start1 <= end2 < end1:
                nranges_d[start2] = max(end1, nranges_d[start1], key=lambda x: x if x is not None else inf)
                del nranges_d[start1]

            if start2 < start1 and end1 <= end2:
                del nranges_d[start1]

        out = list(nranges_d.items())

    return out


def to_incl_incl(ranges: list[Range]) -> list[Range]:
    return [(s, e - 1) for (s, e) in ranges]


def to_incl_excl(ranges: list[Range]) -> list[Range]:
    return [(s, e + 1) for (s, e) in ranges]


class _ranges_to_indices:  # noqa: N801
    def __call__(
        self,
        ref: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN | _RangesCallBack,
        ref_indices: tuple[int, int] = (0, 1),
    ) -> np.typing.NDArray[AnyInt]:
        return VNumpy.zip_arrays(
            np.fromiter(self.gen_indices(ref, ranges, ref_indices), np.uint32, ref.num_frames),
            np.arange(ref.num_frames, dtype=np.uint32),
        )

    def gen_indices(
        self, ref: vs.VideoNode, ranges: FrameRangeN | FrameRangesN | _RangesCallBack, ref_indices: tuple[int, int]
    ) -> Iterable[int]:
        nranges = normalise_ranges(ref, ranges)

        for f in range(ref.num_frames):
            i = ref_indices[0]
            for start, end in nranges:
                if start <= f < end:
                    i = ref_indices[1]
                    break
            yield i


ranges_to_indices = _ranges_to_indices()


def adjust_clip_frames(clip: vs.VideoNode, trims_or_dfs: list[Trim | DuplicateFrame] | Trim) -> vs.VideoNode:
    """Trims and/or duplicates frames"""
    trims_or_dfs = [trims_or_dfs] if isinstance(trims_or_dfs, tuple) else trims_or_dfs
    indices: list[int] = []
    for trim_or_df in trims_or_dfs:
        if isinstance(trim_or_df, tuple):
            ntrim = normalise_ranges(clip, trim_or_df).pop()
            indices.extend(range(*ntrim))
        else:
            df = trim_or_df
            indices.extend([df.numerator] * df.dup)
    return select_frames(clip, indices)


def adjust_audio_frames(
    audio: vs.AudioNode, trims_or_dfs: list[Trim | DuplicateFrame] | Trim, *, ref_fps: Optional[Fraction] = None
) -> vs.AudioNode:
    audios: list[vs.AudioNode] = []
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


def pick_px_op(
    use_expr: bool, operations: tuple[str, Sequence[int] | Sequence[float] | int | float | Callable[..., Any]]
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
                raise ValueError("pick_px_operation: operations[1] is not a valid type!")
        elif isinstance(lut, int):
            func = partial(core.std.Lut, lut=lut)
        elif isinstance(lut, float):
            func = partial(core.std.Lut, lutf=lut)
        else:
            raise ValueError("pick_px_operation: operations[1] is not a valid type!")
    return func


class MutableVideoNode(MutableSequence[vs.VideoNode]):
    def __init__(self, node: vs.VideoNode | Sequence[tuple[int, vs.VideoNode]]) -> None:
        if isinstance(node, vs.VideoNode):
            self._mutable_node: list[vs.VideoNode | tuple[int, None | vs.VideoNode]] = [
                (x, node) for x in range(node.num_frames)
            ]
        else:
            self._mutable_node = list(node)

    @overload
    def __getitem__(self, index: int) -> vs.VideoNode: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> vs.VideoNode | Self:
        self._normalize_inner_list()

        if isinstance(index, int):
            value = cast(tuple[int, vs.VideoNode], self._mutable_node[index])

            return value[1][value[0]]

        values = cast(list[tuple[int, vs.VideoNode]], self._mutable_node[index])

        return self.__class__(values)

    @overload
    def __setitem__(self, index: int, value: vs.VideoNode | tuple[int, None | vs.VideoNode]) -> None: ...

    @overload
    def __setitem__(
        self,
        index: slice,
        value: vs.VideoNode
        | tuple[int, None | vs.VideoNode]
        | Iterable[vs.VideoNode]
        | Iterable[tuple[int, None | vs.VideoNode]],
    ) -> None: ...

    def __setitem__(
        self,
        index: int | slice,
        value: vs.VideoNode
        | tuple[int, None | vs.VideoNode]
        | Iterable[vs.VideoNode]
        | Iterable[tuple[int, None | vs.VideoNode]],
    ) -> None:
        self._normalize_inner_list()

        def _no_iterable(value: Any) -> TypeGuard[vs.VideoNode | tuple[int, None | vs.VideoNode]]:
            return (
                isinstance(value, tuple)
                and isinstance(value[0], int)
                and isinstance(value[1], (NoneType, vs.VideoNode))
            ) or isinstance(value, vs.VideoNode)

        if isinstance(index, int):
            self._mutable_node[index] = cast(vs.VideoNode | tuple[int, None | vs.VideoNode], value)
        elif isinstance(index, slice):
            if _no_iterable(value):
                self._mutable_node[index] = [value]
            else:
                self._mutable_node[index] = cast(Iterable[Any], value)

    def __delitem__(self, index: int | slice) -> None:
        self._normalize_inner_list()
        del self._mutable_node[index]

    def __len__(self) -> int:
        self._normalize_inner_list()
        return len(self.indices)

    def insert(self, index: int, value: vs.VideoNode | tuple[int, vs.VideoNode | None]) -> None:
        self._normalize_inner_list()
        self._mutable_node.insert(index, value)

    def _normalize_inner_list(self) -> None:
        with suppress(AttributeError):
            del self.all_nodes
        with suppress(AttributeError):
            del self.indices
        self._mutable_node = [(f_i, self.all_nodes[c_i]) for c_i, f_i in self.indices]

    @cached_property
    def all_nodes(self) -> list[vs.VideoNode]:
        all_nodes = ClipsCache()

        for n in self._mutable_node:
            if isinstance(n, vs.VideoNode):
                all_nodes[n] = n
            else:
                if isinstance(nn := n[1], vs.VideoNode):
                    all_nodes[nn] = nn
        try:
            return list(all_nodes.keys())
        finally:
            del all_nodes

    @cached_property
    def indices(self) -> list[tuple[int, int]]:
        indices = list[tuple[int, int]]()

        for n in self._mutable_node:
            if isinstance(n, vs.VideoNode):
                indices.extend((self.all_nodes.index(n), i) for i in range(n.num_frames))
            else:
                indices.append((0 if n[1] is None else self.all_nodes.index(n[1]), n[0]))

        return indices

    def to_node(self) -> vs.VideoNode:
        self._normalize_inner_list()
        return select_frames(self.all_nodes, self.indices)
