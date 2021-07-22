from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

from vapoursynth import VideoNode

Range = Union[int, Tuple[Optional[int], Optional[int]]]
Trim = Tuple[Optional[int], Optional[int]]
# Some outputs
Output = Union[
    VideoNode,
    List[VideoNode],
    Tuple[int, VideoNode],
    Tuple[int, List[VideoNode]]
]
# Operator Input
OpInput = Union[
    VideoNode,
    List[VideoNode],
    Tuple[VideoNode, ...],
    Tuple[List[VideoNode], ...],
    Dict[str, VideoNode],
    Dict[str, List[VideoNode]]
]
# VapourSynth Function
VSF = TypeVar('VSF', bound=Callable[..., OpInput])


class DuplicateFrame(int):
    """Class depicting a duplicate frame"""
    dup: int

    def __new__(cls, x: int, /, dup: int = 1) -> DuplicateFrame:
        df = super().__new__(cls, x)
        df.dup = dup
        return df

    def __repr__(self) -> str:
        return f'<DuplicateFrame object: \'x:{super().__repr__()}, dup:{self.dup}\'>'

    def __str__(self) -> str:
        return f'{super().__str__()} * {self.dup}'

    def __add__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup + x)

    def __sub__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup - x)

    def __mul__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup * x)

    def __floordiv__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup // x)
