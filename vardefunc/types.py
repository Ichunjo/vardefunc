from __future__ import annotations

from fractions import Fraction
from os import PathLike
from typing import Any, Self, Sequence

from jetpytools import StrictRange
from numpy import array as np_array
from numpy import c_, int8, int16, int32, uint8, uint16, uint32
from numpy.typing import NDArray
from pytimeconv import Convert
from vapoursynth import VideoNode

type Range = StrictRange
type RangeN = tuple[int | None, int | None]
type Trim = RangeN

__all__ = ["DuplicateFrame"]

# Some outputs
Output = VideoNode | list[VideoNode] | tuple[int, VideoNode] | tuple[int, list[VideoNode]]
# Operator Input
OpInput = (
    VideoNode
    | list[VideoNode]
    | tuple[VideoNode, ...]
    | tuple[list[VideoNode], ...]
    | dict[str, VideoNode]
    | dict[str, list[VideoNode]]
)

# Any Numpy integrer
AnyInt = int8 | int16 | int32 | uint8 | uint16 | uint32

AnyPath = PathLike[str] | str


class VNumpy:
    @staticmethod
    def array(obj: NDArray[AnyInt] | Sequence[Any], **kwargs: Any) -> NDArray[AnyInt]:
        return np_array(obj, **kwargs)

    @classmethod
    def zip_arrays(cls, *arrays: NDArray[AnyInt]) -> NDArray[AnyInt]:
        return c_[*arrays]


class DuplicateFrame(int):
    """Class depicting a duplicate frame"""

    dup: int

    def __new__(cls, x: int, /, dup: int = 1) -> Self:
        df = super().__new__(cls, x)
        df.dup = dup
        return df

    def to_samples(self, ref_fps: Fraction, sample_rate: int) -> DuplicateFrame:
        return DuplicateFrame(Convert.f2samples(int(self), ref_fps, sample_rate), dup=self.dup)

    def __repr__(self) -> str:
        return f"<DuplicateFrame object: 'x:{super().__repr__()}, dup:{self.dup}'>"

    def __str__(self) -> str:
        return f"{super().__str__()} * {self.dup}"

    def __add__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup + x)

    def __sub__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup - x)

    def __mul__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup * x)

    def __floordiv__(self, x: int) -> DuplicateFrame:
        return DuplicateFrame(self, dup=self.dup // x)
