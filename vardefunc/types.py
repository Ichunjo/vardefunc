from __future__ import annotations

from enum import Enum, IntEnum
from os import PathLike
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Tuple, TypeVar, Union, cast)

from numpy import array as np_array
from numpy import int8, int16, int32, uint8, uint16, uint32
from numpy.lib.index_tricks import CClass as NP_CClass
from numpy.typing import NDArray
from vapoursynth import VideoFormat, VideoNode
from vsutil import Dither

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
# Function Debug
F_OpInput = TypeVar('F_OpInput', bound=Callable[..., OpInput])
# Function finalise
F_VN = TypeVar('F_VN', bound=Callable[..., VideoNode])
# Generic function
F = TypeVar('F', bound=Callable[..., Any])
# Any Numpy integrer
AnyInt = Union[int8, int16, int32, uint8, uint16, uint32]


AnyPath = Union[PathLike[str], str]

MATRIX = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
TRANSFER = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
PRIMARIES = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22]
PIXEL_RANGE = Literal[0, 1]


class Zimg:
    class PixelRange(IntEnum):
        """Pixel range (ITU-T H.265 Eq E-4 to E-15)"""
        LIMITED = 0
        FULL = 1

    DitherType = Dither

    class ResampleFilterUV(str, Enum):
        """Scaling method for UV channels to be used in core.resize.XXXXX(..., resample_filter_uv=)"""
        POINT = 'point'
        BILINEAR = 'bilinear'
        BICUBIC = 'bicubic'
        SPLINE16 = 'spline16'
        SPLINE36 = 'spline36'
        SPLINE64 = 'spline64'
        LANCZOS = 'lanczos'


class VNumpy:
    class _CClass(NP_CClass):
        def __getitem__(self, key: Union[NDArray[AnyInt], Tuple[NDArray[AnyInt], ...], slice]) -> NDArray[AnyInt]:
            return cast(NDArray[AnyInt], super().__getitem__(key))

    @staticmethod
    def array(obj: Union[NDArray[AnyInt], Sequence[Any]], **kwargs: Any) -> NDArray[AnyInt]:
        return np_array(obj, **kwargs)

    @classmethod
    def zip_arrays(cls, *arrays: NDArray[AnyInt]) -> NDArray[AnyInt]:
        return cls._CClass()[arrays]


class VideoNode_F(VideoNode):
    """VideoNode object without a None Format"""
    format: VideoFormat


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


class FormatError(Exception):
    """Raised when a format of VideoNode object is not allowed."""


def format_not_none(clip: VideoNode, /) -> VideoNode_F:
    if clip.format is None:
        raise FormatError('Variable format not allowed!')
    return cast(VideoNode_F, clip)
