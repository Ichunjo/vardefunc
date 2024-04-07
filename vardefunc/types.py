from __future__ import annotations

__all__ = ['DuplicateFrame']

from fractions import Fraction
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast

from numpy import array as np_array
from numpy import int8, int16, int32, uint8, uint16, uint32
from numpy.lib.index_tricks import CClass as NP_CClass
from numpy.typing import NDArray
from pytimeconv import Convert
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
# Function Debug
F_OpInput = TypeVar('F_OpInput', bound=Callable[..., OpInput])
# Any Numpy integrer
AnyInt = Union[int8, int16, int32, uint8, uint16, uint32]

AnyPath = Union[PathLike[str], str]


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


class DuplicateFrame(int):
    """Class depicting a duplicate frame"""
    dup: int

    def __new__(cls, x: int, /, dup: int = 1) -> DuplicateFrame:
        df = super().__new__(cls, x)
        df.dup = dup
        return df

    def to_samples(self, ref_fps: Fraction, sample_rate: int) -> DuplicateFrame:
        return DuplicateFrame(Convert.f2samples(int(self), ref_fps, sample_rate), dup=self.dup)

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
