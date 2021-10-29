from __future__ import annotations

from enum import Enum, IntEnum
from fractions import Fraction
from os import PathLike
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Tuple, TypeVar, Union, cast)

from numpy import array as np_array
from numpy import int8, int16, int32, uint8, uint16, uint32
from numpy.lib.index_tricks import CClass as NP_CClass
from numpy.typing import NDArray
from pytimeconv import Convert
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
    class PresetFormatEx(IntEnum):
        GRAY11 = 269156352
        GRAY13 = 269287424
        GRAY15 = 269418496
        GRAY17 = 269549568
        GRAY18 = 269615104
        GRAY19 = 269680640
        GRAY20 = 269746176
        GRAY21 = 269811712
        GRAY22 = 269877248
        GRAY23 = 269942784
        GRAY24 = 270008320
        GRAY25 = 270073856
        GRAY26 = 270139392
        GRAY27 = 270204928
        GRAY28 = 270270464
        GRAY29 = 270336000
        GRAY30 = 270401536
        GRAY31 = 270467072

        YUV444P11 = 806027264
        YUV444P13 = 806158336
        YUV444P15 = 806289408
        YUV444P17 = 806420480
        YUV444P18 = 806486016
        YUV444P19 = 806551552
        YUV444P20 = 806617088
        YUV444P21 = 806682624
        YUV444P22 = 806748160
        YUV444P23 = 806813696
        YUV444P24 = 806879232
        YUV444P25 = 806944768
        YUV444P26 = 807010304
        YUV444P27 = 807075840
        YUV444P28 = 807141376
        YUV444P29 = 807206912
        YUV444P30 = 807272448
        YUV444P31 = 807337984
        YUV444P32 = 807403520

        YUV422P11 = 806027520
        YUV422P13 = 806158592
        YUV422P15 = 806289664
        YUV422P17 = 806420736
        YUV422P18 = 806486272
        YUV422P19 = 806551808
        YUV422P20 = 806617344
        YUV422P21 = 806682880
        YUV422P22 = 806748416
        YUV422P23 = 806813952
        YUV422P24 = 806879488
        YUV422P25 = 806945024
        YUV422P26 = 807010560
        YUV422P27 = 807076096
        YUV422P28 = 807141632
        YUV422P29 = 807207168
        YUV422P30 = 807272704
        YUV422P31 = 807338240
        YUV422P32 = 807403776

        YUV411P9 = 805896704
        YUV411P10 = 805962240
        YUV411P11 = 806027776
        YUV411P12 = 806093312
        YUV411P13 = 806158848
        YUV411P14 = 806224384
        YUV411P15 = 806289920
        YUV411P16 = 806355456
        YUV411P17 = 806420992
        YUV411P18 = 806486528
        YUV411P19 = 806552064
        YUV411P20 = 806617600
        YUV411P21 = 806683136
        YUV411P22 = 806748672
        YUV411P23 = 806814208
        YUV411P24 = 806879744
        YUV411P25 = 806945280
        YUV411P26 = 807010816
        YUV411P27 = 807076352
        YUV411P28 = 807141888
        YUV411P29 = 807207424
        YUV411P30 = 807272960
        YUV411P31 = 807338496
        YUV411P32 = 807404032

        YUV440P9 = 805896193
        YUV440P10 = 805961729
        YUV440P11 = 806027265
        YUV440P12 = 806092801
        YUV440P13 = 806158337
        YUV440P14 = 806223873
        YUV440P15 = 806289409
        YUV440P16 = 806354945
        YUV440P17 = 806420481
        YUV440P18 = 806486017
        YUV440P19 = 806551553
        YUV440P20 = 806617089
        YUV440P21 = 806682625
        YUV440P22 = 806748161
        YUV440P23 = 806813697
        YUV440P24 = 806879233
        YUV440P25 = 806944769
        YUV440P26 = 807010305
        YUV440P27 = 807075841
        YUV440P28 = 807141377
        YUV440P29 = 807206913
        YUV440P30 = 807272449
        YUV440P31 = 807337985
        YUV440P32 = 807403521

        YUV410P9 = 805896706
        YUV410P10 = 805962242
        YUV410P11 = 806027778
        YUV410P12 = 806093314
        YUV410P13 = 806158850
        YUV410P14 = 806224386
        YUV410P15 = 806289922
        YUV410P16 = 806355458
        YUV410P17 = 806420994
        YUV410P18 = 806486530
        YUV410P19 = 806552066
        YUV410P20 = 806617602
        YUV410P21 = 806683138
        YUV410P22 = 806748674
        YUV410P23 = 806814210
        YUV410P24 = 806879746
        YUV410P25 = 806945282
        YUV410P26 = 807010818
        YUV410P27 = 807076354
        YUV410P28 = 807141890
        YUV410P29 = 807207426
        YUV410P30 = 807272962
        YUV410P31 = 807338498
        YUV410P32 = 807404034

        RGB33 = 537591808
        RGB39 = 537722880
        RGB45 = 537853952
        RGB51 = 537985024
        RGB54 = 538050560
        RGB57 = 538116096
        RGB60 = 538181632
        RGB63 = 538247168
        RGB66 = 538312704
        RGB69 = 538378240
        RGB72 = 538443776
        RGB75 = 538509312
        RGB78 = 538574848
        RGB81 = 538640384
        RGB84 = 538705920
        RGB87 = 538771456
        RGB90 = 538836992
        RGB93 = 538902528
        RGB96 = 538968064

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


class FormatError(Exception):
    """Raised when a format of VideoNode object is not allowed."""


def format_not_none(clip: VideoNode, /) -> VideoNode_F:
    if clip.format is None:
        raise FormatError('Variable format not allowed!')
    return cast(VideoNode_F, clip)
