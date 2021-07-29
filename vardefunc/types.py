from __future__ import annotations

from enum import IntEnum
from typing import (Any, Callable, Dict, List, Literal, NoReturn, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

from vapoursynth import Format, VideoNode

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

MATRIX = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
TRANSFER = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
PRIMARIES = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22]
PIXEL_RANGE = Literal[0, 1]


class Zimg:
    class Matrix(IntEnum):
        """Matrix coefficients (ITU-T H.265 Table E.5)"""
        RGB = 0
        GBR = 0
        BT709 = 1
        UNKNOWN = 2
        _RESERVED = 3
        FCC = 4
        BT470BG = 5
        SMPTE170M = 6
        SMPTE240M = 7
        YCGCO = 8
        BT2020NC = 9
        BT2020C = 10
        SMPTE2085 = 11
        CHROMA_DERIVED_NC = 12
        CHROMA_DERIVED_C = 13
        ICTCP = 14

        @property
        def RESERVED(self) -> NoReturn:
            raise PermissionError

    class Transfer(IntEnum):
        """Transfer characteristics (ITU-T H.265 Table E.4)"""
        BT709 = 1
        UNKNOWN = 2
        _RESERVED = 3
        BT470M = 4
        BT470BG = 5
        SMPTE170M = 6
        SMPTE240M = 7
        LINEAR = 8
        LOG100 = 9
        LOG316 = 10
        IEC61966_2_4 = 11
        BT1361E = 12
        IEC61966_2_1 = 13
        BT2020_10 = 14
        BT2020_12 = 15
        SMPTE2084 = 16
        SMPTE428 = 17
        ARIB_STD_B67 = 18

        @property
        def RESERVED(self) -> NoReturn:
            raise PermissionError

    class Primaries(IntEnum):
        """Color primaries (ITU-T H.265 Table E.3)"""
        BT709 = 1
        UNKNOWN = 2
        _RESERVED = 3
        BT470M = 4
        BT470BG = 5
        SMPTE170M = 6
        SMPTE240M = 7
        FILM = 8
        BT2020 = 9
        SMPTE428 = 10
        XYZ = 10
        SMPTE431 = 11
        SMPTE432 = 12
        JEDEC_P22 = 22

        @property
        def RESERVED(self) -> NoReturn:
            raise PermissionError

    class PixelRange(IntEnum):
        """Pixel range (ITU-T H.265 Eq E-4 to E-15)"""
        LIMITED = 0
        FULL = 1


class PropsVal:
    Matrix = Zimg.Matrix
    Transfer = Zimg.Transfer
    Primaries = Zimg.Primaries

    class ChromaLocation(IntEnum):
        """Chroma sample position in YUV formats"""
        LEFT = 0
        CENTER = 1
        TOP_LEFT = 2
        TOP = 3
        BOTTOM_LEFT = 4
        BOTTOM = 5

    class ColorRange(IntEnum):
        """Full or limited range (PC/TV range). Primarily used with YUV formats."""
        FULL = 0
        LIMITED = 1

    class FieldBased(IntEnum):
        """If the frame is composed of two independent fields (interlaced)."""
        FRAME_BASED = 0
        PROGRESSIVE = 0
        BOTTOM_FIELD_FIRST = 1
        BFF = 1
        TOP_FIELD_FIRST = 2
        TFF = 2



class VideoNode_F(VideoNode):
    """VideoNode object without a None Format"""
    format: Format


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
