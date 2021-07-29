# Stop pep8 from complaining (hopefully)
# NOQA

# Ignore Flake Warnings
# flake8: noqa

# Ignore coverage
# (No coverage)

# From https://gist.github.com/pylover/7870c235867cf22817ac5b096defb768
# noinspection PyPep8
# noinspection PyPep8Naming
# noinspection PyTypeChecker
# noinspection PyAbstractClass
# noinspection PyArgumentEqualDefault
# noinspection PyArgumentList
# noinspection PyAssignmentToLoopOrWithParameter
# noinspection PyAttributeOutsideInit
# noinspection PyAugmentAssignment
# noinspection PyBroadException
# noinspection PyByteLiteral
# noinspection PyCallByClass
# noinspection PyChainedComparsons
# noinspection PyClassHasNoInit
# noinspection PyClassicStyleClass
# noinspection PyComparisonWithNone
# noinspection PyCompatibility
# noinspection PyDecorator
# noinspection PyDefaultArgument
# noinspection PyDictCreation
# noinspection PyDictDuplicateKeys
# noinspection PyDocstringTypes
# noinspection PyExceptClausesOrder
# noinspection PyExceptionInheritance
# noinspection PyFromFutureImport
# noinspection PyGlobalUndefined
# noinspection PyIncorrectDocstring
# noinspection PyInitNewSignature
# noinspection PyInterpreter
# noinspection PyListCreation
# noinspection PyMandatoryEncoding
# noinspection PyMethodFirstArgAssignment
# noinspection PyMethodMayBeStatic
# noinspection PyMethodOverriding
# noinspection PyMethodParameters
# noinspection PyMissingConstructor
# noinspection PyMissingOrEmptyDocstring
# noinspection PyNestedDecorators
# noinspection PynonAsciiChar
# noinspection PyNoneFunctionAssignment
# noinspection PyOldStyleClasses
# noinspection PyPackageRequirements
# noinspection PyPropertyAccess
# noinspection PyPropertyDefinition
# noinspection PyProtectedMember
# noinspection PyRaisingNewStyleClass
# noinspection PyRedeclaration
# noinspection PyRedundantParentheses
# noinspection PySetFunctionToLiteral
# noinspection PySimplifyBooleanCheck
# noinspection PySingleQuotedDocstring
# noinspection PyStatementEffect
# noinspection PyStringException
# noinspection PyStringFormat
# noinspection PySuperArguments
# noinspection PyTrailingSemicolon
# noinspection PyTupleAssignmentBalance
# noinspection PyTupleItemAssignment
# noinspection PyUnboundLocalVariable
# noinspection PyUnnecessaryBackslash
# noinspection PyUnreachableCode
# noinspection PyUnresolvedReferences
# noinspection PyUnusedLocal
# noinspection ReturnValueFromInit

import ctypes
import fractions
import types
import typing

T = typing.TypeVar("T")
SingleAndSequence = typing.Union[T, typing.Sequence[T]]

###
# ENUMS AND CONSTANTS
class ColorFamily(int):
    name: str
    value: int

    GRAY: typing.ClassVar['ColorFamily']
    RGB: typing.ClassVar['ColorFamily']
    YUV: typing.ClassVar['ColorFamily']
    YCOCG: typing.ClassVar['ColorFamily']
    COMPAT: typing.ClassVar['ColorFamily']

GRAY: ColorFamily
RGB: ColorFamily
YUV: ColorFamily
YCOCG: ColorFamily
COMPAT: ColorFamily


class SampleType(int):
    name: str
    value: int

    INTEGER: typing.ClassVar['SampleType']
    FLOAT: typing.ClassVar['SampleType']


INTEGER: SampleType
FLOAT: SampleType


class PresetFormat(int):
    name: str
    value: int

    NONE: typing.ClassVar['PresetFormat']

    GRAY8: typing.ClassVar['PresetFormat']
    GRAY16: typing.ClassVar['PresetFormat']

    GRAYH: typing.ClassVar['PresetFormat']
    GRAYS: typing.ClassVar['PresetFormat']

    YUV420P8: typing.ClassVar['PresetFormat']
    YUV422P8: typing.ClassVar['PresetFormat']
    YUV444P8: typing.ClassVar['PresetFormat']
    YUV410P8: typing.ClassVar['PresetFormat']
    YUV411P8: typing.ClassVar['PresetFormat']
    YUV440P8: typing.ClassVar['PresetFormat']

    YUV420P9: typing.ClassVar['PresetFormat']
    YUV422P9: typing.ClassVar['PresetFormat']
    YUV444P9: typing.ClassVar['PresetFormat']

    YUV420P10: typing.ClassVar['PresetFormat']
    YUV422P10: typing.ClassVar['PresetFormat']
    YUV444P10: typing.ClassVar['PresetFormat']

    YUV420P12: typing.ClassVar['PresetFormat']
    YUV422P12: typing.ClassVar['PresetFormat']
    YUV444P12: typing.ClassVar['PresetFormat']

    YUV420P14: typing.ClassVar['PresetFormat']
    YUV422P14: typing.ClassVar['PresetFormat']
    YUV444P14: typing.ClassVar['PresetFormat']

    YUV420P16: typing.ClassVar['PresetFormat']
    YUV422P16: typing.ClassVar['PresetFormat']
    YUV444P16: typing.ClassVar['PresetFormat']

    YUV444PH: typing.ClassVar['PresetFormat']
    YUV444PS: typing.ClassVar['PresetFormat']

    RGB24: typing.ClassVar['PresetFormat']
    RGB27: typing.ClassVar['PresetFormat']
    RGB30: typing.ClassVar['PresetFormat']
    RGB48: typing.ClassVar['PresetFormat']

    RGBH: typing.ClassVar['PresetFormat']
    RGBS: typing.ClassVar['PresetFormat']

    COMPATBGR32: typing.ClassVar['PresetFormat']
    COMPATYUY2: typing.ClassVar['PresetFormat']


NONE: PresetFormat

GRAY8: PresetFormat
GRAY16: PresetFormat

GRAYH: PresetFormat
GRAYS: PresetFormat

YUV420P8: PresetFormat
YUV422P8: PresetFormat
YUV444P8: PresetFormat
YUV410P8: PresetFormat
YUV411P8: PresetFormat
YUV440P8: PresetFormat

YUV420P9: PresetFormat
YUV422P9: PresetFormat
YUV444P9: PresetFormat

YUV420P10: PresetFormat
YUV422P10: PresetFormat
YUV444P10: PresetFormat

YUV420P12: PresetFormat
YUV422P12: PresetFormat
YUV444P12: PresetFormat

YUV420P14: PresetFormat
YUV422P14: PresetFormat
YUV444P14: PresetFormat

YUV420P16: PresetFormat
YUV422P16: PresetFormat
YUV444P16: PresetFormat

YUV444PH: PresetFormat
YUV444PS: PresetFormat

RGB24: PresetFormat
RGB27: PresetFormat
RGB30: PresetFormat
RGB48: PresetFormat

RGBH: PresetFormat
RGBS: PresetFormat

COMPATBGR32: PresetFormat
COMPATYUY2: PresetFormat


###
# VapourSynth Environment SubSystem

class EnvironmentData:
    """
    Contains the data VapourSynth stores for a specific environment.
    """


class Environment:
    alive: bool
    single: bool
    env_id: int
    active: bool

    def copy(self) -> Environment: ...
    def use(self) -> typing.ContextManager[None]: ...

    def __enter__(self) -> Environment: ...
    def __exit__(self, ty: typing.Type[BaseException], tv: BaseException, tb: types.TracebackType) -> None: ...

class EnvironmentPolicyAPI:
    def wrap_environment(self, environment_data: EnvironmentData) -> Environment: ...
    def create_environment(self) -> EnvironmentData: ...
    def unregister_policy(self) -> None: ...

class EnvironmentPolicy:
    def on_policy_registered(self, special_api: EnvironmentPolicyAPI) -> None: ...
    def on_policy_cleared(self) -> None: ...
    def get_current_environment(self) -> typing.Optional[EnvironmentData]: ...
    def set_environment(self, environment: typing.Optional[EnvironmentData]) -> None: ...
    def is_active(self, environment: EnvironmentData) -> bool: ...


_using_vsscript: bool


def register_policy(policy: EnvironmentPolicy) -> None: ...
def has_policy() -> bool: ...

def vpy_current_environment() -> Environment: ...
def get_current_environment() -> Environment: ...


class AlphaOutputTuple(typing.NamedTuple):
    clip: 'VideoNode'
    alpha: 'VideoNode'

Func = typing.Callable[..., typing.Any]

Function = typing.Callable[..., typing.Any]

class Error(Exception): ...

def set_message_handler(handler_func: typing.Callable[[int, str], None]) -> None: ...
def clear_output(index: int = 0) -> None: ...
def clear_outputs() -> None: ...
def get_outputs() -> typing.Mapping[int, typing.Union['VideoNode', AlphaOutputTuple]]: ...
def get_output(index: int = 0) -> typing.Union['VideoNode', AlphaOutputTuple]: ...


class Format:
    id: int
    name: str
    color_family: ColorFamily
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int

    def _as_dict(self) -> typing.Dict[str, typing.Any]: ...
    def replace(self, *,
                color_family: typing.Optional[ColorFamily] = None,
                sample_type: typing.Optional[SampleType] = None,
                bits_per_sample: typing.Optional[int] = None,
                subsampling_w: typing.Optional[int] = None,
                subsampling_h: typing.Optional[int] = None
                ) -> 'Format': ...


_VideoPropsValue = typing.Union[
    SingleAndSequence[int],
    SingleAndSequence[float],
    SingleAndSequence[str],
    SingleAndSequence['VideoNode'],
    SingleAndSequence['VideoFrame'],
    SingleAndSequence[typing.Callable[..., typing.Any]]
]

class VideoProps(typing.MutableMapping[str, _VideoPropsValue]):
    def __getattr__(self, name: str) -> _VideoPropsValue: ...
    def __setattr__(self, name: str, value: _VideoPropsValue) -> None: ...

    # mypy lo vult.
    # In all seriousness, why do I need to manually define them in a typestub?
    def __delitem__(self, name: str) -> None: ...
    def __setitem__(self, name: str, value: _VideoPropsValue) -> None: ...
    def __getitem__(self, name: str) -> _VideoPropsValue: ...
    def __iter__(self) -> typing.Iterator[str]: ...
    def __len__(self) -> int: ...

class VideoPlane:
    width: int
    height: int


class VideoFrame:
    props: VideoProps
    height: int
    width: int
    format: Format
    readonly: bool

    def copy(self) -> 'VideoFrame': ...

    def get_read_ptr(self, plane: int) -> ctypes.c_void_p: ...
    def get_read_array(self, plane: int) -> typing.Optional[memoryview]: ...
    def get_write_ptr(self, plane: int) -> ctypes.c_void_p: ...
    def get_write_array(self, plane: int) -> typing.Optional[memoryview]: ...

    def get_stride(self, plane: int) -> int: ...
    def planes(self) -> typing.Iterator['VideoPlane']: ...


class _Future(typing.Generic[T]):
    def set_result(self, value: T) -> None: ...
    def set_exception(self, exception: BaseException) -> None: ...
    def result(self) -> T: ...
    def exception(self) -> typing.Optional[typing.NoReturn]: ...


class Plugin:
    namespace: str

    def get_functions(self) -> typing.Dict[str, str]: ...
    def list_functions(self) -> str: ...


# implementation: DGMVC
class _Plugin_DGMVC_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DGMVCSource(self, base: typing.Union[str, bytes, bytearray], dependent: typing.Union[str, bytes, bytearray], view: int, frames: int, mode: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_DGMVC_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DGMVCSource(self, dependent: typing.Union[str, bytes, bytearray], view: int, frames: int, mode: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: acrop
class _Plugin_acrop_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AutoCrop(self, clip: "VideoNode", range: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, left: typing.Optional[int] = None, right: typing.Optional[int] = None, color: typing.Union[int, typing.Sequence[int], None] = None, color_second: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def CropProp(self, clip: "VideoNode") -> "VideoNode": ...
    def CropValues(self, clip: "VideoNode", range: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, left: typing.Optional[int] = None, right: typing.Optional[int] = None, color: typing.Union[int, typing.Sequence[int], None] = None, color_second: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_acrop_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AutoCrop(self, range: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, left: typing.Optional[int] = None, right: typing.Optional[int] = None, color: typing.Union[int, typing.Sequence[int], None] = None, color_second: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def CropProp(self) -> "VideoNode": ...
    def CropValues(self, range: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, left: typing.Optional[int] = None, right: typing.Optional[int] = None, color: typing.Union[int, typing.Sequence[int], None] = None, color_second: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: adg
class _Plugin_adg_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Mask(self, clip: "VideoNode", luma_scaling: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_adg_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Mask(self, luma_scaling: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: average
class _Plugin_average_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Mean(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"], None] = None, preset: typing.Optional[int] = None) -> "VideoNode": ...
    def Median(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"], None] = None) -> "VideoNode": ...


class _Plugin_average_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Mean(self, preset: typing.Optional[int] = None) -> "VideoNode": ...
    def Median(self) -> "VideoNode": ...
# end implementation


# implementation: avisource
class _Plugin_avisource_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AVIFileSource(self, path: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], pixel_type: typing.Union[str, bytes, bytearray, None] = None, fourcc: typing.Union[str, bytes, bytearray, None] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def AVISource(self, path: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], pixel_type: typing.Union[str, bytes, bytearray, None] = None, fourcc: typing.Union[str, bytes, bytearray, None] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def OpenDMLSource(self, path: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], pixel_type: typing.Union[str, bytes, bytearray, None] = None, fourcc: typing.Union[str, bytes, bytearray, None] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_avisource_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AVIFileSource(self, pixel_type: typing.Union[str, bytes, bytearray, None] = None, fourcc: typing.Union[str, bytes, bytearray, None] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def AVISource(self, pixel_type: typing.Union[str, bytes, bytearray, None] = None, fourcc: typing.Union[str, bytes, bytearray, None] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def OpenDMLSource(self, pixel_type: typing.Union[str, bytes, bytearray, None] = None, fourcc: typing.Union[str, bytes, bytearray, None] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: avs
class _Plugin_avs_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def LoadPlugin(self, path: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...


class _Plugin_avs_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def LoadPlugin(self) -> "VideoNode": ...
# end implementation


# implementation: avsr
class _Plugin_avsr_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Eval(self, lines: typing.Union[str, bytes, bytearray], bitdepth: typing.Optional[int] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def Import(self, script: typing.Union[str, bytes, bytearray], bitdepth: typing.Optional[int] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_avsr_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Eval(self, bitdepth: typing.Optional[int] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def Import(self, bitdepth: typing.Optional[int] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: avsw
class _Plugin_avsw_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Eval(self, script: typing.Union[str, bytes, bytearray], clips: typing.Union["VideoNode", typing.Sequence["VideoNode"], None] = None, clip_names: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, avisynth: typing.Union[str, bytes, bytearray, None] = None, slave: typing.Union[str, bytes, bytearray, None] = None, slave_log: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_avsw_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Eval(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"], None] = None, clip_names: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, avisynth: typing.Union[str, bytes, bytearray, None] = None, slave: typing.Union[str, bytes, bytearray, None] = None, slave_log: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: bezier
class _Plugin_bezier_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Cubic(self, clip: "VideoNode", accur: typing.Optional[float] = None, input_range: typing.Optional[int] = None, begin: typing.Optional[int] = None, end: typing.Optional[int] = None, x1: typing.Optional[int] = None, y1: typing.Optional[int] = None, x2: typing.Optional[int] = None, y2: typing.Optional[int] = None) -> "VideoNode": ...
    def Quadratic(self, clip: "VideoNode", accur: typing.Optional[float] = None, input_range: typing.Optional[int] = None, begin: typing.Optional[int] = None, end: typing.Optional[int] = None, x1: typing.Optional[int] = None, y1: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_bezier_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Cubic(self, accur: typing.Optional[float] = None, input_range: typing.Optional[int] = None, begin: typing.Optional[int] = None, end: typing.Optional[int] = None, x1: typing.Optional[int] = None, y1: typing.Optional[int] = None, x2: typing.Optional[int] = None, y2: typing.Optional[int] = None) -> "VideoNode": ...
    def Quadratic(self, accur: typing.Optional[float] = None, input_range: typing.Optional[int] = None, begin: typing.Optional[int] = None, end: typing.Optional[int] = None, x1: typing.Optional[int] = None, y1: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: bifrost
class _Plugin_bifrost_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Bifrost(self, clip: "VideoNode", altclip: typing.Optional["VideoNode"] = None, luma_thresh: typing.Optional[float] = None, variation: typing.Optional[int] = None, conservative_mask: typing.Optional[int] = None, interlaced: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None) -> "VideoNode": ...
    def BlockDiff(self, clip: "VideoNode", interlaced: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_bifrost_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Bifrost(self, altclip: typing.Optional["VideoNode"] = None, luma_thresh: typing.Optional[float] = None, variation: typing.Optional[int] = None, conservative_mask: typing.Optional[int] = None, interlaced: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None) -> "VideoNode": ...
    def BlockDiff(self, interlaced: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: bilateral
class _Plugin_bilateral_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Bilateral(self, input: "VideoNode", ref: typing.Optional["VideoNode"] = None, sigmaS: typing.Union[float, typing.Sequence[float], None] = None, sigmaR: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, algorithm: typing.Union[int, typing.Sequence[int], None] = None, PBFICnum: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Gaussian(self, input: "VideoNode", sigma: typing.Union[float, typing.Sequence[float], None] = None, sigmaV: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...


class _Plugin_bilateral_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Bilateral(self, ref: typing.Optional["VideoNode"] = None, sigmaS: typing.Union[float, typing.Sequence[float], None] = None, sigmaR: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, algorithm: typing.Union[int, typing.Sequence[int], None] = None, PBFICnum: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Gaussian(self, sigma: typing.Union[float, typing.Sequence[float], None] = None, sigmaV: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
# end implementation


# implementation: bm3d
class _Plugin_bm3d_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Basic(self, input: "VideoNode", ref: typing.Optional["VideoNode"] = None, profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, hard_thr: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
    def Final(self, input: "VideoNode", ref: "VideoNode", profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
    def OPP2RGB(self, input: "VideoNode", sample: typing.Optional[int] = None) -> "VideoNode": ...
    def RGB2OPP(self, input: "VideoNode", sample: typing.Optional[int] = None) -> "VideoNode": ...
    def VAggregate(self, input: "VideoNode", radius: typing.Optional[int] = None, sample: typing.Optional[int] = None) -> "VideoNode": ...
    def VBasic(self, input: "VideoNode", ref: typing.Optional["VideoNode"] = None, profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, radius: typing.Optional[int] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, ps_num: typing.Optional[int] = None, ps_range: typing.Optional[int] = None, ps_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, hard_thr: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
    def VFinal(self, input: "VideoNode", ref: "VideoNode", profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, radius: typing.Optional[int] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, ps_num: typing.Optional[int] = None, ps_range: typing.Optional[int] = None, ps_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_bm3d_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Basic(self, ref: typing.Optional["VideoNode"] = None, profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, hard_thr: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
    def Final(self, ref: "VideoNode", profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
    def OPP2RGB(self, sample: typing.Optional[int] = None) -> "VideoNode": ...
    def RGB2OPP(self, sample: typing.Optional[int] = None) -> "VideoNode": ...
    def VAggregate(self, radius: typing.Optional[int] = None, sample: typing.Optional[int] = None) -> "VideoNode": ...
    def VBasic(self, ref: typing.Optional["VideoNode"] = None, profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, radius: typing.Optional[int] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, ps_num: typing.Optional[int] = None, ps_range: typing.Optional[int] = None, ps_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, hard_thr: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
    def VFinal(self, ref: "VideoNode", profile: typing.Union[str, bytes, bytearray, None] = None, sigma: typing.Union[float, typing.Sequence[float], None] = None, radius: typing.Optional[int] = None, block_size: typing.Optional[int] = None, block_step: typing.Optional[int] = None, group_size: typing.Optional[int] = None, bm_range: typing.Optional[int] = None, bm_step: typing.Optional[int] = None, ps_num: typing.Optional[int] = None, ps_range: typing.Optional[int] = None, ps_step: typing.Optional[int] = None, th_mse: typing.Optional[float] = None, matrix: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: caffe
class _Plugin_caffe_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Waifu2x(self, clip: "VideoNode", noise: typing.Optional[int] = None, scale: typing.Optional[int] = None, block_w: typing.Optional[int] = None, block_h: typing.Optional[int] = None, model: typing.Optional[int] = None, cudnn: typing.Optional[int] = None, processor: typing.Optional[int] = None, tta: typing.Optional[int] = None, batch: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_caffe_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Waifu2x(self, noise: typing.Optional[int] = None, scale: typing.Optional[int] = None, block_w: typing.Optional[int] = None, block_h: typing.Optional[int] = None, model: typing.Optional[int] = None, cudnn: typing.Optional[int] = None, processor: typing.Optional[int] = None, tta: typing.Optional[int] = None, batch: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: cnr2
class _Plugin_cnr2_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Cnr2(self, clip: "VideoNode", mode: typing.Union[str, bytes, bytearray, None] = None, scdthr: typing.Optional[float] = None, ln: typing.Optional[int] = None, lm: typing.Optional[int] = None, un: typing.Optional[int] = None, um: typing.Optional[int] = None, vn: typing.Optional[int] = None, vm: typing.Optional[int] = None, scenechroma: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_cnr2_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Cnr2(self, mode: typing.Union[str, bytes, bytearray, None] = None, scdthr: typing.Optional[float] = None, ln: typing.Optional[int] = None, lm: typing.Optional[int] = None, un: typing.Optional[int] = None, um: typing.Optional[int] = None, vn: typing.Optional[int] = None, vm: typing.Optional[int] = None, scenechroma: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: colorbars
class _Plugin_colorbars_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ColorBars(self, resolution: typing.Optional[int] = None, format: typing.Optional[int] = None, hdr: typing.Optional[int] = None, wcg: typing.Optional[int] = None, compatability: typing.Optional[int] = None, subblack: typing.Optional[int] = None, superwhite: typing.Optional[int] = None, iq: typing.Optional[int] = None, filter: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_colorbars_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ColorBars(self, format: typing.Optional[int] = None, hdr: typing.Optional[int] = None, wcg: typing.Optional[int] = None, compatability: typing.Optional[int] = None, subblack: typing.Optional[int] = None, superwhite: typing.Optional[int] = None, iq: typing.Optional[int] = None, filter: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: comb
class _Plugin_comb_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def CMaskedMerge(self, base: "VideoNode", alt: "VideoNode", mask: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def CombMask(self, clip: "VideoNode", cthresh: typing.Optional[int] = None, mthresh: typing.Optional[int] = None, mi: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_comb_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def CMaskedMerge(self, alt: "VideoNode", mask: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def CombMask(self, cthresh: typing.Optional[int] = None, mthresh: typing.Optional[int] = None, mi: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: ctmf
class _Plugin_ctmf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def CTMF(self, clip: "VideoNode", radius: typing.Optional[int] = None, memsize: typing.Optional[int] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_ctmf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def CTMF(self, radius: typing.Optional[int] = None, memsize: typing.Optional[int] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: d2v
class _Plugin_d2v_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ApplyRFF(self, clip: "VideoNode", d2v: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def Source(self, input: typing.Union[str, bytes, bytearray], threads: typing.Optional[int] = None, nocrop: typing.Optional[int] = None, rff: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_d2v_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ApplyRFF(self, d2v: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def Source(self, threads: typing.Optional[int] = None, nocrop: typing.Optional[int] = None, rff: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: damb
class _Plugin_damb_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Read(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], delay: typing.Optional[float] = None) -> "VideoNode": ...
    def Write(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], format: typing.Union[str, bytes, bytearray, None] = None, sample_type: typing.Union[str, bytes, bytearray, None] = None, quality: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_damb_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Read(self, file: typing.Union[str, bytes, bytearray], delay: typing.Optional[float] = None) -> "VideoNode": ...
    def Write(self, file: typing.Union[str, bytes, bytearray], format: typing.Union[str, bytes, bytearray, None] = None, sample_type: typing.Union[str, bytes, bytearray, None] = None, quality: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: dct
class _Plugin_dct_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Filter(self, clip: "VideoNode", factors: typing.Union[float, typing.Sequence[float]]) -> "VideoNode": ...


class _Plugin_dct_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Filter(self, factors: typing.Union[float, typing.Sequence[float]]) -> "VideoNode": ...
# end implementation


# implementation: dctf
class _Plugin_dctf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DCTFilter(self, clip: "VideoNode", factors: typing.Union[float, typing.Sequence[float]], planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_dctf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DCTFilter(self, factors: typing.Union[float, typing.Sequence[float]], planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: deblock
class _Plugin_deblock_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deblock(self, clip: "VideoNode", quant: typing.Optional[int] = None, aoffset: typing.Optional[int] = None, boffset: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_deblock_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deblock(self, quant: typing.Optional[int] = None, aoffset: typing.Optional[int] = None, boffset: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: decross
class _Plugin_decross_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DeCross(self, clip: "VideoNode", thresholdy: typing.Optional[int] = None, noise: typing.Optional[int] = None, margin: typing.Optional[int] = None, debug: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_decross_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DeCross(self, thresholdy: typing.Optional[int] = None, noise: typing.Optional[int] = None, margin: typing.Optional[int] = None, debug: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: dedot
class _Plugin_dedot_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Dedot(self, clip: "VideoNode", luma_2d: typing.Optional[int] = None, luma_t: typing.Optional[int] = None, chroma_t1: typing.Optional[int] = None, chroma_t2: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_dedot_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Dedot(self, luma_2d: typing.Optional[int] = None, luma_t: typing.Optional[int] = None, chroma_t1: typing.Optional[int] = None, chroma_t2: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: delogo
class _Plugin_delogo_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AddLogo(self, clip: "VideoNode", logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, pos_x: typing.Optional[int] = None, pos_y: typing.Optional[int] = None, depth: typing.Optional[int] = None, yc_y: typing.Optional[int] = None, yc_u: typing.Optional[int] = None, yc_v: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...
    def EraseLogo(self, clip: "VideoNode", logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, pos_x: typing.Optional[int] = None, pos_y: typing.Optional[int] = None, depth: typing.Optional[int] = None, yc_y: typing.Optional[int] = None, yc_u: typing.Optional[int] = None, yc_v: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_delogo_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AddLogo(self, logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, pos_x: typing.Optional[int] = None, pos_y: typing.Optional[int] = None, depth: typing.Optional[int] = None, yc_y: typing.Optional[int] = None, yc_u: typing.Optional[int] = None, yc_v: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...
    def EraseLogo(self, logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, pos_x: typing.Optional[int] = None, pos_y: typing.Optional[int] = None, depth: typing.Optional[int] = None, yc_y: typing.Optional[int] = None, yc_u: typing.Optional[int] = None, yc_v: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: delogohd
class _Plugin_delogohd_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AddlogoHD(self, clip: "VideoNode", logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, left: typing.Optional[int] = None, top: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, mono: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...
    def DelogoHD(self, clip: "VideoNode", logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, left: typing.Optional[int] = None, top: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, mono: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_delogohd_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AddlogoHD(self, logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, left: typing.Optional[int] = None, top: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, mono: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...
    def DelogoHD(self, logofile: typing.Union[str, bytes, bytearray], logoname: typing.Union[str, bytes, bytearray, None] = None, left: typing.Optional[int] = None, top: typing.Optional[int] = None, start: typing.Optional[int] = None, end: typing.Optional[int] = None, fadein: typing.Optional[int] = None, fadeout: typing.Optional[int] = None, mono: typing.Optional[int] = None, cutoff: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: depan
class _Plugin_depan_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DePan(self, clip: "VideoNode", data: "VideoNode", offset: typing.Optional[float] = None, pixaspect: typing.Optional[float] = None, matchfields: typing.Optional[int] = None, mirror: typing.Optional[int] = None, blur: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def DePanEstimate(self, clip: "VideoNode", range: typing.Optional[int] = None, trust: typing.Optional[float] = None, winx: typing.Optional[int] = None, winy: typing.Optional[int] = None, wleft: typing.Optional[int] = None, wtop: typing.Optional[int] = None, dxmax: typing.Optional[int] = None, dymax: typing.Optional[int] = None, zoommax: typing.Optional[float] = None, stab: typing.Optional[float] = None, pixaspect: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_depan_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DePan(self, data: "VideoNode", offset: typing.Optional[float] = None, pixaspect: typing.Optional[float] = None, matchfields: typing.Optional[int] = None, mirror: typing.Optional[int] = None, blur: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def DePanEstimate(self, range: typing.Optional[int] = None, trust: typing.Optional[float] = None, winx: typing.Optional[int] = None, winy: typing.Optional[int] = None, wleft: typing.Optional[int] = None, wtop: typing.Optional[int] = None, dxmax: typing.Optional[int] = None, dymax: typing.Optional[int] = None, zoommax: typing.Optional[float] = None, stab: typing.Optional[float] = None, pixaspect: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: descale
class _Plugin_descale_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Debicubic(self, src: "VideoNode", width: int, height: int, b: typing.Optional[float] = None, c: typing.Optional[float] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Debilinear(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Delanczos(self, src: "VideoNode", width: int, height: int, taps: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Despline16(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Despline36(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Despline64(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_descale_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Debicubic(self, width: int, height: int, b: typing.Optional[float] = None, c: typing.Optional[float] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Debilinear(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Delanczos(self, width: int, height: int, taps: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Despline16(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Despline36(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
    def Despline64(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: descale_getnative
class _Plugin_descale_getnative_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Debicubic(self, src: "VideoNode", width: int, height: int, b: typing.Optional[float] = None, c: typing.Optional[float] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Debilinear(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Delanczos(self, src: "VideoNode", width: int, height: int, taps: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Despline16(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Despline36(self, src: "VideoNode", width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_descale_getnative_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Debicubic(self, width: int, height: int, b: typing.Optional[float] = None, c: typing.Optional[float] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Debilinear(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Delanczos(self, width: int, height: int, taps: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Despline16(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
    def Despline36(self, width: int, height: int, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, cache_size: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: dfttest
class _Plugin_dfttest_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DFTTest(self, clip: "VideoNode", ftype: typing.Optional[int] = None, sigma: typing.Optional[float] = None, sigma2: typing.Optional[float] = None, pmin: typing.Optional[float] = None, pmax: typing.Optional[float] = None, sbsize: typing.Optional[int] = None, smode: typing.Optional[int] = None, sosize: typing.Optional[int] = None, tbsize: typing.Optional[int] = None, tmode: typing.Optional[int] = None, tosize: typing.Optional[int] = None, swin: typing.Optional[int] = None, twin: typing.Optional[int] = None, sbeta: typing.Optional[float] = None, tbeta: typing.Optional[float] = None, zmean: typing.Optional[int] = None, f0beta: typing.Optional[float] = None, nlocation: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, slocation: typing.Union[float, typing.Sequence[float], None] = None, ssx: typing.Union[float, typing.Sequence[float], None] = None, ssy: typing.Union[float, typing.Sequence[float], None] = None, sst: typing.Union[float, typing.Sequence[float], None] = None, ssystem: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_dfttest_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DFTTest(self, ftype: typing.Optional[int] = None, sigma: typing.Optional[float] = None, sigma2: typing.Optional[float] = None, pmin: typing.Optional[float] = None, pmax: typing.Optional[float] = None, sbsize: typing.Optional[int] = None, smode: typing.Optional[int] = None, sosize: typing.Optional[int] = None, tbsize: typing.Optional[int] = None, tmode: typing.Optional[int] = None, tosize: typing.Optional[int] = None, swin: typing.Optional[int] = None, twin: typing.Optional[int] = None, sbeta: typing.Optional[float] = None, tbeta: typing.Optional[float] = None, zmean: typing.Optional[int] = None, f0beta: typing.Optional[float] = None, nlocation: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, slocation: typing.Union[float, typing.Sequence[float], None] = None, ssx: typing.Union[float, typing.Sequence[float], None] = None, ssy: typing.Union[float, typing.Sequence[float], None] = None, sst: typing.Union[float, typing.Sequence[float], None] = None, ssystem: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: dgdecodenv
class _Plugin_dgdecodenv_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DGSource(self, source: typing.Union[str, bytes, bytearray], i420: typing.Optional[int] = None, deinterlace: typing.Optional[int] = None, use_top_field: typing.Optional[int] = None, use_pf: typing.Optional[int] = None, ct: typing.Optional[int] = None, cb: typing.Optional[int] = None, cl: typing.Optional[int] = None, cr: typing.Optional[int] = None, rw: typing.Optional[int] = None, rh: typing.Optional[int] = None, fieldop: typing.Optional[int] = None, show: typing.Optional[int] = None, show2: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_dgdecodenv_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DGSource(self, i420: typing.Optional[int] = None, deinterlace: typing.Optional[int] = None, use_top_field: typing.Optional[int] = None, use_pf: typing.Optional[int] = None, ct: typing.Optional[int] = None, cb: typing.Optional[int] = None, cl: typing.Optional[int] = None, cr: typing.Optional[int] = None, rw: typing.Optional[int] = None, rh: typing.Optional[int] = None, fieldop: typing.Optional[int] = None, show: typing.Optional[int] = None, show2: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: dghdrtosdr
class _Plugin_dghdrtosdr_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DGHDRtoSDR(self, clip: "VideoNode", light: typing.Optional[float] = None, gamma: typing.Optional[float] = None, hue: typing.Optional[float] = None, sat: typing.Optional[float] = None, tm: typing.Optional[float] = None, roll: typing.Optional[float] = None, fulldepth: typing.Optional[int] = None, impl: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_dghdrtosdr_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DGHDRtoSDR(self, light: typing.Optional[float] = None, gamma: typing.Optional[float] = None, hue: typing.Optional[float] = None, sat: typing.Optional[float] = None, tm: typing.Optional[float] = None, roll: typing.Optional[float] = None, fulldepth: typing.Optional[int] = None, impl: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: dgm
class _Plugin_dgm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DegrainMedian(self, clip: "VideoNode", limit: typing.Union[int, typing.Sequence[int], None] = None, mode: typing.Union[int, typing.Sequence[int], None] = None, interlaced: typing.Optional[int] = None, norow: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_dgm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DegrainMedian(self, limit: typing.Union[int, typing.Sequence[int], None] = None, mode: typing.Union[int, typing.Sequence[int], None] = None, interlaced: typing.Optional[int] = None, norow: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: dotkill
class _Plugin_dotkill_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DotKillS(self, clip: "VideoNode", iterations: typing.Optional[int] = None, usematch: typing.Optional[int] = None) -> "VideoNode": ...
    def DotKillT(self, clip: "VideoNode", order: typing.Optional[int] = None, offset: typing.Optional[int] = None, dupthresh: typing.Optional[int] = None, tratio: typing.Optional[int] = None, show: typing.Optional[int] = None) -> "VideoNode": ...
    def DotKillZ(self, clip: "VideoNode", order: typing.Optional[int] = None, offset: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_dotkill_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DotKillS(self, iterations: typing.Optional[int] = None, usematch: typing.Optional[int] = None) -> "VideoNode": ...
    def DotKillT(self, order: typing.Optional[int] = None, offset: typing.Optional[int] = None, dupthresh: typing.Optional[int] = None, tratio: typing.Optional[int] = None, show: typing.Optional[int] = None) -> "VideoNode": ...
    def DotKillZ(self, order: typing.Optional[int] = None, offset: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: eedi2
class _Plugin_eedi2_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def EEDI2(self, clip: "VideoNode", field: int, mthresh: typing.Optional[int] = None, lthresh: typing.Optional[int] = None, vthresh: typing.Optional[int] = None, estr: typing.Optional[int] = None, dstr: typing.Optional[int] = None, maxd: typing.Optional[int] = None, map: typing.Optional[int] = None, nt: typing.Optional[int] = None, pp: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_eedi2_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def EEDI2(self, field: int, mthresh: typing.Optional[int] = None, lthresh: typing.Optional[int] = None, vthresh: typing.Optional[int] = None, estr: typing.Optional[int] = None, dstr: typing.Optional[int] = None, maxd: typing.Optional[int] = None, map: typing.Optional[int] = None, nt: typing.Optional[int] = None, pp: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: eedi3
class _Plugin_eedi3_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def eedi3(self, clip: "VideoNode", field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, beta: typing.Optional[float] = None, gamma: typing.Optional[float] = None, nrad: typing.Optional[int] = None, mdis: typing.Optional[int] = None, hp: typing.Optional[int] = None, ucubic: typing.Optional[int] = None, cost3: typing.Optional[int] = None, vcheck: typing.Optional[int] = None, vthresh0: typing.Optional[float] = None, vthresh1: typing.Optional[float] = None, vthresh2: typing.Optional[float] = None, sclip: typing.Optional["VideoNode"] = None) -> "VideoNode": ...


class _Plugin_eedi3_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def eedi3(self, field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, beta: typing.Optional[float] = None, gamma: typing.Optional[float] = None, nrad: typing.Optional[int] = None, mdis: typing.Optional[int] = None, hp: typing.Optional[int] = None, ucubic: typing.Optional[int] = None, cost3: typing.Optional[int] = None, vcheck: typing.Optional[int] = None, vthresh0: typing.Optional[float] = None, vthresh1: typing.Optional[float] = None, vthresh2: typing.Optional[float] = None, sclip: typing.Optional["VideoNode"] = None) -> "VideoNode": ...
# end implementation


# implementation: eedi3m
class _Plugin_eedi3m_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def EEDI3(self, clip: "VideoNode", field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, beta: typing.Optional[float] = None, gamma: typing.Optional[float] = None, nrad: typing.Optional[int] = None, mdis: typing.Optional[int] = None, hp: typing.Optional[int] = None, ucubic: typing.Optional[int] = None, cost3: typing.Optional[int] = None, vcheck: typing.Optional[int] = None, vthresh0: typing.Optional[float] = None, vthresh1: typing.Optional[float] = None, vthresh2: typing.Optional[float] = None, sclip: typing.Optional["VideoNode"] = None, mclip: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def EEDI3CL(self, clip: "VideoNode", field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, beta: typing.Optional[float] = None, gamma: typing.Optional[float] = None, nrad: typing.Optional[int] = None, mdis: typing.Optional[int] = None, hp: typing.Optional[int] = None, ucubic: typing.Optional[int] = None, cost3: typing.Optional[int] = None, vcheck: typing.Optional[int] = None, vthresh0: typing.Optional[float] = None, vthresh1: typing.Optional[float] = None, vthresh2: typing.Optional[float] = None, sclip: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None, device: typing.Optional[int] = None, list_device: typing.Optional[int] = None, info: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_eedi3m_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def EEDI3(self, field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, beta: typing.Optional[float] = None, gamma: typing.Optional[float] = None, nrad: typing.Optional[int] = None, mdis: typing.Optional[int] = None, hp: typing.Optional[int] = None, ucubic: typing.Optional[int] = None, cost3: typing.Optional[int] = None, vcheck: typing.Optional[int] = None, vthresh0: typing.Optional[float] = None, vthresh1: typing.Optional[float] = None, vthresh2: typing.Optional[float] = None, sclip: typing.Optional["VideoNode"] = None, mclip: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def EEDI3CL(self, field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, alpha: typing.Optional[float] = None, beta: typing.Optional[float] = None, gamma: typing.Optional[float] = None, nrad: typing.Optional[int] = None, mdis: typing.Optional[int] = None, hp: typing.Optional[int] = None, ucubic: typing.Optional[int] = None, cost3: typing.Optional[int] = None, vcheck: typing.Optional[int] = None, vthresh0: typing.Optional[float] = None, vthresh1: typing.Optional[float] = None, vthresh2: typing.Optional[float] = None, sclip: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None, device: typing.Optional[int] = None, list_device: typing.Optional[int] = None, info: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: f3kdb
class _Plugin_f3kdb_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deband(self, clip: "VideoNode", range: typing.Optional[int] = None, y: typing.Optional[int] = None, cb: typing.Optional[int] = None, cr: typing.Optional[int] = None, grainy: typing.Optional[int] = None, grainc: typing.Optional[int] = None, sample_mode: typing.Optional[int] = None, seed: typing.Optional[int] = None, blur_first: typing.Optional[int] = None, dynamic_grain: typing.Optional[int] = None, opt: typing.Optional[int] = None, dither_algo: typing.Optional[int] = None, keep_tv_range: typing.Optional[int] = None, output_depth: typing.Optional[int] = None, random_algo_ref: typing.Optional[int] = None, random_algo_grain: typing.Optional[int] = None, random_param_ref: typing.Optional[float] = None, random_param_grain: typing.Optional[float] = None, preset: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_f3kdb_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deband(self, range: typing.Optional[int] = None, y: typing.Optional[int] = None, cb: typing.Optional[int] = None, cr: typing.Optional[int] = None, grainy: typing.Optional[int] = None, grainc: typing.Optional[int] = None, sample_mode: typing.Optional[int] = None, seed: typing.Optional[int] = None, blur_first: typing.Optional[int] = None, dynamic_grain: typing.Optional[int] = None, opt: typing.Optional[int] = None, dither_algo: typing.Optional[int] = None, keep_tv_range: typing.Optional[int] = None, output_depth: typing.Optional[int] = None, random_algo_ref: typing.Optional[int] = None, random_algo_grain: typing.Optional[int] = None, random_param_ref: typing.Optional[float] = None, random_param_grain: typing.Optional[float] = None, preset: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: fb
class _Plugin_fb_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FillBorders(self, clip: "VideoNode", left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, mode: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_fb_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FillBorders(self, left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, mode: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: ffms2
class _Plugin_ffms2_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def GetLogLevel(self) -> "VideoNode": ...
    def Index(self, source: typing.Union[str, bytes, bytearray], cachefile: typing.Union[str, bytes, bytearray, None] = None, indextracks: typing.Union[int, typing.Sequence[int], None] = None, errorhandling: typing.Optional[int] = None, overwrite: typing.Optional[int] = None) -> "VideoNode": ...
    def SetLogLevel(self, level: int) -> "VideoNode": ...
    def Source(self, source: typing.Union[str, bytes, bytearray], track: typing.Optional[int] = None, cache: typing.Optional[int] = None, cachefile: typing.Union[str, bytes, bytearray, None] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, threads: typing.Optional[int] = None, timecodes: typing.Union[str, bytes, bytearray, None] = None, seekmode: typing.Optional[int] = None, width: typing.Optional[int] = None, height: typing.Optional[int] = None, resizer: typing.Union[str, bytes, bytearray, None] = None, format: typing.Optional[int] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def Version(self) -> "VideoNode": ...


class _Plugin_ffms2_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def GetLogLevel(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Index(self, cachefile: typing.Union[str, bytes, bytearray, None] = None, indextracks: typing.Union[int, typing.Sequence[int], None] = None, errorhandling: typing.Optional[int] = None, overwrite: typing.Optional[int] = None) -> "VideoNode": ...
    def SetLogLevel(self) -> "VideoNode": ...
    def Source(self, track: typing.Optional[int] = None, cache: typing.Optional[int] = None, cachefile: typing.Union[str, bytes, bytearray, None] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, threads: typing.Optional[int] = None, timecodes: typing.Union[str, bytes, bytearray, None] = None, seekmode: typing.Optional[int] = None, width: typing.Optional[int] = None, height: typing.Optional[int] = None, resizer: typing.Union[str, bytes, bytearray, None] = None, format: typing.Optional[int] = None, alpha: typing.Optional[int] = None) -> "VideoNode": ...
    def Version(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
# end implementation


# implementation: fft3dfilter
class _Plugin_fft3dfilter_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FFT3DFilter(self, clip: "VideoNode", sigma: typing.Optional[float] = None, beta: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, bw: typing.Optional[int] = None, bh: typing.Optional[int] = None, bt: typing.Optional[int] = None, ow: typing.Optional[int] = None, oh: typing.Optional[int] = None, kratio: typing.Optional[float] = None, sharpen: typing.Optional[float] = None, scutoff: typing.Optional[float] = None, svr: typing.Optional[float] = None, smin: typing.Optional[float] = None, smax: typing.Optional[float] = None, measure: typing.Optional[int] = None, interlaced: typing.Optional[int] = None, wintype: typing.Optional[int] = None, pframe: typing.Optional[int] = None, px: typing.Optional[int] = None, py: typing.Optional[int] = None, pshow: typing.Optional[int] = None, pcutoff: typing.Optional[float] = None, pfactor: typing.Optional[float] = None, sigma2: typing.Optional[float] = None, sigma3: typing.Optional[float] = None, sigma4: typing.Optional[float] = None, degrid: typing.Optional[float] = None, dehalo: typing.Optional[float] = None, hr: typing.Optional[float] = None, ht: typing.Optional[float] = None, ncpu: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_fft3dfilter_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FFT3DFilter(self, sigma: typing.Optional[float] = None, beta: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, bw: typing.Optional[int] = None, bh: typing.Optional[int] = None, bt: typing.Optional[int] = None, ow: typing.Optional[int] = None, oh: typing.Optional[int] = None, kratio: typing.Optional[float] = None, sharpen: typing.Optional[float] = None, scutoff: typing.Optional[float] = None, svr: typing.Optional[float] = None, smin: typing.Optional[float] = None, smax: typing.Optional[float] = None, measure: typing.Optional[int] = None, interlaced: typing.Optional[int] = None, wintype: typing.Optional[int] = None, pframe: typing.Optional[int] = None, px: typing.Optional[int] = None, py: typing.Optional[int] = None, pshow: typing.Optional[int] = None, pcutoff: typing.Optional[float] = None, pfactor: typing.Optional[float] = None, sigma2: typing.Optional[float] = None, sigma3: typing.Optional[float] = None, sigma4: typing.Optional[float] = None, degrid: typing.Optional[float] = None, dehalo: typing.Optional[float] = None, hr: typing.Optional[float] = None, ht: typing.Optional[float] = None, ncpu: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: fftspectrum
class _Plugin_fftspectrum_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FFTSpectrum(self, clip: "VideoNode", grid: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_fftspectrum_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FFTSpectrum(self, grid: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: fh
class _Plugin_fh_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FieldHint(self, clip: "VideoNode", ovr: typing.Union[str, bytes, bytearray, None] = None, tff: typing.Optional[int] = None, matches: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Fieldhint(self, clip: "VideoNode", ovr: typing.Union[str, bytes, bytearray, None] = None, tff: typing.Optional[int] = None, matches: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_fh_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FieldHint(self, ovr: typing.Union[str, bytes, bytearray, None] = None, tff: typing.Optional[int] = None, matches: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Fieldhint(self, ovr: typing.Union[str, bytes, bytearray, None] = None, tff: typing.Optional[int] = None, matches: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: flux
class _Plugin_flux_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SmoothST(self, clip: "VideoNode", temporal_threshold: typing.Optional[int] = None, spatial_threshold: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def SmoothT(self, clip: "VideoNode", temporal_threshold: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_flux_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SmoothST(self, temporal_threshold: typing.Optional[int] = None, spatial_threshold: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def SmoothT(self, temporal_threshold: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: fmtc
class _Plugin_fmtc_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def bitdepth(self, clip: "VideoNode", csp: typing.Optional[int] = None, bits: typing.Optional[int] = None, flt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, dmode: typing.Optional[int] = None, ampo: typing.Optional[float] = None, ampn: typing.Optional[float] = None, dyn: typing.Optional[int] = None, staticnoise: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None, patsize: typing.Optional[int] = None) -> "VideoNode": ...
    def histluma(self, clip: "VideoNode", full: typing.Optional[int] = None, amp: typing.Optional[int] = None) -> "VideoNode": ...
    def matrix(self, clip: "VideoNode", mat: typing.Union[str, bytes, bytearray, None] = None, mats: typing.Union[str, bytes, bytearray, None] = None, matd: typing.Union[str, bytes, bytearray, None] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, coef: typing.Union[float, typing.Sequence[float], None] = None, csp: typing.Optional[int] = None, col_fam: typing.Optional[int] = None, bits: typing.Optional[int] = None, singleout: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def matrix2020cl(self, clip: "VideoNode", full: typing.Optional[int] = None, csp: typing.Optional[int] = None, bits: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def nativetostack16(self, clip: "VideoNode") -> "VideoNode": ...
    def primaries(self, clip: "VideoNode", rs: typing.Union[float, typing.Sequence[float], None] = None, gs: typing.Union[float, typing.Sequence[float], None] = None, bs: typing.Union[float, typing.Sequence[float], None] = None, ws: typing.Union[float, typing.Sequence[float], None] = None, rd: typing.Union[float, typing.Sequence[float], None] = None, gd: typing.Union[float, typing.Sequence[float], None] = None, bd: typing.Union[float, typing.Sequence[float], None] = None, wd: typing.Union[float, typing.Sequence[float], None] = None, prims: typing.Union[str, bytes, bytearray, None] = None, primd: typing.Union[str, bytes, bytearray, None] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def resample(self, clip: "VideoNode", w: typing.Optional[int] = None, h: typing.Optional[int] = None, sx: typing.Union[float, typing.Sequence[float], None] = None, sy: typing.Union[float, typing.Sequence[float], None] = None, sw: typing.Union[float, typing.Sequence[float], None] = None, sh: typing.Union[float, typing.Sequence[float], None] = None, scale: typing.Optional[float] = None, scaleh: typing.Optional[float] = None, scalev: typing.Optional[float] = None, kernel: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, kernelh: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, kernelv: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, impulse: typing.Union[float, typing.Sequence[float], None] = None, impulseh: typing.Union[float, typing.Sequence[float], None] = None, impulsev: typing.Union[float, typing.Sequence[float], None] = None, taps: typing.Union[int, typing.Sequence[int], None] = None, tapsh: typing.Union[int, typing.Sequence[int], None] = None, tapsv: typing.Union[int, typing.Sequence[int], None] = None, a1: typing.Union[float, typing.Sequence[float], None] = None, a2: typing.Union[float, typing.Sequence[float], None] = None, a3: typing.Union[float, typing.Sequence[float], None] = None, kovrspl: typing.Union[int, typing.Sequence[int], None] = None, fh: typing.Union[float, typing.Sequence[float], None] = None, fv: typing.Union[float, typing.Sequence[float], None] = None, cnorm: typing.Union[int, typing.Sequence[int], None] = None, totalh: typing.Union[float, typing.Sequence[float], None] = None, totalv: typing.Union[float, typing.Sequence[float], None] = None, invks: typing.Union[int, typing.Sequence[int], None] = None, invksh: typing.Union[int, typing.Sequence[int], None] = None, invksv: typing.Union[int, typing.Sequence[int], None] = None, invkstaps: typing.Union[int, typing.Sequence[int], None] = None, invkstapsh: typing.Union[int, typing.Sequence[int], None] = None, invkstapsv: typing.Union[int, typing.Sequence[int], None] = None, csp: typing.Optional[int] = None, css: typing.Union[str, bytes, bytearray, None] = None, planes: typing.Union[float, typing.Sequence[float], None] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, center: typing.Union[int, typing.Sequence[int], None] = None, cplace: typing.Union[str, bytes, bytearray, None] = None, cplaces: typing.Union[str, bytes, bytearray, None] = None, cplaced: typing.Union[str, bytes, bytearray, None] = None, interlaced: typing.Optional[int] = None, interlacedd: typing.Optional[int] = None, tff: typing.Optional[int] = None, tffd: typing.Optional[int] = None, flt: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def stack16tonative(self, clip: "VideoNode") -> "VideoNode": ...
    def transfer(self, clip: "VideoNode", transs: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, transd: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, cont: typing.Optional[float] = None, gcor: typing.Optional[float] = None, bits: typing.Optional[int] = None, flt: typing.Optional[int] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None, blacklvl: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_fmtc_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def bitdepth(self, csp: typing.Optional[int] = None, bits: typing.Optional[int] = None, flt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, dmode: typing.Optional[int] = None, ampo: typing.Optional[float] = None, ampn: typing.Optional[float] = None, dyn: typing.Optional[int] = None, staticnoise: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None, patsize: typing.Optional[int] = None) -> "VideoNode": ...
    def histluma(self, full: typing.Optional[int] = None, amp: typing.Optional[int] = None) -> "VideoNode": ...
    def matrix(self, mat: typing.Union[str, bytes, bytearray, None] = None, mats: typing.Union[str, bytes, bytearray, None] = None, matd: typing.Union[str, bytes, bytearray, None] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, coef: typing.Union[float, typing.Sequence[float], None] = None, csp: typing.Optional[int] = None, col_fam: typing.Optional[int] = None, bits: typing.Optional[int] = None, singleout: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def matrix2020cl(self, full: typing.Optional[int] = None, csp: typing.Optional[int] = None, bits: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def nativetostack16(self) -> "VideoNode": ...
    def primaries(self, rs: typing.Union[float, typing.Sequence[float], None] = None, gs: typing.Union[float, typing.Sequence[float], None] = None, bs: typing.Union[float, typing.Sequence[float], None] = None, ws: typing.Union[float, typing.Sequence[float], None] = None, rd: typing.Union[float, typing.Sequence[float], None] = None, gd: typing.Union[float, typing.Sequence[float], None] = None, bd: typing.Union[float, typing.Sequence[float], None] = None, wd: typing.Union[float, typing.Sequence[float], None] = None, prims: typing.Union[str, bytes, bytearray, None] = None, primd: typing.Union[str, bytes, bytearray, None] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def resample(self, w: typing.Optional[int] = None, h: typing.Optional[int] = None, sx: typing.Union[float, typing.Sequence[float], None] = None, sy: typing.Union[float, typing.Sequence[float], None] = None, sw: typing.Union[float, typing.Sequence[float], None] = None, sh: typing.Union[float, typing.Sequence[float], None] = None, scale: typing.Optional[float] = None, scaleh: typing.Optional[float] = None, scalev: typing.Optional[float] = None, kernel: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, kernelh: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, kernelv: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, impulse: typing.Union[float, typing.Sequence[float], None] = None, impulseh: typing.Union[float, typing.Sequence[float], None] = None, impulsev: typing.Union[float, typing.Sequence[float], None] = None, taps: typing.Union[int, typing.Sequence[int], None] = None, tapsh: typing.Union[int, typing.Sequence[int], None] = None, tapsv: typing.Union[int, typing.Sequence[int], None] = None, a1: typing.Union[float, typing.Sequence[float], None] = None, a2: typing.Union[float, typing.Sequence[float], None] = None, a3: typing.Union[float, typing.Sequence[float], None] = None, kovrspl: typing.Union[int, typing.Sequence[int], None] = None, fh: typing.Union[float, typing.Sequence[float], None] = None, fv: typing.Union[float, typing.Sequence[float], None] = None, cnorm: typing.Union[int, typing.Sequence[int], None] = None, totalh: typing.Union[float, typing.Sequence[float], None] = None, totalv: typing.Union[float, typing.Sequence[float], None] = None, invks: typing.Union[int, typing.Sequence[int], None] = None, invksh: typing.Union[int, typing.Sequence[int], None] = None, invksv: typing.Union[int, typing.Sequence[int], None] = None, invkstaps: typing.Union[int, typing.Sequence[int], None] = None, invkstapsh: typing.Union[int, typing.Sequence[int], None] = None, invkstapsv: typing.Union[int, typing.Sequence[int], None] = None, csp: typing.Optional[int] = None, css: typing.Union[str, bytes, bytearray, None] = None, planes: typing.Union[float, typing.Sequence[float], None] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, center: typing.Union[int, typing.Sequence[int], None] = None, cplace: typing.Union[str, bytes, bytearray, None] = None, cplaces: typing.Union[str, bytes, bytearray, None] = None, cplaced: typing.Union[str, bytes, bytearray, None] = None, interlaced: typing.Optional[int] = None, interlacedd: typing.Optional[int] = None, tff: typing.Optional[int] = None, tffd: typing.Optional[int] = None, flt: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None) -> "VideoNode": ...
    def stack16tonative(self) -> "VideoNode": ...
    def transfer(self, transs: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, transd: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, cont: typing.Optional[float] = None, gcor: typing.Optional[float] = None, bits: typing.Optional[int] = None, flt: typing.Optional[int] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, cpuopt: typing.Optional[int] = None, blacklvl: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: focus
class _Plugin_focus_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SpatialSoften(self, clip: "VideoNode", radius: typing.Optional[int] = None, luma_threshold: typing.Optional[float] = None, chroma_threshold: typing.Optional[float] = None) -> "VideoNode": ...
    def TemporalSoften(self, clip: "VideoNode", radius: typing.Optional[int] = None, luma_threshold: typing.Optional[float] = None, chroma_threshold: typing.Optional[float] = None, scenechange: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_focus_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SpatialSoften(self, radius: typing.Optional[int] = None, luma_threshold: typing.Optional[float] = None, chroma_threshold: typing.Optional[float] = None) -> "VideoNode": ...
    def TemporalSoften(self, radius: typing.Optional[int] = None, luma_threshold: typing.Optional[float] = None, chroma_threshold: typing.Optional[float] = None, scenechange: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: focus2
class _Plugin_focus2_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TemporalSoften2(self, clip: "VideoNode", radius: typing.Optional[int] = None, luma_threshold: typing.Optional[int] = None, chroma_threshold: typing.Optional[int] = None, scenechange: typing.Optional[int] = None, mode: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_focus2_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TemporalSoften2(self, radius: typing.Optional[int] = None, luma_threshold: typing.Optional[int] = None, chroma_threshold: typing.Optional[int] = None, scenechange: typing.Optional[int] = None, mode: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: ftf
class _Plugin_ftf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FixFades(self, clip: "VideoNode", mode: typing.Optional[int] = None, threshold: typing.Optional[float] = None, color: typing.Union[float, typing.Sequence[float], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_ftf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def FixFades(self, mode: typing.Optional[int] = None, threshold: typing.Optional[float] = None, color: typing.Union[float, typing.Sequence[float], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: grad
class _Plugin_grad_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Curve(self, clip: "VideoNode", fname: typing.Union[str, bytes, bytearray, None] = None, ftype: typing.Optional[int] = None, pmode: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_grad_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Curve(self, fname: typing.Union[str, bytes, bytearray, None] = None, ftype: typing.Optional[int] = None, pmode: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: grain
class _Plugin_grain_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Add(self, clip: "VideoNode", var: typing.Optional[float] = None, uvar: typing.Optional[float] = None, hcorr: typing.Optional[float] = None, vcorr: typing.Optional[float] = None, seed: typing.Optional[int] = None, constant: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_grain_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Add(self, var: typing.Optional[float] = None, uvar: typing.Optional[float] = None, hcorr: typing.Optional[float] = None, vcorr: typing.Optional[float] = None, seed: typing.Optional[int] = None, constant: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: hist
class _Plugin_hist_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Classic(self, clip: "VideoNode") -> "VideoNode": ...
    def Color(self, clip: "VideoNode") -> "VideoNode": ...
    def Color2(self, clip: "VideoNode") -> "VideoNode": ...
    def Levels(self, clip: "VideoNode", factor: typing.Optional[float] = None) -> "VideoNode": ...
    def Luma(self, clip: "VideoNode") -> "VideoNode": ...


class _Plugin_hist_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Classic(self) -> "VideoNode": ...
    def Color(self) -> "VideoNode": ...
    def Color2(self) -> "VideoNode": ...
    def Levels(self, factor: typing.Optional[float] = None) -> "VideoNode": ...
    def Luma(self) -> "VideoNode": ...
# end implementation


# implementation: hqdn3d
class _Plugin_hqdn3d_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Hqdn3d(self, clip: "VideoNode", lum_spac: typing.Optional[float] = None, chrom_spac: typing.Optional[float] = None, lum_tmp: typing.Optional[float] = None, chrom_tmp: typing.Optional[float] = None, restart_lap: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_hqdn3d_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Hqdn3d(self, lum_spac: typing.Optional[float] = None, chrom_spac: typing.Optional[float] = None, lum_tmp: typing.Optional[float] = None, chrom_tmp: typing.Optional[float] = None, restart_lap: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: imwri
class _Plugin_imwri_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Read(self, filename: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], firstnum: typing.Optional[int] = None, mismatch: typing.Optional[int] = None, alpha: typing.Optional[int] = None, float_output: typing.Optional[int] = None) -> "VideoNode": ...
    def Write(self, clip: "VideoNode", imgformat: typing.Union[str, bytes, bytearray], filename: typing.Union[str, bytes, bytearray], firstnum: typing.Optional[int] = None, quality: typing.Optional[int] = None, dither: typing.Optional[int] = None, compression_type: typing.Union[str, bytes, bytearray, None] = None, overwrite: typing.Optional[int] = None, alpha: typing.Optional["VideoNode"] = None) -> "VideoNode": ...


class _Plugin_imwri_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Read(self, firstnum: typing.Optional[int] = None, mismatch: typing.Optional[int] = None, alpha: typing.Optional[int] = None, float_output: typing.Optional[int] = None) -> "VideoNode": ...
    def Write(self, imgformat: typing.Union[str, bytes, bytearray], filename: typing.Union[str, bytes, bytearray], firstnum: typing.Optional[int] = None, quality: typing.Optional[int] = None, dither: typing.Optional[int] = None, compression_type: typing.Union[str, bytes, bytearray, None] = None, overwrite: typing.Optional[int] = None, alpha: typing.Optional["VideoNode"] = None) -> "VideoNode": ...
# end implementation


# implementation: it
class _Plugin_it_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def IT(self, clip: "VideoNode", fps: typing.Optional[int] = None, threshold: typing.Optional[int] = None, pthreshold: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_it_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def IT(self, fps: typing.Optional[int] = None, threshold: typing.Optional[int] = None, pthreshold: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: knlm
class _Plugin_knlm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def KNLMeansCL(self, clip: "VideoNode", d: typing.Optional[int] = None, a: typing.Optional[int] = None, s: typing.Optional[int] = None, h: typing.Optional[float] = None, channels: typing.Union[str, bytes, bytearray, None] = None, wmode: typing.Optional[int] = None, wref: typing.Optional[float] = None, rclip: typing.Optional["VideoNode"] = None, device_type: typing.Union[str, bytes, bytearray, None] = None, device_id: typing.Optional[int] = None, ocl_x: typing.Optional[int] = None, ocl_y: typing.Optional[int] = None, ocl_r: typing.Optional[int] = None, info: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_knlm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def KNLMeansCL(self, d: typing.Optional[int] = None, a: typing.Optional[int] = None, s: typing.Optional[int] = None, h: typing.Optional[float] = None, channels: typing.Union[str, bytes, bytearray, None] = None, wmode: typing.Optional[int] = None, wref: typing.Optional[float] = None, rclip: typing.Optional["VideoNode"] = None, device_type: typing.Union[str, bytes, bytearray, None] = None, device_id: typing.Optional[int] = None, ocl_x: typing.Optional[int] = None, ocl_y: typing.Optional[int] = None, ocl_r: typing.Optional[int] = None, info: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: lsmas
class _Plugin_lsmas_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def LWLibavSource(self, source: typing.Union[str, bytes, bytearray], stream_index: typing.Optional[int] = None, cache: typing.Optional[int] = None, threads: typing.Optional[int] = None, seek_mode: typing.Optional[int] = None, seek_threshold: typing.Optional[int] = None, dr: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, variable: typing.Optional[int] = None, format: typing.Union[str, bytes, bytearray, None] = None, decoder: typing.Union[str, bytes, bytearray, None] = None, repeat: typing.Optional[int] = None, dominance: typing.Optional[int] = None) -> "VideoNode": ...
    def LibavSMASHSource(self, source: typing.Union[str, bytes, bytearray], track: typing.Optional[int] = None, threads: typing.Optional[int] = None, seek_mode: typing.Optional[int] = None, seek_threshold: typing.Optional[int] = None, dr: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, variable: typing.Optional[int] = None, format: typing.Union[str, bytes, bytearray, None] = None, decoder: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_lsmas_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def LWLibavSource(self, stream_index: typing.Optional[int] = None, cache: typing.Optional[int] = None, threads: typing.Optional[int] = None, seek_mode: typing.Optional[int] = None, seek_threshold: typing.Optional[int] = None, dr: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, variable: typing.Optional[int] = None, format: typing.Union[str, bytes, bytearray, None] = None, decoder: typing.Union[str, bytes, bytearray, None] = None, repeat: typing.Optional[int] = None, dominance: typing.Optional[int] = None) -> "VideoNode": ...
    def LibavSMASHSource(self, track: typing.Optional[int] = None, threads: typing.Optional[int] = None, seek_mode: typing.Optional[int] = None, seek_threshold: typing.Optional[int] = None, dr: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, variable: typing.Optional[int] = None, format: typing.Union[str, bytes, bytearray, None] = None, decoder: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: median
class _Plugin_median_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Median(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], sync: typing.Optional[int] = None, samples: typing.Optional[int] = None, debug: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def MedianBlend(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], low: typing.Optional[int] = None, high: typing.Optional[int] = None, closest: typing.Optional[int] = None, sync: typing.Optional[int] = None, samples: typing.Optional[int] = None, debug: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def TemporalMedian(self, clip: "VideoNode", radius: typing.Optional[int] = None, debug: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_median_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Median(self, sync: typing.Optional[int] = None, samples: typing.Optional[int] = None, debug: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def MedianBlend(self, low: typing.Optional[int] = None, high: typing.Optional[int] = None, closest: typing.Optional[int] = None, sync: typing.Optional[int] = None, samples: typing.Optional[int] = None, debug: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def TemporalMedian(self, radius: typing.Optional[int] = None, debug: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: minideen
class _Plugin_minideen_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MiniDeen(self, clip: "VideoNode", radius: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Union[int, typing.Sequence[int], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_minideen_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MiniDeen(self, radius: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Union[int, typing.Sequence[int], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: minsrp
class _Plugin_minsrp_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Sharp(self, clip: "VideoNode", str: typing.Union[float, typing.Sequence[float], None] = None, mode: typing.Union[int, typing.Sequence[int], None] = None, linear: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_minsrp_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Sharp(self, str: typing.Union[float, typing.Sequence[float], None] = None, mode: typing.Union[int, typing.Sequence[int], None] = None, linear: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: misc
class _Plugin_misc_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AverageFrames(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], weights: typing.Union[float, typing.Sequence[float]], scale: typing.Optional[float] = None, scenechange: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Hysteresis(self, clipa: "VideoNode", clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def SCDetect(self, clip: "VideoNode", threshold: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_misc_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AverageFrames(self, weights: typing.Union[float, typing.Sequence[float]], scale: typing.Optional[float] = None, scenechange: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Hysteresis(self, clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def SCDetect(self, threshold: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: morpho
class _Plugin_morpho_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def BottomHat(self, clip: "VideoNode", size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Close(self, clip: "VideoNode", size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Dilate(self, clip: "VideoNode", size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Erode(self, clip: "VideoNode", size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Open(self, clip: "VideoNode", size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def TopHat(self, clip: "VideoNode", size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_morpho_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def BottomHat(self, size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Close(self, size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Dilate(self, size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Erode(self, size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def Open(self, size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
    def TopHat(self, size: typing.Optional[int] = None, shape: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: motionmask
class _Plugin_motionmask_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MotionMask(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, th1: typing.Union[int, typing.Sequence[int], None] = None, th2: typing.Union[int, typing.Sequence[int], None] = None, tht: typing.Optional[int] = None, sc_value: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_motionmask_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MotionMask(self, planes: typing.Union[int, typing.Sequence[int], None] = None, th1: typing.Union[int, typing.Sequence[int], None] = None, th2: typing.Union[int, typing.Sequence[int], None] = None, tht: typing.Optional[int] = None, sc_value: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: msmoosh
class _Plugin_msmoosh_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MSharpen(self, clip: "VideoNode", threshold: typing.Optional[float] = None, strength: typing.Optional[float] = None, mask: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def MSmooth(self, clip: "VideoNode", threshold: typing.Optional[float] = None, strength: typing.Optional[int] = None, mask: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_msmoosh_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MSharpen(self, threshold: typing.Optional[float] = None, strength: typing.Optional[float] = None, mask: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def MSmooth(self, threshold: typing.Optional[float] = None, strength: typing.Optional[int] = None, mask: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: mv
class _Plugin_mv_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Analyse(self, super: "VideoNode", blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, levels: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, pelsearch: typing.Optional[int] = None, isb: typing.Optional[int] = None, lambda_: typing.Optional[int] = None, chroma: typing.Optional[int] = None, delta: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, lsad: typing.Optional[int] = None, plevel: typing.Optional[int] = None, global_: typing.Optional[int] = None, pnew: typing.Optional[int] = None, pzero: typing.Optional[int] = None, pglobal: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, badsad: typing.Optional[int] = None, badrange: typing.Optional[int] = None, opt: typing.Optional[int] = None, meander: typing.Optional[int] = None, trymany: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, search_coarse: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def BlockFPS(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mode: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Compensate(self, clip: "VideoNode", super: "VideoNode", vectors: "VideoNode", scbehavior: typing.Optional[int] = None, thsad: typing.Optional[int] = None, fields: typing.Optional[int] = None, time: typing.Optional[float] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain1(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", thsad: typing.Optional[int] = None, thsadc: typing.Optional[int] = None, plane: typing.Optional[int] = None, limit: typing.Optional[int] = None, limitc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain2(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", thsad: typing.Optional[int] = None, thsadc: typing.Optional[int] = None, plane: typing.Optional[int] = None, limit: typing.Optional[int] = None, limitc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain3(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", thsad: typing.Optional[int] = None, thsadc: typing.Optional[int] = None, plane: typing.Optional[int] = None, limit: typing.Optional[int] = None, limitc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanAnalyse(self, clip: "VideoNode", vectors: "VideoNode", mask: typing.Optional["VideoNode"] = None, zoom: typing.Optional[int] = None, rot: typing.Optional[int] = None, pixaspect: typing.Optional[float] = None, error: typing.Optional[float] = None, info: typing.Optional[int] = None, wrong: typing.Optional[float] = None, zerow: typing.Optional[float] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanCompensate(self, clip: "VideoNode", data: "VideoNode", offset: typing.Optional[float] = None, subpixel: typing.Optional[int] = None, pixaspect: typing.Optional[float] = None, matchfields: typing.Optional[int] = None, mirror: typing.Optional[int] = None, blur: typing.Optional[int] = None, info: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanEstimate(self, clip: "VideoNode", trust: typing.Optional[float] = None, winx: typing.Optional[int] = None, winy: typing.Optional[int] = None, wleft: typing.Optional[int] = None, wtop: typing.Optional[int] = None, dxmax: typing.Optional[int] = None, dymax: typing.Optional[int] = None, zoommax: typing.Optional[float] = None, stab: typing.Optional[float] = None, pixaspect: typing.Optional[float] = None, info: typing.Optional[int] = None, show: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanStabilise(self, clip: "VideoNode", data: "VideoNode", cutoff: typing.Optional[float] = None, damping: typing.Optional[float] = None, initzoom: typing.Optional[float] = None, addzoom: typing.Optional[int] = None, prev: typing.Optional[int] = None, next: typing.Optional[int] = None, mirror: typing.Optional[int] = None, blur: typing.Optional[int] = None, dxmax: typing.Optional[float] = None, dymax: typing.Optional[float] = None, zoommax: typing.Optional[float] = None, rotmax: typing.Optional[float] = None, subpixel: typing.Optional[int] = None, pixaspect: typing.Optional[float] = None, fitlast: typing.Optional[int] = None, tzoom: typing.Optional[float] = None, info: typing.Optional[int] = None, method: typing.Optional[int] = None, fields: typing.Optional[int] = None) -> "VideoNode": ...
    def Finest(self, super: "VideoNode", opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Flow(self, clip: "VideoNode", super: "VideoNode", vectors: "VideoNode", time: typing.Optional[float] = None, mode: typing.Optional[int] = None, fields: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowBlur(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", blur: typing.Optional[float] = None, prec: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowFPS(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mask: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowInter(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", time: typing.Optional[float] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Mask(self, clip: "VideoNode", vectors: "VideoNode", ml: typing.Optional[float] = None, gamma: typing.Optional[float] = None, kind: typing.Optional[int] = None, time: typing.Optional[float] = None, ysc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Recalculate(self, super: "VideoNode", vectors: "VideoNode", thsad: typing.Optional[int] = None, smooth: typing.Optional[int] = None, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, lambda_: typing.Optional[int] = None, chroma: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, pnew: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, opt: typing.Optional[int] = None, meander: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def SCDetection(self, clip: "VideoNode", vectors: "VideoNode", thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None) -> "VideoNode": ...
    def Super(self, clip: "VideoNode", hpad: typing.Optional[int] = None, vpad: typing.Optional[int] = None, pel: typing.Optional[int] = None, levels: typing.Optional[int] = None, chroma: typing.Optional[int] = None, sharp: typing.Optional[int] = None, rfilter: typing.Optional[int] = None, pelclip: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_mv_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Analyse(self, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, levels: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, pelsearch: typing.Optional[int] = None, isb: typing.Optional[int] = None, lambda_: typing.Optional[int] = None, chroma: typing.Optional[int] = None, delta: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, lsad: typing.Optional[int] = None, plevel: typing.Optional[int] = None, global_: typing.Optional[int] = None, pnew: typing.Optional[int] = None, pzero: typing.Optional[int] = None, pglobal: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, badsad: typing.Optional[int] = None, badrange: typing.Optional[int] = None, opt: typing.Optional[int] = None, meander: typing.Optional[int] = None, trymany: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, search_coarse: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def BlockFPS(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mode: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Compensate(self, super: "VideoNode", vectors: "VideoNode", scbehavior: typing.Optional[int] = None, thsad: typing.Optional[int] = None, fields: typing.Optional[int] = None, time: typing.Optional[float] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain1(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", thsad: typing.Optional[int] = None, thsadc: typing.Optional[int] = None, plane: typing.Optional[int] = None, limit: typing.Optional[int] = None, limitc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain2(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", thsad: typing.Optional[int] = None, thsadc: typing.Optional[int] = None, plane: typing.Optional[int] = None, limit: typing.Optional[int] = None, limitc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain3(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", thsad: typing.Optional[int] = None, thsadc: typing.Optional[int] = None, plane: typing.Optional[int] = None, limit: typing.Optional[int] = None, limitc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanAnalyse(self, vectors: "VideoNode", mask: typing.Optional["VideoNode"] = None, zoom: typing.Optional[int] = None, rot: typing.Optional[int] = None, pixaspect: typing.Optional[float] = None, error: typing.Optional[float] = None, info: typing.Optional[int] = None, wrong: typing.Optional[float] = None, zerow: typing.Optional[float] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanCompensate(self, data: "VideoNode", offset: typing.Optional[float] = None, subpixel: typing.Optional[int] = None, pixaspect: typing.Optional[float] = None, matchfields: typing.Optional[int] = None, mirror: typing.Optional[int] = None, blur: typing.Optional[int] = None, info: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanEstimate(self, trust: typing.Optional[float] = None, winx: typing.Optional[int] = None, winy: typing.Optional[int] = None, wleft: typing.Optional[int] = None, wtop: typing.Optional[int] = None, dxmax: typing.Optional[int] = None, dymax: typing.Optional[int] = None, zoommax: typing.Optional[float] = None, stab: typing.Optional[float] = None, pixaspect: typing.Optional[float] = None, info: typing.Optional[int] = None, show: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DepanStabilise(self, data: "VideoNode", cutoff: typing.Optional[float] = None, damping: typing.Optional[float] = None, initzoom: typing.Optional[float] = None, addzoom: typing.Optional[int] = None, prev: typing.Optional[int] = None, next: typing.Optional[int] = None, mirror: typing.Optional[int] = None, blur: typing.Optional[int] = None, dxmax: typing.Optional[float] = None, dymax: typing.Optional[float] = None, zoommax: typing.Optional[float] = None, rotmax: typing.Optional[float] = None, subpixel: typing.Optional[int] = None, pixaspect: typing.Optional[float] = None, fitlast: typing.Optional[int] = None, tzoom: typing.Optional[float] = None, info: typing.Optional[int] = None, method: typing.Optional[int] = None, fields: typing.Optional[int] = None) -> "VideoNode": ...
    def Finest(self, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Flow(self, super: "VideoNode", vectors: "VideoNode", time: typing.Optional[float] = None, mode: typing.Optional[int] = None, fields: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowBlur(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", blur: typing.Optional[float] = None, prec: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowFPS(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mask: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowInter(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", time: typing.Optional[float] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Mask(self, vectors: "VideoNode", ml: typing.Optional[float] = None, gamma: typing.Optional[float] = None, kind: typing.Optional[int] = None, time: typing.Optional[float] = None, ysc: typing.Optional[int] = None, thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def Recalculate(self, vectors: "VideoNode", thsad: typing.Optional[int] = None, smooth: typing.Optional[int] = None, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, lambda_: typing.Optional[int] = None, chroma: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, pnew: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, opt: typing.Optional[int] = None, meander: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def SCDetection(self, vectors: "VideoNode", thscd1: typing.Optional[int] = None, thscd2: typing.Optional[int] = None) -> "VideoNode": ...
    def Super(self, hpad: typing.Optional[int] = None, vpad: typing.Optional[int] = None, pel: typing.Optional[int] = None, levels: typing.Optional[int] = None, chroma: typing.Optional[int] = None, sharp: typing.Optional[int] = None, rfilter: typing.Optional[int] = None, pelclip: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: mvsf
class _Plugin_mvsf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Analyse(self, super: "VideoNode", blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, levels: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, pelsearch: typing.Optional[int] = None, isb: typing.Optional[int] = None, lambda_: typing.Optional[float] = None, chroma: typing.Optional[int] = None, delta: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, lsad: typing.Optional[float] = None, plevel: typing.Optional[int] = None, global_: typing.Optional[int] = None, pnew: typing.Optional[int] = None, pzero: typing.Optional[int] = None, pglobal: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, badsad: typing.Optional[float] = None, badrange: typing.Optional[int] = None, meander: typing.Optional[int] = None, trymany: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, search_coarse: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def Analyze(self, super: "VideoNode", blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, levels: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, pelsearch: typing.Optional[int] = None, isb: typing.Optional[int] = None, lambda_: typing.Optional[float] = None, chroma: typing.Optional[int] = None, delta: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, lsad: typing.Optional[float] = None, plevel: typing.Optional[int] = None, global_: typing.Optional[int] = None, pnew: typing.Optional[int] = None, pzero: typing.Optional[int] = None, pglobal: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, badsad: typing.Optional[float] = None, badrange: typing.Optional[int] = None, meander: typing.Optional[int] = None, trymany: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, search_coarse: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def BlockFPS(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mode: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Compensate(self, clip: "VideoNode", super: "VideoNode", vectors: "VideoNode", scbehavior: typing.Optional[int] = None, thsad: typing.Optional[float] = None, fields: typing.Optional[int] = None, time: typing.Optional[float] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain1(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain10(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain11(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain12(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain13(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain14(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain15(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain16(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain17(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain18(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain19(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain2(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain20(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain21(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain22(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", mvbw22: "VideoNode", mvfw22: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain23(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", mvbw22: "VideoNode", mvfw22: "VideoNode", mvbw23: "VideoNode", mvfw23: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain24(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", mvbw22: "VideoNode", mvfw22: "VideoNode", mvbw23: "VideoNode", mvfw23: "VideoNode", mvbw24: "VideoNode", mvfw24: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain3(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain4(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain5(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain6(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain7(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain8(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain9(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Finest(self, super: "VideoNode") -> "VideoNode": ...
    def Flow(self, clip: "VideoNode", super: "VideoNode", vectors: "VideoNode", time: typing.Optional[float] = None, mode: typing.Optional[int] = None, fields: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowBlur(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", blur: typing.Optional[float] = None, prec: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def FlowFPS(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mask: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def FlowInter(self, clip: "VideoNode", super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", time: typing.Optional[float] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Mask(self, clip: "VideoNode", vectors: "VideoNode", ml: typing.Optional[float] = None, gamma: typing.Optional[float] = None, kind: typing.Optional[int] = None, time: typing.Optional[float] = None, ysc: typing.Optional[float] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Recalculate(self, super: "VideoNode", vectors: "VideoNode", thsad: typing.Optional[float] = None, smooth: typing.Optional[int] = None, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, lambda_: typing.Optional[float] = None, chroma: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, pnew: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, meander: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def SCDetection(self, clip: "VideoNode", vectors: "VideoNode", thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Super(self, clip: "VideoNode", hpad: typing.Optional[int] = None, vpad: typing.Optional[int] = None, pel: typing.Optional[int] = None, levels: typing.Optional[int] = None, chroma: typing.Optional[int] = None, sharp: typing.Optional[int] = None, rfilter: typing.Optional[int] = None, pelclip: typing.Optional["VideoNode"] = None) -> "VideoNode": ...


class _Plugin_mvsf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Analyse(self, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, levels: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, pelsearch: typing.Optional[int] = None, isb: typing.Optional[int] = None, lambda_: typing.Optional[float] = None, chroma: typing.Optional[int] = None, delta: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, lsad: typing.Optional[float] = None, plevel: typing.Optional[int] = None, global_: typing.Optional[int] = None, pnew: typing.Optional[int] = None, pzero: typing.Optional[int] = None, pglobal: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, badsad: typing.Optional[float] = None, badrange: typing.Optional[int] = None, meander: typing.Optional[int] = None, trymany: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, search_coarse: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def Analyze(self, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, levels: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, pelsearch: typing.Optional[int] = None, isb: typing.Optional[int] = None, lambda_: typing.Optional[float] = None, chroma: typing.Optional[int] = None, delta: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, lsad: typing.Optional[float] = None, plevel: typing.Optional[int] = None, global_: typing.Optional[int] = None, pnew: typing.Optional[int] = None, pzero: typing.Optional[int] = None, pglobal: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, badsad: typing.Optional[float] = None, badrange: typing.Optional[int] = None, meander: typing.Optional[int] = None, trymany: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, search_coarse: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def BlockFPS(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mode: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Compensate(self, super: "VideoNode", vectors: "VideoNode", scbehavior: typing.Optional[int] = None, thsad: typing.Optional[float] = None, fields: typing.Optional[int] = None, time: typing.Optional[float] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def Degrain1(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain10(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain11(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain12(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain13(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain14(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain15(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain16(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain17(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain18(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain19(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain2(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain20(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain21(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain22(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", mvbw22: "VideoNode", mvfw22: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain23(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", mvbw22: "VideoNode", mvfw22: "VideoNode", mvbw23: "VideoNode", mvfw23: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain24(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", mvbw10: "VideoNode", mvfw10: "VideoNode", mvbw11: "VideoNode", mvfw11: "VideoNode", mvbw12: "VideoNode", mvfw12: "VideoNode", mvbw13: "VideoNode", mvfw13: "VideoNode", mvbw14: "VideoNode", mvfw14: "VideoNode", mvbw15: "VideoNode", mvfw15: "VideoNode", mvbw16: "VideoNode", mvfw16: "VideoNode", mvbw17: "VideoNode", mvfw17: "VideoNode", mvbw18: "VideoNode", mvfw18: "VideoNode", mvbw19: "VideoNode", mvfw19: "VideoNode", mvbw20: "VideoNode", mvfw20: "VideoNode", mvbw21: "VideoNode", mvfw21: "VideoNode", mvbw22: "VideoNode", mvfw22: "VideoNode", mvbw23: "VideoNode", mvfw23: "VideoNode", mvbw24: "VideoNode", mvfw24: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain3(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain4(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain5(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain6(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain7(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain8(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Degrain9(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", mvbw2: "VideoNode", mvfw2: "VideoNode", mvbw3: "VideoNode", mvfw3: "VideoNode", mvbw4: "VideoNode", mvfw4: "VideoNode", mvbw5: "VideoNode", mvfw5: "VideoNode", mvbw6: "VideoNode", mvfw6: "VideoNode", mvbw7: "VideoNode", mvfw7: "VideoNode", mvbw8: "VideoNode", mvfw8: "VideoNode", mvbw9: "VideoNode", mvfw9: "VideoNode", thsad: typing.Union[float, typing.Sequence[float], None] = None, plane: typing.Optional[int] = None, limit: typing.Union[float, typing.Sequence[float], None] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Finest(self) -> "VideoNode": ...
    def Flow(self, super: "VideoNode", vectors: "VideoNode", time: typing.Optional[float] = None, mode: typing.Optional[int] = None, fields: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def FlowBlur(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", blur: typing.Optional[float] = None, prec: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def FlowFPS(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", num: typing.Optional[int] = None, den: typing.Optional[int] = None, mask: typing.Optional[int] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def FlowInter(self, super: "VideoNode", mvbw: "VideoNode", mvfw: "VideoNode", time: typing.Optional[float] = None, ml: typing.Optional[float] = None, blend: typing.Optional[int] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Mask(self, vectors: "VideoNode", ml: typing.Optional[float] = None, gamma: typing.Optional[float] = None, kind: typing.Optional[int] = None, time: typing.Optional[float] = None, ysc: typing.Optional[float] = None, thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Recalculate(self, vectors: "VideoNode", thsad: typing.Optional[float] = None, smooth: typing.Optional[int] = None, blksize: typing.Optional[int] = None, blksizev: typing.Optional[int] = None, search: typing.Optional[int] = None, searchparam: typing.Optional[int] = None, lambda_: typing.Optional[float] = None, chroma: typing.Optional[int] = None, truemotion: typing.Optional[int] = None, pnew: typing.Optional[int] = None, overlap: typing.Optional[int] = None, overlapv: typing.Optional[int] = None, divide: typing.Optional[int] = None, meander: typing.Optional[int] = None, fields: typing.Optional[int] = None, tff: typing.Optional[int] = None, dct: typing.Optional[int] = None) -> "VideoNode": ...
    def SCDetection(self, vectors: "VideoNode", thscd1: typing.Optional[float] = None, thscd2: typing.Optional[float] = None) -> "VideoNode": ...
    def Super(self, hpad: typing.Optional[int] = None, vpad: typing.Optional[int] = None, pel: typing.Optional[int] = None, levels: typing.Optional[int] = None, chroma: typing.Optional[int] = None, sharp: typing.Optional[int] = None, rfilter: typing.Optional[int] = None, pelclip: typing.Optional["VideoNode"] = None) -> "VideoNode": ...
# end implementation


# implementation: neo_f3kdb
class _Plugin_neo_f3kdb_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deband(self, clip: "VideoNode", range: typing.Optional[int] = None, y: typing.Optional[int] = None, cb: typing.Optional[int] = None, cr: typing.Optional[int] = None, grainy: typing.Optional[int] = None, grainc: typing.Optional[int] = None, sample_mode: typing.Optional[int] = None, seed: typing.Optional[int] = None, blur_first: typing.Optional[int] = None, dynamic_grain: typing.Optional[int] = None, opt: typing.Optional[int] = None, dither_algo: typing.Optional[int] = None, keep_tv_range: typing.Optional[int] = None, output_depth: typing.Optional[int] = None, random_algo_ref: typing.Optional[int] = None, random_algo_grain: typing.Optional[int] = None, random_param_ref: typing.Optional[float] = None, random_param_grain: typing.Optional[float] = None, preset: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_neo_f3kdb_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deband(self, range: typing.Optional[int] = None, y: typing.Optional[int] = None, cb: typing.Optional[int] = None, cr: typing.Optional[int] = None, grainy: typing.Optional[int] = None, grainc: typing.Optional[int] = None, sample_mode: typing.Optional[int] = None, seed: typing.Optional[int] = None, blur_first: typing.Optional[int] = None, dynamic_grain: typing.Optional[int] = None, opt: typing.Optional[int] = None, dither_algo: typing.Optional[int] = None, keep_tv_range: typing.Optional[int] = None, output_depth: typing.Optional[int] = None, random_algo_ref: typing.Optional[int] = None, random_algo_grain: typing.Optional[int] = None, random_param_ref: typing.Optional[float] = None, random_param_grain: typing.Optional[float] = None, preset: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: nnedi3
class _Plugin_nnedi3_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def nnedi3(self, clip: "VideoNode", field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, nsize: typing.Optional[int] = None, nns: typing.Optional[int] = None, qual: typing.Optional[int] = None, etype: typing.Optional[int] = None, pscrn: typing.Optional[int] = None, opt: typing.Optional[int] = None, int16_prescreener: typing.Optional[int] = None, int16_predictor: typing.Optional[int] = None, exp: typing.Optional[int] = None, show_mask: typing.Optional[int] = None, combed_only: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_nnedi3_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def nnedi3(self, field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, nsize: typing.Optional[int] = None, nns: typing.Optional[int] = None, qual: typing.Optional[int] = None, etype: typing.Optional[int] = None, pscrn: typing.Optional[int] = None, opt: typing.Optional[int] = None, int16_prescreener: typing.Optional[int] = None, int16_predictor: typing.Optional[int] = None, exp: typing.Optional[int] = None, show_mask: typing.Optional[int] = None, combed_only: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: nnedi3cl
class _Plugin_nnedi3cl_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def NNEDI3CL(self, clip: "VideoNode", field: int, dh: typing.Optional[int] = None, dw: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, nsize: typing.Optional[int] = None, nns: typing.Optional[int] = None, qual: typing.Optional[int] = None, etype: typing.Optional[int] = None, pscrn: typing.Optional[int] = None, device: typing.Optional[int] = None, list_device: typing.Optional[int] = None, info: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_nnedi3cl_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def NNEDI3CL(self, field: int, dh: typing.Optional[int] = None, dw: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, nsize: typing.Optional[int] = None, nns: typing.Optional[int] = None, qual: typing.Optional[int] = None, etype: typing.Optional[int] = None, pscrn: typing.Optional[int] = None, device: typing.Optional[int] = None, list_device: typing.Optional[int] = None, info: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: noisegen
class _Plugin_noisegen_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Generate(self, clip: "VideoNode", str: typing.Optional[float] = None, limit: typing.Optional[float] = None, type: typing.Optional[int] = None, mean: typing.Optional[float] = None, var: typing.Optional[float] = None, dyn: typing.Optional[int] = None, full: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_noisegen_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Generate(self, str: typing.Optional[float] = None, limit: typing.Optional[float] = None, type: typing.Optional[int] = None, mean: typing.Optional[float] = None, var: typing.Optional[float] = None, dyn: typing.Optional[int] = None, full: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: placebo
class _Plugin_placebo_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deband(self, clip: "VideoNode", planes: typing.Optional[int] = None, iterations: typing.Optional[int] = None, threshold: typing.Optional[float] = None, radius: typing.Optional[float] = None, grain: typing.Optional[float] = None, dither: typing.Optional[int] = None, dither_algo: typing.Optional[int] = None, renderer_api: typing.Optional[int] = None) -> "VideoNode": ...
    def Resample(self, clip: "VideoNode", width: int, height: int, filter: typing.Union[str, bytes, bytearray, None] = None, clamp: typing.Optional[float] = None, blur: typing.Optional[float] = None, taper: typing.Optional[float] = None, radius: typing.Optional[float] = None, param1: typing.Optional[float] = None, param2: typing.Optional[float] = None, sx: typing.Optional[float] = None, sy: typing.Optional[float] = None, antiring: typing.Optional[float] = None, lut_entries: typing.Optional[int] = None, cutoff: typing.Optional[float] = None, sigmoidize: typing.Optional[int] = None, sigmoid_center: typing.Optional[float] = None, sigmoid_slope: typing.Optional[float] = None, linearize: typing.Optional[int] = None, trc: typing.Optional[int] = None) -> "VideoNode": ...
    def Shader(self, clip: "VideoNode", shader: typing.Union[str, bytes, bytearray], width: typing.Optional[int] = None, height: typing.Optional[int] = None, chroma_loc: typing.Optional[int] = None, matrix: typing.Optional[int] = None, trc: typing.Optional[int] = None, linearize: typing.Optional[int] = None, sigmoidize: typing.Optional[int] = None, sigmoid_center: typing.Optional[float] = None, sigmoid_slope: typing.Optional[float] = None, lut_entries: typing.Optional[int] = None, antiring: typing.Optional[float] = None, filter: typing.Union[str, bytes, bytearray, None] = None, clamp: typing.Optional[float] = None, blur: typing.Optional[float] = None, taper: typing.Optional[float] = None, radius: typing.Optional[float] = None, param1: typing.Optional[float] = None, param2: typing.Optional[float] = None) -> "VideoNode": ...
    def Tonemap(self, clip: "VideoNode", srcp: typing.Optional[int] = None, srct: typing.Optional[int] = None, srcl: typing.Optional[int] = None, src_peak: typing.Optional[float] = None, src_avg: typing.Optional[float] = None, src_scale: typing.Optional[float] = None, dstp: typing.Optional[int] = None, dstt: typing.Optional[int] = None, dstl: typing.Optional[int] = None, dst_peak: typing.Optional[float] = None, dst_avg: typing.Optional[float] = None, dst_scale: typing.Optional[float] = None, dynamic_peak_detection: typing.Optional[int] = None, smoothing_period: typing.Optional[float] = None, scene_threshold_low: typing.Optional[float] = None, scene_threshold_high: typing.Optional[float] = None, intent: typing.Optional[int] = None, tone_mapping_algo: typing.Optional[int] = None, tone_mapping_param: typing.Optional[float] = None, desaturation_strength: typing.Optional[float] = None, desaturation_exponent: typing.Optional[float] = None, desaturation_base: typing.Optional[float] = None, max_boost: typing.Optional[float] = None, gamut_warning: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_placebo_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Deband(self, planes: typing.Optional[int] = None, iterations: typing.Optional[int] = None, threshold: typing.Optional[float] = None, radius: typing.Optional[float] = None, grain: typing.Optional[float] = None, dither: typing.Optional[int] = None, dither_algo: typing.Optional[int] = None, renderer_api: typing.Optional[int] = None) -> "VideoNode": ...
    def Resample(self, width: int, height: int, filter: typing.Union[str, bytes, bytearray, None] = None, clamp: typing.Optional[float] = None, blur: typing.Optional[float] = None, taper: typing.Optional[float] = None, radius: typing.Optional[float] = None, param1: typing.Optional[float] = None, param2: typing.Optional[float] = None, sx: typing.Optional[float] = None, sy: typing.Optional[float] = None, antiring: typing.Optional[float] = None, lut_entries: typing.Optional[int] = None, cutoff: typing.Optional[float] = None, sigmoidize: typing.Optional[int] = None, sigmoid_center: typing.Optional[float] = None, sigmoid_slope: typing.Optional[float] = None, linearize: typing.Optional[int] = None, trc: typing.Optional[int] = None) -> "VideoNode": ...
    def Shader(self, shader: typing.Union[str, bytes, bytearray], width: typing.Optional[int] = None, height: typing.Optional[int] = None, chroma_loc: typing.Optional[int] = None, matrix: typing.Optional[int] = None, trc: typing.Optional[int] = None, linearize: typing.Optional[int] = None, sigmoidize: typing.Optional[int] = None, sigmoid_center: typing.Optional[float] = None, sigmoid_slope: typing.Optional[float] = None, lut_entries: typing.Optional[int] = None, antiring: typing.Optional[float] = None, filter: typing.Union[str, bytes, bytearray, None] = None, clamp: typing.Optional[float] = None, blur: typing.Optional[float] = None, taper: typing.Optional[float] = None, radius: typing.Optional[float] = None, param1: typing.Optional[float] = None, param2: typing.Optional[float] = None) -> "VideoNode": ...
    def Tonemap(self, srcp: typing.Optional[int] = None, srct: typing.Optional[int] = None, srcl: typing.Optional[int] = None, src_peak: typing.Optional[float] = None, src_avg: typing.Optional[float] = None, src_scale: typing.Optional[float] = None, dstp: typing.Optional[int] = None, dstt: typing.Optional[int] = None, dstl: typing.Optional[int] = None, dst_peak: typing.Optional[float] = None, dst_avg: typing.Optional[float] = None, dst_scale: typing.Optional[float] = None, dynamic_peak_detection: typing.Optional[int] = None, smoothing_period: typing.Optional[float] = None, scene_threshold_low: typing.Optional[float] = None, scene_threshold_high: typing.Optional[float] = None, intent: typing.Optional[int] = None, tone_mapping_algo: typing.Optional[int] = None, tone_mapping_param: typing.Optional[float] = None, desaturation_strength: typing.Optional[float] = None, desaturation_exponent: typing.Optional[float] = None, desaturation_base: typing.Optional[float] = None, max_boost: typing.Optional[float] = None, gamut_warning: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: pp7
class _Plugin_pp7_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DeblockPP7(self, clip: "VideoNode", qp: typing.Optional[float] = None, mode: typing.Optional[int] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_pp7_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DeblockPP7(self, qp: typing.Optional[float] = None, mode: typing.Optional[int] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: qr
class _Plugin_qr_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Code(self, message: typing.Union[str, bytes, bytearray], version: typing.Optional[int] = None, error_correction: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_qr_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Code(self, version: typing.Optional[int] = None, error_correction: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: raws
class _Plugin_raws_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Source(self, source: typing.Union[str, bytes, bytearray], width: typing.Optional[int] = None, height: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, sarnum: typing.Optional[int] = None, sarden: typing.Optional[int] = None, src_fmt: typing.Union[str, bytes, bytearray, None] = None, off_header: typing.Optional[int] = None, off_frame: typing.Optional[int] = None, rowbytes_align: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_raws_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Source(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, sarnum: typing.Optional[int] = None, sarden: typing.Optional[int] = None, src_fmt: typing.Union[str, bytes, bytearray, None] = None, off_header: typing.Optional[int] = None, off_frame: typing.Optional[int] = None, rowbytes_align: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: rdvs
class _Plugin_rdvs_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DupBlocks(self, input: "VideoNode", gmthreshold: typing.Optional[int] = None, mthreshold: typing.Optional[int] = None, noise: typing.Optional[int] = None, noisy: typing.Optional[int] = None, dist: typing.Optional[int] = None, tolerance: typing.Optional[int] = None, dmode: typing.Optional[int] = None, pthreshold: typing.Optional[int] = None, cthreshold: typing.Optional[int] = None, grey: typing.Optional[int] = None) -> "VideoNode": ...
    def RestoreMotionBlocks(self, input: "VideoNode", restore: "VideoNode", neighbour: typing.Optional["VideoNode"] = None, neighbour2: typing.Optional["VideoNode"] = None, alternative: typing.Optional["VideoNode"] = None, gmthreshold: typing.Optional[int] = None, mthreshold: typing.Optional[int] = None, noise: typing.Optional[int] = None, noisy: typing.Optional[int] = None, dist: typing.Optional[int] = None, tolerance: typing.Optional[int] = None, dmode: typing.Optional[int] = None, pthreshold: typing.Optional[int] = None, cthreshold: typing.Optional[int] = None, grey: typing.Optional[int] = None) -> "VideoNode": ...
    def SCSelect(self, input: "VideoNode", sceneBegin: "VideoNode", sceneEnd: "VideoNode", globalMotion: "VideoNode", dfactor: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_rdvs_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DupBlocks(self, gmthreshold: typing.Optional[int] = None, mthreshold: typing.Optional[int] = None, noise: typing.Optional[int] = None, noisy: typing.Optional[int] = None, dist: typing.Optional[int] = None, tolerance: typing.Optional[int] = None, dmode: typing.Optional[int] = None, pthreshold: typing.Optional[int] = None, cthreshold: typing.Optional[int] = None, grey: typing.Optional[int] = None) -> "VideoNode": ...
    def RestoreMotionBlocks(self, restore: "VideoNode", neighbour: typing.Optional["VideoNode"] = None, neighbour2: typing.Optional["VideoNode"] = None, alternative: typing.Optional["VideoNode"] = None, gmthreshold: typing.Optional[int] = None, mthreshold: typing.Optional[int] = None, noise: typing.Optional[int] = None, noisy: typing.Optional[int] = None, dist: typing.Optional[int] = None, tolerance: typing.Optional[int] = None, dmode: typing.Optional[int] = None, pthreshold: typing.Optional[int] = None, cthreshold: typing.Optional[int] = None, grey: typing.Optional[int] = None) -> "VideoNode": ...
    def SCSelect(self, sceneBegin: "VideoNode", sceneEnd: "VideoNode", globalMotion: "VideoNode", dfactor: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: reduceflicker
class _Plugin_reduceflicker_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ReduceFlicker(self, clip: "VideoNode", strength: typing.Optional[int] = None, aggressive: typing.Optional[int] = None, grey: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_reduceflicker_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ReduceFlicker(self, strength: typing.Optional[int] = None, aggressive: typing.Optional[int] = None, grey: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: resize
class _Plugin_resize_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Bicubic(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Bilinear(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Lanczos(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Point(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Spline16(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Spline36(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Spline64(self, clip: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_resize_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Bicubic(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Bilinear(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Lanczos(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Point(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Spline16(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Spline36(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
    def Spline64(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None, range: typing.Optional[int] = None, range_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc: typing.Optional[int] = None, chromaloc_s: typing.Union[str, bytes, bytearray, None] = None, matrix_in: typing.Optional[int] = None, matrix_in_s: typing.Union[str, bytes, bytearray, None] = None, transfer_in: typing.Optional[int] = None, transfer_in_s: typing.Union[str, bytes, bytearray, None] = None, primaries_in: typing.Optional[int] = None, primaries_in_s: typing.Union[str, bytes, bytearray, None] = None, range_in: typing.Optional[int] = None, range_in_s: typing.Union[str, bytes, bytearray, None] = None, chromaloc_in: typing.Optional[int] = None, chromaloc_in_s: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, resample_filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, dither_type: typing.Union[str, bytes, bytearray, None] = None, cpu_type: typing.Union[str, bytes, bytearray, None] = None, prefer_props: typing.Optional[int] = None, src_left: typing.Optional[float] = None, src_top: typing.Optional[float] = None, src_width: typing.Optional[float] = None, src_height: typing.Optional[float] = None, nominal_luminance: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: retinex
class _Plugin_retinex_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MSRCP(self, input: "VideoNode", sigma: typing.Union[float, typing.Sequence[float], None] = None, lower_thr: typing.Optional[float] = None, upper_thr: typing.Optional[float] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, chroma_protect: typing.Optional[float] = None) -> "VideoNode": ...
    def MSRCR(self, input: "VideoNode", sigma: typing.Union[float, typing.Sequence[float], None] = None, lower_thr: typing.Optional[float] = None, upper_thr: typing.Optional[float] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, restore: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_retinex_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def MSRCP(self, sigma: typing.Union[float, typing.Sequence[float], None] = None, lower_thr: typing.Optional[float] = None, upper_thr: typing.Optional[float] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, chroma_protect: typing.Optional[float] = None) -> "VideoNode": ...
    def MSRCR(self, sigma: typing.Union[float, typing.Sequence[float], None] = None, lower_thr: typing.Optional[float] = None, upper_thr: typing.Optional[float] = None, fulls: typing.Optional[int] = None, fulld: typing.Optional[int] = None, restore: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: rf
class _Plugin_rf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Replace(self, clip: "VideoNode", clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], intervals: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]]) -> "VideoNode": ...


class _Plugin_rf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Replace(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], intervals: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]]) -> "VideoNode": ...
# end implementation


# implementation: rgsf
class _Plugin_rgsf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def BackwardClense(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Clense(self, clip: "VideoNode", previous: typing.Optional["VideoNode"] = None, next: typing.Optional["VideoNode"] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def ForwardClense(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def RemoveGrain(self, clip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Repair(self, clip: "VideoNode", repairclip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def VerticalCleaner(self, clip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...


class _Plugin_rgsf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def BackwardClense(self, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Clense(self, previous: typing.Optional["VideoNode"] = None, next: typing.Optional["VideoNode"] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def ForwardClense(self, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def RemoveGrain(self, mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Repair(self, repairclip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def VerticalCleaner(self, mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
# end implementation


# implementation: rgvs
class _Plugin_rgvs_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def BackwardClense(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Clense(self, clip: "VideoNode", previous: typing.Optional["VideoNode"] = None, next: typing.Optional["VideoNode"] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def ForwardClense(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def RemoveGrain(self, clip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Repair(self, clip: "VideoNode", repairclip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def VerticalCleaner(self, clip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...


class _Plugin_rgvs_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def BackwardClense(self, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Clense(self, previous: typing.Optional["VideoNode"] = None, next: typing.Optional["VideoNode"] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def ForwardClense(self, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def RemoveGrain(self, mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Repair(self, repairclip: "VideoNode", mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def VerticalCleaner(self, mode: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
# end implementation


# implementation: sangnom
class _Plugin_sangnom_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SangNom(self, clip: "VideoNode", order: typing.Optional[int] = None, dh: typing.Optional[int] = None, aa: typing.Union[int, typing.Sequence[int], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_sangnom_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SangNom(self, order: typing.Optional[int] = None, dh: typing.Optional[int] = None, aa: typing.Union[int, typing.Sequence[int], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: scd
class _Plugin_scd_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ApplyLog(self, clip: "VideoNode", log: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def Detect(self, clip: "VideoNode", thresh: typing.Optional[int] = None, interval_h: typing.Optional[int] = None, interval_v: typing.Optional[int] = None, log: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_scd_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ApplyLog(self, log: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def Detect(self, thresh: typing.Optional[int] = None, interval_h: typing.Optional[int] = None, interval_v: typing.Optional[int] = None, log: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: scrawl
class _Plugin_scrawl_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ClipInfo(self, clip: "VideoNode", alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def CoreInfo(self, clip: "VideoNode", alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameNum(self, clip: "VideoNode", alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameProps(self, clip: "VideoNode", alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def Text(self, clip: "VideoNode", text: typing.Union[str, bytes, bytearray], alignment: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_scrawl_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ClipInfo(self, alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def CoreInfo(self, alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameNum(self, alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameProps(self, alignment: typing.Optional[int] = None) -> "VideoNode": ...
    def Text(self, text: typing.Union[str, bytes, bytearray], alignment: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: scxvid
class _Plugin_scxvid_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Scxvid(self, clip: "VideoNode", log: typing.Union[str, bytes, bytearray, None] = None, use_slices: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_scxvid_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Scxvid(self, log: typing.Union[str, bytes, bytearray, None] = None, use_slices: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: smoothuv
class _Plugin_smoothuv_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SmoothUV(self, clip: "VideoNode", radius: typing.Optional[int] = None, threshold: typing.Optional[int] = None, interlaced: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_smoothuv_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SmoothUV(self, radius: typing.Optional[int] = None, threshold: typing.Optional[int] = None, interlaced: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: ssiq
class _Plugin_ssiq_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SSIQ(self, clip: "VideoNode", diameter: typing.Optional[int] = None, strength: typing.Optional[int] = None, interlaced: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_ssiq_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SSIQ(self, diameter: typing.Optional[int] = None, strength: typing.Optional[int] = None, interlaced: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: std
class _Plugin_std_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AddBorders(self, clip: "VideoNode", left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, color: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
    def AssumeFPS(self, clip: "VideoNode", src: typing.Optional["VideoNode"] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None) -> "VideoNode": ...
    def Binarize(self, clip: "VideoNode", threshold: typing.Union[float, typing.Sequence[float], None] = None, v0: typing.Union[float, typing.Sequence[float], None] = None, v1: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def BlankClip(self, clip: typing.Optional["VideoNode"] = None, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, length: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, color: typing.Union[float, typing.Sequence[float], None] = None, keep: typing.Optional[int] = None) -> "VideoNode": ...
    def BoxBlur(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, hradius: typing.Optional[int] = None, hpasses: typing.Optional[int] = None, vradius: typing.Optional[int] = None, vpasses: typing.Optional[int] = None) -> "VideoNode": ...
    def Cache(self, clip: "VideoNode", size: typing.Optional[int] = None, fixed: typing.Optional[int] = None, make_linear: typing.Optional[int] = None) -> "VideoNode": ...
    def ClipToProp(self, clip: "VideoNode", mclip: "VideoNode", prop: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Convolution(self, clip: "VideoNode", matrix: typing.Union[float, typing.Sequence[float]], bias: typing.Optional[float] = None, divisor: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, saturate: typing.Optional[int] = None, mode: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Crop(self, clip: "VideoNode", left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None) -> "VideoNode": ...
    def CropAbs(self, clip: "VideoNode", width: int, height: int, left: typing.Optional[int] = None, top: typing.Optional[int] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None) -> "VideoNode": ...
    def CropRel(self, clip: "VideoNode", left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None) -> "VideoNode": ...
    def Deflate(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None) -> "VideoNode": ...
    def DeleteFrames(self, clip: "VideoNode", frames: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def DoubleWeave(self, clip: "VideoNode", tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DuplicateFrames(self, clip: "VideoNode", frames: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Expr(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], expr: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], format: typing.Optional[int] = None) -> "VideoNode": ...
    def FlipHorizontal(self, clip: "VideoNode") -> "VideoNode": ...
    def FlipVertical(self, clip: "VideoNode") -> "VideoNode": ...
    def FrameEval(self, clip: "VideoNode", eval: typing.Callable[..., typing.Any], prop_src: typing.Union["VideoNode", typing.Sequence["VideoNode"], None] = None) -> "VideoNode": ...
    def FreezeFrames(self, clip: "VideoNode", first: typing.Union[int, typing.Sequence[int]], last: typing.Union[int, typing.Sequence[int]], replacement: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Inflate(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None) -> "VideoNode": ...
    def Interleave(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], extend: typing.Optional[int] = None, mismatch: typing.Optional[int] = None, modify_duration: typing.Optional[int] = None) -> "VideoNode": ...
    def Invert(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Levels(self, clip: "VideoNode", min_in: typing.Union[float, typing.Sequence[float], None] = None, max_in: typing.Union[float, typing.Sequence[float], None] = None, gamma: typing.Union[float, typing.Sequence[float], None] = None, min_out: typing.Union[float, typing.Sequence[float], None] = None, max_out: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Limiter(self, clip: "VideoNode", min: typing.Union[float, typing.Sequence[float], None] = None, max: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def LoadPlugin(self, path: typing.Union[str, bytes, bytearray], altsearchpath: typing.Optional[int] = None, forcens: typing.Union[str, bytes, bytearray, None] = None, forceid: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Loop(self, clip: "VideoNode", times: typing.Optional[int] = None) -> "VideoNode": ...
    def Lut(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, lut: typing.Union[int, typing.Sequence[int], None] = None, lutf: typing.Union[float, typing.Sequence[float], None] = None, function: typing.Optional[typing.Callable[..., typing.Any]] = None, bits: typing.Optional[int] = None, floatout: typing.Optional[int] = None) -> "VideoNode": ...
    def Lut2(self, clipa: "VideoNode", clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, lut: typing.Union[int, typing.Sequence[int], None] = None, lutf: typing.Union[float, typing.Sequence[float], None] = None, function: typing.Optional[typing.Callable[..., typing.Any]] = None, bits: typing.Optional[int] = None, floatout: typing.Optional[int] = None) -> "VideoNode": ...
    def MakeDiff(self, clipa: "VideoNode", clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def MaskedMerge(self, clipa: "VideoNode", clipb: "VideoNode", mask: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, first_plane: typing.Optional[int] = None, premultiplied: typing.Optional[int] = None) -> "VideoNode": ...
    def Maximum(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None, coordinates: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Median(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Merge(self, clipa: "VideoNode", clipb: "VideoNode", weight: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
    def MergeDiff(self, clipa: "VideoNode", clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Minimum(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None, coordinates: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def ModifyFrame(self, clip: "VideoNode", clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], selector: typing.Callable[..., typing.Any]) -> "VideoNode": ...
    def PEMVerifier(self, clip: "VideoNode", upper: typing.Union[float, typing.Sequence[float], None] = None, lower: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
    def PlaneStats(self, clipa: "VideoNode", clipb: typing.Optional["VideoNode"] = None, plane: typing.Optional[int] = None, prop: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def PreMultiply(self, clip: "VideoNode", alpha: "VideoNode") -> "VideoNode": ...
    def Prewitt(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, scale: typing.Optional[float] = None) -> "VideoNode": ...
    def PropToClip(self, clip: "VideoNode", prop: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Reverse(self, clip: "VideoNode") -> "VideoNode": ...
    def SelectEvery(self, clip: "VideoNode", cycle: int, offsets: typing.Union[int, typing.Sequence[int]], modify_duration: typing.Optional[int] = None) -> "VideoNode": ...
    def SeparateFields(self, clip: "VideoNode", tff: typing.Optional[int] = None, modify_duration: typing.Optional[int] = None) -> "VideoNode": ...
    def SetFieldBased(self, clip: "VideoNode", value: int) -> "VideoNode": ...
    def SetFrameProp(self, clip: "VideoNode", prop: typing.Union[str, bytes, bytearray], delete: typing.Optional[int] = None, intval: typing.Union[int, typing.Sequence[int], None] = None, floatval: typing.Union[float, typing.Sequence[float], None] = None, data: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None) -> "VideoNode": ...
    def SetMaxCPU(self, cpu: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def ShufflePlanes(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], planes: typing.Union[int, typing.Sequence[int]], colorfamily: int) -> "VideoNode": ...
    def Sobel(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, scale: typing.Optional[float] = None) -> "VideoNode": ...
    def Splice(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], mismatch: typing.Optional[int] = None) -> "VideoNode": ...
    def StackHorizontal(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]]) -> "VideoNode": ...
    def StackVertical(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]]) -> "VideoNode": ...
    def Transpose(self, clip: "VideoNode") -> "VideoNode": ...
    def Trim(self, clip: "VideoNode", first: typing.Optional[int] = None, last: typing.Optional[int] = None, length: typing.Optional[int] = None) -> "VideoNode": ...
    def Turn180(self, clip: "VideoNode") -> "VideoNode": ...


class _Plugin_std_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def AddBorders(self, left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None, color: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
    def AssumeFPS(self, src: typing.Optional["VideoNode"] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None) -> "VideoNode": ...
    def Binarize(self, threshold: typing.Union[float, typing.Sequence[float], None] = None, v0: typing.Union[float, typing.Sequence[float], None] = None, v1: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def BlankClip(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, format: typing.Optional[int] = None, length: typing.Optional[int] = None, fpsnum: typing.Optional[int] = None, fpsden: typing.Optional[int] = None, color: typing.Union[float, typing.Sequence[float], None] = None, keep: typing.Optional[int] = None) -> "VideoNode": ...
    def BoxBlur(self, planes: typing.Union[int, typing.Sequence[int], None] = None, hradius: typing.Optional[int] = None, hpasses: typing.Optional[int] = None, vradius: typing.Optional[int] = None, vpasses: typing.Optional[int] = None) -> "VideoNode": ...
    def Cache(self, size: typing.Optional[int] = None, fixed: typing.Optional[int] = None, make_linear: typing.Optional[int] = None) -> "VideoNode": ...
    def ClipToProp(self, mclip: "VideoNode", prop: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Convolution(self, matrix: typing.Union[float, typing.Sequence[float]], bias: typing.Optional[float] = None, divisor: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, saturate: typing.Optional[int] = None, mode: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Crop(self, left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None) -> "VideoNode": ...
    def CropAbs(self, width: int, height: int, left: typing.Optional[int] = None, top: typing.Optional[int] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None) -> "VideoNode": ...
    def CropRel(self, left: typing.Optional[int] = None, right: typing.Optional[int] = None, top: typing.Optional[int] = None, bottom: typing.Optional[int] = None) -> "VideoNode": ...
    def Deflate(self, planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None) -> "VideoNode": ...
    def DeleteFrames(self, frames: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def DoubleWeave(self, tff: typing.Optional[int] = None) -> "VideoNode": ...
    def DuplicateFrames(self, frames: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Expr(self, expr: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], format: typing.Optional[int] = None) -> "VideoNode": ...
    def FlipHorizontal(self) -> "VideoNode": ...
    def FlipVertical(self) -> "VideoNode": ...
    def FrameEval(self, eval: typing.Callable[..., typing.Any], prop_src: typing.Union["VideoNode", typing.Sequence["VideoNode"], None] = None) -> "VideoNode": ...
    def FreezeFrames(self, first: typing.Union[int, typing.Sequence[int]], last: typing.Union[int, typing.Sequence[int]], replacement: typing.Union[int, typing.Sequence[int]]) -> "VideoNode": ...
    def Inflate(self, planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None) -> "VideoNode": ...
    def Interleave(self, extend: typing.Optional[int] = None, mismatch: typing.Optional[int] = None, modify_duration: typing.Optional[int] = None) -> "VideoNode": ...
    def Invert(self, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Levels(self, min_in: typing.Union[float, typing.Sequence[float], None] = None, max_in: typing.Union[float, typing.Sequence[float], None] = None, gamma: typing.Union[float, typing.Sequence[float], None] = None, min_out: typing.Union[float, typing.Sequence[float], None] = None, max_out: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Limiter(self, min: typing.Union[float, typing.Sequence[float], None] = None, max: typing.Union[float, typing.Sequence[float], None] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def LoadPlugin(self, altsearchpath: typing.Optional[int] = None, forcens: typing.Union[str, bytes, bytearray, None] = None, forceid: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Loop(self, times: typing.Optional[int] = None) -> "VideoNode": ...
    def Lut(self, planes: typing.Union[int, typing.Sequence[int], None] = None, lut: typing.Union[int, typing.Sequence[int], None] = None, lutf: typing.Union[float, typing.Sequence[float], None] = None, function: typing.Optional[typing.Callable[..., typing.Any]] = None, bits: typing.Optional[int] = None, floatout: typing.Optional[int] = None) -> "VideoNode": ...
    def Lut2(self, clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, lut: typing.Union[int, typing.Sequence[int], None] = None, lutf: typing.Union[float, typing.Sequence[float], None] = None, function: typing.Optional[typing.Callable[..., typing.Any]] = None, bits: typing.Optional[int] = None, floatout: typing.Optional[int] = None) -> "VideoNode": ...
    def MakeDiff(self, clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def MaskedMerge(self, clipb: "VideoNode", mask: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, first_plane: typing.Optional[int] = None, premultiplied: typing.Optional[int] = None) -> "VideoNode": ...
    def Maximum(self, planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None, coordinates: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Median(self, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Merge(self, clipb: "VideoNode", weight: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
    def MergeDiff(self, clipb: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Minimum(self, planes: typing.Union[int, typing.Sequence[int], None] = None, threshold: typing.Optional[float] = None, coordinates: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def ModifyFrame(self, clips: typing.Union["VideoNode", typing.Sequence["VideoNode"]], selector: typing.Callable[..., typing.Any]) -> "VideoNode": ...
    def PEMVerifier(self, upper: typing.Union[float, typing.Sequence[float], None] = None, lower: typing.Union[float, typing.Sequence[float], None] = None) -> "VideoNode": ...
    def PlaneStats(self, clipb: typing.Optional["VideoNode"] = None, plane: typing.Optional[int] = None, prop: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def PreMultiply(self, alpha: "VideoNode") -> "VideoNode": ...
    def Prewitt(self, planes: typing.Union[int, typing.Sequence[int], None] = None, scale: typing.Optional[float] = None) -> "VideoNode": ...
    def PropToClip(self, prop: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Reverse(self) -> "VideoNode": ...
    def SelectEvery(self, cycle: int, offsets: typing.Union[int, typing.Sequence[int]], modify_duration: typing.Optional[int] = None) -> "VideoNode": ...
    def SeparateFields(self, tff: typing.Optional[int] = None, modify_duration: typing.Optional[int] = None) -> "VideoNode": ...
    def SetFieldBased(self, value: int) -> "VideoNode": ...
    def SetFrameProp(self, prop: typing.Union[str, bytes, bytearray], delete: typing.Optional[int] = None, intval: typing.Union[int, typing.Sequence[int], None] = None, floatval: typing.Union[float, typing.Sequence[float], None] = None, data: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None) -> "VideoNode": ...
    def SetMaxCPU(self) -> "VideoNode": ...
    def ShufflePlanes(self, planes: typing.Union[int, typing.Sequence[int]], colorfamily: int) -> "VideoNode": ...
    def Sobel(self, planes: typing.Union[int, typing.Sequence[int], None] = None, scale: typing.Optional[float] = None) -> "VideoNode": ...
    def Splice(self, mismatch: typing.Optional[int] = None) -> "VideoNode": ...
    def StackHorizontal(self) -> "VideoNode": ...
    def StackVertical(self) -> "VideoNode": ...
    def Transpose(self) -> "VideoNode": ...
    def Trim(self, first: typing.Optional[int] = None, last: typing.Optional[int] = None, length: typing.Optional[int] = None) -> "VideoNode": ...
    def Turn180(self) -> "VideoNode": ...
# end implementation


# implementation: sub
class _Plugin_sub_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ImageFile(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], id: typing.Optional[int] = None, palette: typing.Union[int, typing.Sequence[int], None] = None, gray: typing.Optional[int] = None, info: typing.Optional[int] = None, flatten: typing.Optional[int] = None, blend: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Subtitle(self, clip: "VideoNode", text: typing.Union[str, bytes, bytearray], start: typing.Optional[int] = None, end: typing.Optional[int] = None, debuglevel: typing.Optional[int] = None, fontdir: typing.Union[str, bytes, bytearray, None] = None, linespacing: typing.Optional[float] = None, margins: typing.Union[int, typing.Sequence[int], None] = None, sar: typing.Optional[float] = None, style: typing.Union[str, bytes, bytearray, None] = None, blend: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def TextFile(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], charset: typing.Union[str, bytes, bytearray, None] = None, scale: typing.Optional[float] = None, debuglevel: typing.Optional[int] = None, fontdir: typing.Union[str, bytes, bytearray, None] = None, linespacing: typing.Optional[float] = None, margins: typing.Union[int, typing.Sequence[int], None] = None, sar: typing.Optional[float] = None, style: typing.Union[str, bytes, bytearray, None] = None, blend: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_sub_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ImageFile(self, file: typing.Union[str, bytes, bytearray], id: typing.Optional[int] = None, palette: typing.Union[int, typing.Sequence[int], None] = None, gray: typing.Optional[int] = None, info: typing.Optional[int] = None, flatten: typing.Optional[int] = None, blend: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def Subtitle(self, text: typing.Union[str, bytes, bytearray], start: typing.Optional[int] = None, end: typing.Optional[int] = None, debuglevel: typing.Optional[int] = None, fontdir: typing.Union[str, bytes, bytearray, None] = None, linespacing: typing.Optional[float] = None, margins: typing.Union[int, typing.Sequence[int], None] = None, sar: typing.Optional[float] = None, style: typing.Union[str, bytes, bytearray, None] = None, blend: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def TextFile(self, file: typing.Union[str, bytes, bytearray], charset: typing.Union[str, bytes, bytearray, None] = None, scale: typing.Optional[float] = None, debuglevel: typing.Optional[int] = None, fontdir: typing.Union[str, bytes, bytearray, None] = None, linespacing: typing.Optional[float] = None, margins: typing.Union[int, typing.Sequence[int], None] = None, sar: typing.Optional[float] = None, style: typing.Union[str, bytes, bytearray, None] = None, blend: typing.Optional[int] = None, matrix: typing.Optional[int] = None, matrix_s: typing.Union[str, bytes, bytearray, None] = None, transfer: typing.Optional[int] = None, transfer_s: typing.Union[str, bytes, bytearray, None] = None, primaries: typing.Optional[int] = None, primaries_s: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: surfaceblur
class _Plugin_surfaceblur_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def surfaceblur(self, input: "VideoNode", threshold: typing.Optional[float] = None, radius: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_surfaceblur_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def surfaceblur(self, threshold: typing.Optional[float] = None, radius: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: svp1
class _Plugin_svp1_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Analyse(self, clip: "VideoNode", sdata: int, src: "VideoNode", opt: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def Super(self, clip: "VideoNode", opt: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...


class _Plugin_svp1_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Analyse(self, sdata: int, src: "VideoNode", opt: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
    def Super(self, opt: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
# end implementation


# implementation: svp2
class _Plugin_svp2_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SmoothFps(self, clip: "VideoNode", super: "VideoNode", sdata: int, vectors: "VideoNode", vdata: int, opt: typing.Union[str, bytes, bytearray], src: typing.Optional["VideoNode"] = None, fps: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_svp2_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def SmoothFps(self, super: "VideoNode", sdata: int, vectors: "VideoNode", vdata: int, opt: typing.Union[str, bytes, bytearray], src: typing.Optional["VideoNode"] = None, fps: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: tcanny
class _Plugin_tcanny_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TCanny(self, clip: "VideoNode", sigma: typing.Union[float, typing.Sequence[float], None] = None, t_h: typing.Optional[float] = None, t_l: typing.Optional[float] = None, mode: typing.Optional[int] = None, op: typing.Optional[int] = None, gmmax: typing.Optional[float] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def TCannyCL(self, clip: "VideoNode", sigma: typing.Union[float, typing.Sequence[float], None] = None, t_h: typing.Optional[float] = None, t_l: typing.Optional[float] = None, mode: typing.Optional[int] = None, op: typing.Optional[int] = None, gmmax: typing.Optional[float] = None, device: typing.Optional[int] = None, list_device: typing.Optional[int] = None, info: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_tcanny_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TCanny(self, sigma: typing.Union[float, typing.Sequence[float], None] = None, t_h: typing.Optional[float] = None, t_l: typing.Optional[float] = None, mode: typing.Optional[int] = None, op: typing.Optional[int] = None, gmmax: typing.Optional[float] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def TCannyCL(self, sigma: typing.Union[float, typing.Sequence[float], None] = None, t_h: typing.Optional[float] = None, t_l: typing.Optional[float] = None, mode: typing.Optional[int] = None, op: typing.Optional[int] = None, gmmax: typing.Optional[float] = None, device: typing.Optional[int] = None, list_device: typing.Optional[int] = None, info: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: tcm
class _Plugin_tcm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TColorMask(self, clip: "VideoNode", colors: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], tolerance: typing.Optional[int] = None, bt601: typing.Optional[int] = None, gray: typing.Optional[int] = None, lutthr: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tcm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TColorMask(self, colors: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]]], tolerance: typing.Optional[int] = None, bt601: typing.Optional[int] = None, gray: typing.Optional[int] = None, lutthr: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tcomb
class _Plugin_tcomb_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TComb(self, clip: "VideoNode", mode: typing.Optional[int] = None, fthreshl: typing.Optional[int] = None, fthreshc: typing.Optional[int] = None, othreshl: typing.Optional[int] = None, othreshc: typing.Optional[int] = None, map: typing.Optional[int] = None, scthresh: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_tcomb_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TComb(self, mode: typing.Optional[int] = None, fthreshl: typing.Optional[int] = None, fthreshc: typing.Optional[int] = None, othreshl: typing.Optional[int] = None, othreshc: typing.Optional[int] = None, map: typing.Optional[int] = None, scthresh: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: tdm
class _Plugin_tdm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def IsCombed(self, clip: "VideoNode", cthresh: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, chroma: typing.Optional[int] = None, mi: typing.Optional[int] = None, metric: typing.Optional[int] = None) -> "VideoNode": ...
    def TDeintMod(self, clip: "VideoNode", order: int, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, length: typing.Optional[int] = None, mtype: typing.Optional[int] = None, ttype: typing.Optional[int] = None, mtql: typing.Optional[int] = None, mthl: typing.Optional[int] = None, mtqc: typing.Optional[int] = None, mthc: typing.Optional[int] = None, nt: typing.Optional[int] = None, minthresh: typing.Optional[int] = None, maxthresh: typing.Optional[int] = None, cstr: typing.Optional[int] = None, athresh: typing.Optional[int] = None, metric: typing.Optional[int] = None, expand: typing.Optional[int] = None, link: typing.Optional[int] = None, show: typing.Optional[int] = None, edeint: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_tdm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def IsCombed(self, cthresh: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, chroma: typing.Optional[int] = None, mi: typing.Optional[int] = None, metric: typing.Optional[int] = None) -> "VideoNode": ...
    def TDeintMod(self, order: int, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, length: typing.Optional[int] = None, mtype: typing.Optional[int] = None, ttype: typing.Optional[int] = None, mtql: typing.Optional[int] = None, mthl: typing.Optional[int] = None, mtqc: typing.Optional[int] = None, mthc: typing.Optional[int] = None, nt: typing.Optional[int] = None, minthresh: typing.Optional[int] = None, maxthresh: typing.Optional[int] = None, cstr: typing.Optional[int] = None, athresh: typing.Optional[int] = None, metric: typing.Optional[int] = None, expand: typing.Optional[int] = None, link: typing.Optional[int] = None, show: typing.Optional[int] = None, edeint: typing.Optional["VideoNode"] = None, opt: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: tedgemask
class _Plugin_tedgemask_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TEdgeMask(self, clip: "VideoNode", threshold: typing.Union[float, typing.Sequence[float], None] = None, type: typing.Optional[int] = None, link: typing.Optional[int] = None, scale: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tedgemask_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TEdgeMask(self, threshold: typing.Union[float, typing.Sequence[float], None] = None, type: typing.Optional[int] = None, link: typing.Optional[int] = None, scale: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: text
class _Plugin_text_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ClipInfo(self, clip: "VideoNode", alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def CoreInfo(self, clip: typing.Optional["VideoNode"] = None, alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameNum(self, clip: "VideoNode", alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameProps(self, clip: "VideoNode", props: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def Text(self, clip: "VideoNode", text: typing.Union[str, bytes, bytearray], alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_text_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ClipInfo(self, alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def CoreInfo(self, alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameNum(self, alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def FrameProps(self, props: typing.Union[str, bytes, bytearray, typing.Sequence[typing.Union[str, bytes, bytearray]], None] = None, alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
    def Text(self, text: typing.Union[str, bytes, bytearray], alignment: typing.Optional[int] = None, scale: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: timecube
class _Plugin_timecube_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Cube(self, clip: "VideoNode", cube: typing.Union[str, bytes, bytearray], format: typing.Optional[int] = None, range: typing.Optional[int] = None, cpu: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_timecube_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Cube(self, cube: typing.Union[str, bytes, bytearray], format: typing.Optional[int] = None, range: typing.Optional[int] = None, cpu: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tivtc
class _Plugin_tivtc_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TDecimate(self, clip: "VideoNode", mode: typing.Optional[int] = None, cycleR: typing.Optional[int] = None, cycle: typing.Optional[int] = None, rate: typing.Optional[float] = None, dupThresh: typing.Optional[float] = None, vidThresh: typing.Optional[float] = None, sceneThresh: typing.Optional[float] = None, hybrid: typing.Optional[int] = None, vidDetect: typing.Optional[int] = None, conCycle: typing.Optional[int] = None, conCycleTP: typing.Optional[int] = None, ovr: typing.Union[str, bytes, bytearray, None] = None, output: typing.Union[str, bytes, bytearray, None] = None, input: typing.Union[str, bytes, bytearray, None] = None, tfmIn: typing.Union[str, bytes, bytearray, None] = None, mkvOut: typing.Union[str, bytes, bytearray, None] = None, nt: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, debug: typing.Optional[int] = None, display: typing.Optional[int] = None, vfrDec: typing.Optional[int] = None, batch: typing.Optional[int] = None, tcfv1: typing.Optional[int] = None, se: typing.Optional[int] = None, chroma: typing.Optional[int] = None, exPP: typing.Optional[int] = None, maxndl: typing.Optional[int] = None, m2PA: typing.Optional[int] = None, denoise: typing.Optional[int] = None, noblend: typing.Optional[int] = None, ssd: typing.Optional[int] = None, hint: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None, sdlim: typing.Optional[int] = None, opt: typing.Optional[int] = None, orgOut: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def TFM(self, clip: "VideoNode", order: typing.Optional[int] = None, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, PP: typing.Optional[int] = None, ovr: typing.Union[str, bytes, bytearray, None] = None, input: typing.Union[str, bytes, bytearray, None] = None, output: typing.Union[str, bytes, bytearray, None] = None, outputC: typing.Union[str, bytes, bytearray, None] = None, debug: typing.Optional[int] = None, display: typing.Optional[int] = None, slow: typing.Optional[int] = None, mChroma: typing.Optional[int] = None, cNum: typing.Optional[int] = None, cthresh: typing.Optional[int] = None, MI: typing.Optional[int] = None, chroma: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, y0: typing.Optional[int] = None, y1: typing.Optional[int] = None, mthresh: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None, d2v: typing.Union[str, bytes, bytearray, None] = None, ovrDefault: typing.Optional[int] = None, flags: typing.Optional[int] = None, scthresh: typing.Optional[float] = None, micout: typing.Optional[int] = None, micmatching: typing.Optional[int] = None, trimIn: typing.Union[str, bytes, bytearray, None] = None, hint: typing.Optional[int] = None, metric: typing.Optional[int] = None, batch: typing.Optional[int] = None, ubsco: typing.Optional[int] = None, mmsco: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tivtc_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TDecimate(self, mode: typing.Optional[int] = None, cycleR: typing.Optional[int] = None, cycle: typing.Optional[int] = None, rate: typing.Optional[float] = None, dupThresh: typing.Optional[float] = None, vidThresh: typing.Optional[float] = None, sceneThresh: typing.Optional[float] = None, hybrid: typing.Optional[int] = None, vidDetect: typing.Optional[int] = None, conCycle: typing.Optional[int] = None, conCycleTP: typing.Optional[int] = None, ovr: typing.Union[str, bytes, bytearray, None] = None, output: typing.Union[str, bytes, bytearray, None] = None, input: typing.Union[str, bytes, bytearray, None] = None, tfmIn: typing.Union[str, bytes, bytearray, None] = None, mkvOut: typing.Union[str, bytes, bytearray, None] = None, nt: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, debug: typing.Optional[int] = None, display: typing.Optional[int] = None, vfrDec: typing.Optional[int] = None, batch: typing.Optional[int] = None, tcfv1: typing.Optional[int] = None, se: typing.Optional[int] = None, chroma: typing.Optional[int] = None, exPP: typing.Optional[int] = None, maxndl: typing.Optional[int] = None, m2PA: typing.Optional[int] = None, denoise: typing.Optional[int] = None, noblend: typing.Optional[int] = None, ssd: typing.Optional[int] = None, hint: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None, sdlim: typing.Optional[int] = None, opt: typing.Optional[int] = None, orgOut: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def TFM(self, order: typing.Optional[int] = None, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, PP: typing.Optional[int] = None, ovr: typing.Union[str, bytes, bytearray, None] = None, input: typing.Union[str, bytes, bytearray, None] = None, output: typing.Union[str, bytes, bytearray, None] = None, outputC: typing.Union[str, bytes, bytearray, None] = None, debug: typing.Optional[int] = None, display: typing.Optional[int] = None, slow: typing.Optional[int] = None, mChroma: typing.Optional[int] = None, cNum: typing.Optional[int] = None, cthresh: typing.Optional[int] = None, MI: typing.Optional[int] = None, chroma: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, y0: typing.Optional[int] = None, y1: typing.Optional[int] = None, mthresh: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None, d2v: typing.Union[str, bytes, bytearray, None] = None, ovrDefault: typing.Optional[int] = None, flags: typing.Optional[int] = None, scthresh: typing.Optional[float] = None, micout: typing.Optional[int] = None, micmatching: typing.Optional[int] = None, trimIn: typing.Union[str, bytes, bytearray, None] = None, hint: typing.Optional[int] = None, metric: typing.Optional[int] = None, batch: typing.Optional[int] = None, ubsco: typing.Optional[int] = None, mmsco: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tla
class _Plugin_tla_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TempLinearApproximate(self, clip: "VideoNode", radius: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, gamma: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tla_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TempLinearApproximate(self, radius: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, gamma: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tmap
class _Plugin_tmap_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def tm(self, clip: "VideoNode", source_peak: float, desat: typing.Optional[float] = None, lin: typing.Optional[int] = None, show_satmask: typing.Optional[int] = None, show_clipped: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tmap_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def tm(self, source_peak: float, desat: typing.Optional[float] = None, lin: typing.Optional[int] = None, show_satmask: typing.Optional[int] = None, show_clipped: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tmc
class _Plugin_tmc_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TMaskCleaner(self, clip: "VideoNode", length: typing.Optional[int] = None, thresh: typing.Optional[int] = None, fade: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tmc_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TMaskCleaner(self, length: typing.Optional[int] = None, thresh: typing.Optional[int] = None, fade: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tmedian
class _Plugin_tmedian_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TemporalMedian(self, clip: "VideoNode", radius: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_tmedian_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TemporalMedian(self, radius: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: tnlm
class _Plugin_tnlm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TNLMeans(self, clip: "VideoNode", ax: typing.Optional[int] = None, ay: typing.Optional[int] = None, az: typing.Optional[int] = None, sx: typing.Optional[int] = None, sy: typing.Optional[int] = None, bx: typing.Optional[int] = None, by: typing.Optional[int] = None, a: typing.Optional[float] = None, h: typing.Optional[float] = None, ssd: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_tnlm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TNLMeans(self, ax: typing.Optional[int] = None, ay: typing.Optional[int] = None, az: typing.Optional[int] = None, sx: typing.Optional[int] = None, sy: typing.Optional[int] = None, bx: typing.Optional[int] = None, by: typing.Optional[int] = None, a: typing.Optional[float] = None, h: typing.Optional[float] = None, ssd: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: tonemap
class _Plugin_tonemap_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Hable(self, clip: "VideoNode", exposure: typing.Optional[float] = None, a: typing.Optional[float] = None, b: typing.Optional[float] = None, c: typing.Optional[float] = None, d: typing.Optional[float] = None, e: typing.Optional[float] = None, f: typing.Optional[float] = None, w: typing.Optional[float] = None) -> "VideoNode": ...
    def Mobius(self, clip: "VideoNode", exposure: typing.Optional[float] = None, transition: typing.Optional[float] = None, peak: typing.Optional[float] = None) -> "VideoNode": ...
    def Reinhard(self, clip: "VideoNode", exposure: typing.Optional[float] = None, contrast: typing.Optional[float] = None, peak: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_tonemap_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Hable(self, exposure: typing.Optional[float] = None, a: typing.Optional[float] = None, b: typing.Optional[float] = None, c: typing.Optional[float] = None, d: typing.Optional[float] = None, e: typing.Optional[float] = None, f: typing.Optional[float] = None, w: typing.Optional[float] = None) -> "VideoNode": ...
    def Mobius(self, exposure: typing.Optional[float] = None, transition: typing.Optional[float] = None, peak: typing.Optional[float] = None) -> "VideoNode": ...
    def Reinhard(self, exposure: typing.Optional[float] = None, contrast: typing.Optional[float] = None, peak: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: trans
class _Plugin_trans_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Accord(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, twin: typing.Optional[int] = None, open: typing.Optional[int] = None) -> "VideoNode": ...
    def Bubbles(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, static: typing.Optional[int] = None) -> "VideoNode": ...
    def Central(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, nturns: typing.Optional[int] = None, emerge: typing.Optional[int] = None, resize: typing.Optional[int] = None) -> "VideoNode": ...
    def Crumple(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, crumple: typing.Optional[int] = None, emerge: typing.Optional[int] = None) -> "VideoNode": ...
    def Disco(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, emerge: typing.Optional[int] = None, rad: typing.Optional[int] = None, nturns: typing.Optional[int] = None) -> "VideoNode": ...
    def Door(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, open: typing.Optional[int] = None, vert: typing.Optional[int] = None) -> "VideoNode": ...
    def FlipPage(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, left: typing.Optional[int] = None) -> "VideoNode": ...
    def Funnel(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Paint(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None) -> "VideoNode": ...
    def Push(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Ripple(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, origin: typing.Optional[int] = None, amp: typing.Optional[int] = None, wlength: typing.Optional[int] = None, merge: typing.Optional[int] = None) -> "VideoNode": ...
    def Roll(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, rollin: typing.Optional[int] = None) -> "VideoNode": ...
    def Scratch(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None) -> "VideoNode": ...
    def Shuffle(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Slide(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, slidein: typing.Optional[int] = None) -> "VideoNode": ...
    def Sprite(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Swing(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, open: typing.Optional[int] = None, ndoors: typing.Optional[int] = None, corner: typing.Optional[int] = None) -> "VideoNode": ...
    def Swirl(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, qr: typing.Optional[int] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def VenitianBlinds(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None, slatwidth: typing.Optional[int] = None, open: typing.Optional[int] = None) -> "VideoNode": ...
    def Weave(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None, slatwidth: typing.Optional[int] = None) -> "VideoNode": ...
    def Wipe(self, clip: "VideoNode", clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_trans_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Accord(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, twin: typing.Optional[int] = None, open: typing.Optional[int] = None) -> "VideoNode": ...
    def Bubbles(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, static: typing.Optional[int] = None) -> "VideoNode": ...
    def Central(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, nturns: typing.Optional[int] = None, emerge: typing.Optional[int] = None, resize: typing.Optional[int] = None) -> "VideoNode": ...
    def Crumple(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, crumple: typing.Optional[int] = None, emerge: typing.Optional[int] = None) -> "VideoNode": ...
    def Disco(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, emerge: typing.Optional[int] = None, rad: typing.Optional[int] = None, nturns: typing.Optional[int] = None) -> "VideoNode": ...
    def Door(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, open: typing.Optional[int] = None, vert: typing.Optional[int] = None) -> "VideoNode": ...
    def FlipPage(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, left: typing.Optional[int] = None) -> "VideoNode": ...
    def Funnel(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Paint(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None) -> "VideoNode": ...
    def Push(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Ripple(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, origin: typing.Optional[int] = None, amp: typing.Optional[int] = None, wlength: typing.Optional[int] = None, merge: typing.Optional[int] = None) -> "VideoNode": ...
    def Roll(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, rollin: typing.Optional[int] = None) -> "VideoNode": ...
    def Scratch(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None) -> "VideoNode": ...
    def Shuffle(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Slide(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, slidein: typing.Optional[int] = None) -> "VideoNode": ...
    def Sprite(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def Swing(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None, open: typing.Optional[int] = None, ndoors: typing.Optional[int] = None, corner: typing.Optional[int] = None) -> "VideoNode": ...
    def Swirl(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, qr: typing.Optional[int] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
    def VenitianBlinds(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None, slatwidth: typing.Optional[int] = None, open: typing.Optional[int] = None) -> "VideoNode": ...
    def Weave(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, style: typing.Optional[int] = None, slatwidth: typing.Optional[int] = None) -> "VideoNode": ...
    def Wipe(self, clipb: "VideoNode", overlap: typing.Optional[float] = None, dir: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: ttmpsm
class _Plugin_ttmpsm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TTempSmooth(self, clip: "VideoNode", maxr: typing.Optional[int] = None, thresh: typing.Union[int, typing.Sequence[int], None] = None, mdiff: typing.Union[int, typing.Sequence[int], None] = None, strength: typing.Optional[int] = None, scthresh: typing.Optional[float] = None, fp: typing.Optional[int] = None, pfclip: typing.Optional["VideoNode"] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_ttmpsm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TTempSmooth(self, maxr: typing.Optional[int] = None, thresh: typing.Union[int, typing.Sequence[int], None] = None, mdiff: typing.Union[int, typing.Sequence[int], None] = None, strength: typing.Optional[int] = None, scthresh: typing.Optional[float] = None, fp: typing.Optional[int] = None, pfclip: typing.Optional["VideoNode"] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: vcfreq
class _Plugin_vcfreq_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Blur(self, clip: "VideoNode", line: typing.Optional[int] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None) -> "VideoNode": ...
    def F1Quiver(self, clip: "VideoNode", filter: typing.Union[int, typing.Sequence[int]], morph: typing.Optional[int] = None, custom: typing.Optional[int] = None, test: typing.Optional[int] = None, strow: typing.Optional[int] = None, nrows: typing.Optional[int] = None, gamma: typing.Optional[float] = None) -> "VideoNode": ...
    def F2Quiver(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Sharp(self, clip: "VideoNode", line: typing.Optional[int] = None, wn: typing.Optional[float] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None, fr: typing.Optional[int] = None, scale: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_vcfreq_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Blur(self, line: typing.Optional[int] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None) -> "VideoNode": ...
    def F1Quiver(self, filter: typing.Union[int, typing.Sequence[int]], morph: typing.Optional[int] = None, custom: typing.Optional[int] = None, test: typing.Optional[int] = None, strow: typing.Optional[int] = None, nrows: typing.Optional[int] = None, gamma: typing.Optional[float] = None) -> "VideoNode": ...
    def F2Quiver(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Sharp(self, line: typing.Optional[int] = None, wn: typing.Optional[float] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None, fr: typing.Optional[int] = None, scale: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: vcmod
class _Plugin_vcmod_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Amplitude(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Fan(self, clip: "VideoNode", span: typing.Optional[int] = None, edge: typing.Optional[int] = None, plus: typing.Optional[int] = None, minus: typing.Optional[int] = None, uv: typing.Optional[int] = None) -> "VideoNode": ...
    def GBlur(self, clip: "VideoNode", ksize: typing.Optional[int] = None, sd: typing.Optional[float] = None) -> "VideoNode": ...
    def Histogram(self, clip: "VideoNode", clipm: typing.Optional["VideoNode"] = None, type: typing.Optional[int] = None, table: typing.Union[int, typing.Sequence[int], None] = None, mf: typing.Optional[int] = None, window: typing.Optional[int] = None, limit: typing.Optional[int] = None) -> "VideoNode": ...
    def MBlur(self, clip: "VideoNode", type: typing.Optional[int] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None) -> "VideoNode": ...
    def Median(self, clip: "VideoNode", maxgrid: typing.Optional[int] = None, plane: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Neural(self, clip: "VideoNode", txt: typing.Union[str, bytes, bytearray, None] = None, fname: typing.Union[str, bytes, bytearray, None] = None, tclip: typing.Optional["VideoNode"] = None, xpts: typing.Optional[int] = None, ypts: typing.Optional[int] = None, tlx: typing.Optional[int] = None, tty: typing.Optional[int] = None, trx: typing.Optional[int] = None, tby: typing.Optional[int] = None, iter: typing.Optional[int] = None, bestof: typing.Optional[int] = None, wset: typing.Optional[int] = None, rgb: typing.Optional[int] = None) -> "VideoNode": ...
    def SaltPepper(self, clip: "VideoNode", planes: typing.Union[int, typing.Sequence[int], None] = None, tol: typing.Optional[int] = None, avg: typing.Optional[int] = None) -> "VideoNode": ...
    def Variance(self, clip: "VideoNode", lx: int, wd: int, ty: int, ht: int, fn: typing.Optional[int] = None, uv: typing.Optional[int] = None, xgrid: typing.Optional[int] = None, ygrid: typing.Optional[int] = None) -> "VideoNode": ...
    def Veed(self, clip: "VideoNode", str: typing.Optional[int] = None, rad: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, plimit: typing.Union[int, typing.Sequence[int], None] = None, mlimit: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_vcmod_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Amplitude(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Fan(self, span: typing.Optional[int] = None, edge: typing.Optional[int] = None, plus: typing.Optional[int] = None, minus: typing.Optional[int] = None, uv: typing.Optional[int] = None) -> "VideoNode": ...
    def GBlur(self, ksize: typing.Optional[int] = None, sd: typing.Optional[float] = None) -> "VideoNode": ...
    def Histogram(self, clipm: typing.Optional["VideoNode"] = None, type: typing.Optional[int] = None, table: typing.Union[int, typing.Sequence[int], None] = None, mf: typing.Optional[int] = None, window: typing.Optional[int] = None, limit: typing.Optional[int] = None) -> "VideoNode": ...
    def MBlur(self, type: typing.Optional[int] = None, x: typing.Optional[int] = None, y: typing.Optional[int] = None) -> "VideoNode": ...
    def Median(self, maxgrid: typing.Optional[int] = None, plane: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
    def Neural(self, txt: typing.Union[str, bytes, bytearray, None] = None, fname: typing.Union[str, bytes, bytearray, None] = None, tclip: typing.Optional["VideoNode"] = None, xpts: typing.Optional[int] = None, ypts: typing.Optional[int] = None, tlx: typing.Optional[int] = None, tty: typing.Optional[int] = None, trx: typing.Optional[int] = None, tby: typing.Optional[int] = None, iter: typing.Optional[int] = None, bestof: typing.Optional[int] = None, wset: typing.Optional[int] = None, rgb: typing.Optional[int] = None) -> "VideoNode": ...
    def SaltPepper(self, planes: typing.Union[int, typing.Sequence[int], None] = None, tol: typing.Optional[int] = None, avg: typing.Optional[int] = None) -> "VideoNode": ...
    def Variance(self, lx: int, wd: int, ty: int, ht: int, fn: typing.Optional[int] = None, uv: typing.Optional[int] = None, xgrid: typing.Optional[int] = None, ygrid: typing.Optional[int] = None) -> "VideoNode": ...
    def Veed(self, str: typing.Optional[int] = None, rad: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, plimit: typing.Union[int, typing.Sequence[int], None] = None, mlimit: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: vcmove
class _Plugin_vcmove_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DeBarrel(self, clip: "VideoNode", a: float, b: float, c: float, vhr: typing.Optional[float] = None, pin: typing.Optional[int] = None, yind: typing.Optional[int] = None, ypin: typing.Optional[int] = None, ya: typing.Optional[float] = None, yb: typing.Optional[float] = None, yc: typing.Optional[float] = None, test: typing.Optional[int] = None) -> "VideoNode": ...
    def Quad2Rect(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Rect2Quad(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Rotate(self, clip: "VideoNode", bkg: "VideoNode", angle: float, dinc: typing.Optional[float] = None, lx: typing.Optional[int] = None, wd: typing.Optional[int] = None, ty: typing.Optional[int] = None, ht: typing.Optional[int] = None, axx: typing.Optional[int] = None, axy: typing.Optional[int] = None, intq: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_vcmove_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def DeBarrel(self, a: float, b: float, c: float, vhr: typing.Optional[float] = None, pin: typing.Optional[int] = None, yind: typing.Optional[int] = None, ypin: typing.Optional[int] = None, ya: typing.Optional[float] = None, yb: typing.Optional[float] = None, yc: typing.Optional[float] = None, test: typing.Optional[int] = None) -> "VideoNode": ...
    def Quad2Rect(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Rect2Quad(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Optional["VideoNode"]: ...
    def Rotate(self, bkg: "VideoNode", angle: float, dinc: typing.Optional[float] = None, lx: typing.Optional[int] = None, wd: typing.Optional[int] = None, ty: typing.Optional[int] = None, ht: typing.Optional[int] = None, axx: typing.Optional[int] = None, axy: typing.Optional[int] = None, intq: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: vd
class _Plugin_vd_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def VagueDenoiser(self, clip: "VideoNode", threshold: typing.Optional[float] = None, method: typing.Optional[int] = None, nsteps: typing.Optional[int] = None, percent: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...


class _Plugin_vd_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def VagueDenoiser(self, threshold: typing.Optional[float] = None, method: typing.Optional[int] = None, nsteps: typing.Optional[int] = None, percent: typing.Optional[float] = None, planes: typing.Union[int, typing.Sequence[int], None] = None) -> "VideoNode": ...
# end implementation


# implementation: vfrtocfr
class _Plugin_vfrtocfr_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def VFRToCFR(self, clip: "VideoNode", timecodes: typing.Union[str, bytes, bytearray], fpsnum: int, fpsden: int, drop: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_vfrtocfr_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def VFRToCFR(self, timecodes: typing.Union[str, bytes, bytearray], fpsnum: int, fpsden: int, drop: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: vinverse
class _Plugin_vinverse_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Vinverse(self, clip: "VideoNode", sstr: typing.Optional[float] = None, amnt: typing.Optional[int] = None, scl: typing.Optional[float] = None) -> "VideoNode": ...


class _Plugin_vinverse_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Vinverse(self, sstr: typing.Optional[float] = None, amnt: typing.Optional[int] = None, scl: typing.Optional[float] = None) -> "VideoNode": ...
# end implementation


# implementation: vivtc
class _Plugin_vivtc_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def VDecimate(self, clip: "VideoNode", cycle: typing.Optional[int] = None, chroma: typing.Optional[int] = None, dupthresh: typing.Optional[float] = None, scthresh: typing.Optional[float] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None, ovr: typing.Union[str, bytes, bytearray, None] = None, dryrun: typing.Optional[int] = None) -> "VideoNode": ...
    def VFM(self, clip: "VideoNode", order: int, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, mchroma: typing.Optional[int] = None, cthresh: typing.Optional[int] = None, mi: typing.Optional[int] = None, chroma: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, y0: typing.Optional[int] = None, y1: typing.Optional[int] = None, scthresh: typing.Optional[float] = None, micmatch: typing.Optional[int] = None, micout: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None) -> "VideoNode": ...


class _Plugin_vivtc_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def VDecimate(self, cycle: typing.Optional[int] = None, chroma: typing.Optional[int] = None, dupthresh: typing.Optional[float] = None, scthresh: typing.Optional[float] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None, ovr: typing.Union[str, bytes, bytearray, None] = None, dryrun: typing.Optional[int] = None) -> "VideoNode": ...
    def VFM(self, order: int, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, mchroma: typing.Optional[int] = None, cthresh: typing.Optional[int] = None, mi: typing.Optional[int] = None, chroma: typing.Optional[int] = None, blockx: typing.Optional[int] = None, blocky: typing.Optional[int] = None, y0: typing.Optional[int] = None, y1: typing.Optional[int] = None, scthresh: typing.Optional[float] = None, micmatch: typing.Optional[int] = None, micout: typing.Optional[int] = None, clip2: typing.Optional["VideoNode"] = None) -> "VideoNode": ...
# end implementation


# implementation: vscope
class _Plugin_vscope_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Scope(self, clip: "VideoNode", mode: typing.Union[str, bytes, bytearray, None] = None, tickmarks: typing.Optional[int] = None, side: typing.Union[str, bytes, bytearray, None] = None, bottom: typing.Union[str, bytes, bytearray, None] = None, corner: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_vscope_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Scope(self, mode: typing.Union[str, bytes, bytearray, None] = None, tickmarks: typing.Optional[int] = None, side: typing.Union[str, bytes, bytearray, None] = None, bottom: typing.Union[str, bytes, bytearray, None] = None, corner: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: vsf
class _Plugin_vsf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TextSub(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], charset: typing.Optional[int] = None, fps: typing.Optional[float] = None, vfr: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def VobSub(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...


class _Plugin_vsf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TextSub(self, file: typing.Union[str, bytes, bytearray], charset: typing.Optional[int] = None, fps: typing.Optional[float] = None, vfr: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def VobSub(self, file: typing.Union[str, bytes, bytearray]) -> "VideoNode": ...
# end implementation


# implementation: vsfm
class _Plugin_vsfm_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TextSubMod(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], charset: typing.Optional[int] = None, fps: typing.Optional[float] = None, vfr: typing.Union[str, bytes, bytearray, None] = None, accurate: typing.Optional[int] = None) -> "VideoNode": ...
    def VobSub(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], accurate: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_vsfm_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TextSubMod(self, file: typing.Union[str, bytes, bytearray], charset: typing.Optional[int] = None, fps: typing.Optional[float] = None, vfr: typing.Union[str, bytes, bytearray, None] = None, accurate: typing.Optional[int] = None) -> "VideoNode": ...
    def VobSub(self, file: typing.Union[str, bytes, bytearray], accurate: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: w2xc
class _Plugin_w2xc_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Waifu2x(self, clip: "VideoNode", noise: typing.Optional[int] = None, scale: typing.Optional[int] = None, block: typing.Optional[int] = None, photo: typing.Optional[int] = None, gpu: typing.Optional[int] = None, processor: typing.Optional[int] = None, list_proc: typing.Optional[int] = None, log: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_w2xc_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Waifu2x(self, noise: typing.Optional[int] = None, scale: typing.Optional[int] = None, block: typing.Optional[int] = None, photo: typing.Optional[int] = None, gpu: typing.Optional[int] = None, processor: typing.Optional[int] = None, list_proc: typing.Optional[int] = None, log: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: w2xnvk
class _Plugin_w2xnvk_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Waifu2x(self, clip: "VideoNode", noise: typing.Optional[int] = None, scale: typing.Optional[int] = None, model: typing.Optional[int] = None, tile_size: typing.Optional[int] = None, gpu_id: typing.Optional[int] = None, gpu_thread: typing.Optional[int] = None, precision: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_w2xnvk_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Waifu2x(self, noise: typing.Optional[int] = None, scale: typing.Optional[int] = None, model: typing.Optional[int] = None, tile_size: typing.Optional[int] = None, gpu_id: typing.Optional[int] = None, gpu_thread: typing.Optional[int] = None, precision: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: w3fdif
class _Plugin_w3fdif_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def W3FDIF(self, clip: "VideoNode", order: int, mode: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_w3fdif_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def W3FDIF(self, order: int, mode: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: waifu2x
class _Plugin_waifu2x_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Denoise(self, input: "VideoNode", mode: typing.Optional[int] = None, matrix: typing.Optional[int] = None, full: typing.Optional[int] = None, block_width: typing.Optional[int] = None, block_height: typing.Optional[int] = None, threads: typing.Optional[int] = None) -> "VideoNode": ...
    def Resize(self, input: "VideoNode", width: typing.Optional[int] = None, height: typing.Optional[int] = None, shift_w: typing.Optional[float] = None, shift_h: typing.Optional[float] = None, subwidth: typing.Optional[float] = None, subheight: typing.Optional[float] = None, filter: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, subsample_w: typing.Optional[int] = None, subsample_h: typing.Optional[int] = None, chroma_loc_in: typing.Union[str, bytes, bytearray, None] = None, chroma_loc_out: typing.Union[str, bytes, bytearray, None] = None, matrix: typing.Optional[int] = None, full: typing.Optional[int] = None, block_width: typing.Optional[int] = None, block_height: typing.Optional[int] = None, threads: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_waifu2x_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Denoise(self, mode: typing.Optional[int] = None, matrix: typing.Optional[int] = None, full: typing.Optional[int] = None, block_width: typing.Optional[int] = None, block_height: typing.Optional[int] = None, threads: typing.Optional[int] = None) -> "VideoNode": ...
    def Resize(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None, shift_w: typing.Optional[float] = None, shift_h: typing.Optional[float] = None, subwidth: typing.Optional[float] = None, subheight: typing.Optional[float] = None, filter: typing.Union[str, bytes, bytearray, None] = None, filter_param_a: typing.Optional[float] = None, filter_param_b: typing.Optional[float] = None, filter_uv: typing.Union[str, bytes, bytearray, None] = None, filter_param_a_uv: typing.Optional[float] = None, filter_param_b_uv: typing.Optional[float] = None, subsample_w: typing.Optional[int] = None, subsample_h: typing.Optional[int] = None, chroma_loc_in: typing.Union[str, bytes, bytearray, None] = None, chroma_loc_out: typing.Union[str, bytes, bytearray, None] = None, matrix: typing.Optional[int] = None, full: typing.Optional[int] = None, block_width: typing.Optional[int] = None, block_height: typing.Optional[int] = None, threads: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: warp
class _Plugin_warp_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ABlur(self, clip: "VideoNode", blur: typing.Optional[int] = None, type: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def ASobel(self, clip: "VideoNode", thresh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def AWarp(self, clip: "VideoNode", mask: "VideoNode", depth: typing.Union[int, typing.Sequence[int], None] = None, chroma: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None, cplace: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def AWarpSharp2(self, clip: "VideoNode", thresh: typing.Optional[int] = None, blur: typing.Optional[int] = None, type: typing.Optional[int] = None, depth: typing.Union[int, typing.Sequence[int], None] = None, chroma: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None, cplace: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_warp_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def ABlur(self, blur: typing.Optional[int] = None, type: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def ASobel(self, thresh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
    def AWarp(self, mask: "VideoNode", depth: typing.Union[int, typing.Sequence[int], None] = None, chroma: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None, cplace: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
    def AWarpSharp2(self, thresh: typing.Optional[int] = None, blur: typing.Optional[int] = None, type: typing.Optional[int] = None, depth: typing.Union[int, typing.Sequence[int], None] = None, chroma: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, opt: typing.Optional[int] = None, cplace: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


# implementation: wwxd
class _Plugin_wwxd_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def WWXD(self, clip: "VideoNode") -> "VideoNode": ...


class _Plugin_wwxd_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def WWXD(self) -> "VideoNode": ...
# end implementation


# implementation: xyvsf
class _Plugin_xyvsf_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TextSub(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], charset: typing.Optional[int] = None, fps: typing.Optional[float] = None, vfr: typing.Union[str, bytes, bytearray, None] = None, swapuv: typing.Optional[int] = None) -> "VideoNode": ...
    def VobSub(self, clip: "VideoNode", file: typing.Union[str, bytes, bytearray], swapuv: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_xyvsf_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def TextSub(self, file: typing.Union[str, bytes, bytearray], charset: typing.Optional[int] = None, fps: typing.Optional[float] = None, vfr: typing.Union[str, bytes, bytearray, None] = None, swapuv: typing.Optional[int] = None) -> "VideoNode": ...
    def VobSub(self, file: typing.Union[str, bytes, bytearray], swapuv: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: yadifmod
class _Plugin_yadifmod_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Yadifmod(self, clip: "VideoNode", edeint: "VideoNode", order: int, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...


class _Plugin_yadifmod_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def Yadifmod(self, edeint: "VideoNode", order: int, field: typing.Optional[int] = None, mode: typing.Optional[int] = None, opt: typing.Optional[int] = None) -> "VideoNode": ...
# end implementation


# implementation: znedi3
class _Plugin_znedi3_Unbound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def nnedi3(self, clip: "VideoNode", field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, nsize: typing.Optional[int] = None, nns: typing.Optional[int] = None, qual: typing.Optional[int] = None, etype: typing.Optional[int] = None, pscrn: typing.Optional[int] = None, opt: typing.Optional[int] = None, int16_prescreener: typing.Optional[int] = None, int16_predictor: typing.Optional[int] = None, exp: typing.Optional[int] = None, show_mask: typing.Optional[int] = None, x_nnedi3_weights_bin: typing.Union[str, bytes, bytearray, None] = None, x_cpu: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...


class _Plugin_znedi3_Bound(Plugin):
    """
    This class implements the module definitions for the corresponding VapourSynth plugin.
    This class cannot be imported.
    """
    def nnedi3(self, field: int, dh: typing.Optional[int] = None, planes: typing.Union[int, typing.Sequence[int], None] = None, nsize: typing.Optional[int] = None, nns: typing.Optional[int] = None, qual: typing.Optional[int] = None, etype: typing.Optional[int] = None, pscrn: typing.Optional[int] = None, opt: typing.Optional[int] = None, int16_prescreener: typing.Optional[int] = None, int16_predictor: typing.Optional[int] = None, exp: typing.Optional[int] = None, show_mask: typing.Optional[int] = None, x_nnedi3_weights_bin: typing.Union[str, bytes, bytearray, None] = None, x_cpu: typing.Union[str, bytes, bytearray, None] = None) -> "VideoNode": ...
# end implementation


class VideoNode:
# instance_bound: DGMVC
    @property
    def DGMVC(self) -> _Plugin_DGMVC_Bound:
        """
        DGMVCSource for VapourSynth
        """
# end instance
# instance_bound: acrop
    @property
    def acrop(self) -> _Plugin_acrop_Bound:
        """
        VapourSynth Auto Crop
        """
# end instance
# instance_bound: adg
    @property
    def adg(self) -> _Plugin_adg_Bound:
        """
        Adaptive grain
        """
# end instance
# instance_bound: average
    @property
    def average(self) -> _Plugin_average_Bound:
        """
        vs-average
        """
# end instance
# instance_bound: avisource
    @property
    def avisource(self) -> _Plugin_avisource_Bound:
        """
        VapourSynth AVISource Port
        """
# end instance
# instance_bound: avs
    @property
    def avs(self) -> _Plugin_avs_Bound:
        """
        VapourSynth Avisynth Compatibility
        """
# end instance
# instance_bound: avsr
    @property
    def avsr(self) -> _Plugin_avsr_Bound:
        """
        AviSynth Script Reader for VapourSynth v1.0.0
        """
# end instance
# instance_bound: avsw
    @property
    def avsw(self) -> _Plugin_avsw_Bound:
        """
        avsproxy
        """
# end instance
# instance_bound: bezier
    @property
    def bezier(self) -> _Plugin_bezier_Bound:
        """
        vapoursynth bezier curve test
        """
# end instance
# instance_bound: bifrost
    @property
    def bifrost(self) -> _Plugin_bifrost_Bound:
        """
        Bifrost plugin for VapourSynth
        """
# end instance
# instance_bound: bilateral
    @property
    def bilateral(self) -> _Plugin_bilateral_Bound:
        """
        Bilateral filter and Gaussian filter for VapourSynth.
        """
# end instance
# instance_bound: bm3d
    @property
    def bm3d(self) -> _Plugin_bm3d_Bound:
        """
        Implementation of BM3D denoising filter for VapourSynth.
        """
# end instance
# instance_bound: caffe
    @property
    def caffe(self) -> _Plugin_caffe_Bound:
        """
        Image Super-Resolution using Deep Convolutional Neural Networks
        """
# end instance
# instance_bound: cnr2
    @property
    def cnr2(self) -> _Plugin_cnr2_Bound:
        """
        Chroma noise reducer
        """
# end instance
# instance_bound: colorbars
    @property
    def colorbars(self) -> _Plugin_colorbars_Bound:
        """
        SMPTE RP 219-2:2016 and ITU-BT.2111 color bar generator for VapourSynth
        """
# end instance
# instance_bound: comb
    @property
    def comb(self) -> _Plugin_comb_Bound:
        """
        comb filters v0.0.1
        """
# end instance
# instance_bound: ctmf
    @property
    def ctmf(self) -> _Plugin_ctmf_Bound:
        """
        Constant Time Median Filtering
        """
# end instance
# instance_bound: d2v
    @property
    def d2v(self) -> _Plugin_d2v_Bound:
        """
        D2V Source
        """
# end instance
# instance_bound: damb
    @property
    def damb(self) -> _Plugin_damb_Bound:
        """
        Audio file reader and writer
        """
# end instance
# instance_bound: dct
    @property
    def dct(self) -> _Plugin_dct_Bound:
        """
        DCT filtering plugin rev10-b311e2e
        """
# end instance
# instance_bound: dctf
    @property
    def dctf(self) -> _Plugin_dctf_Bound:
        """
        DCT/IDCT Frequency Suppressor
        """
# end instance
# instance_bound: deblock
    @property
    def deblock(self) -> _Plugin_deblock_Bound:
        """
        It does a deblocking of the picture, using the deblocking filter of h264
        """
# end instance
# instance_bound: decross
    @property
    def decross(self) -> _Plugin_decross_Bound:
        """
        Spatio-temporal derainbow filter
        """
# end instance
# instance_bound: dedot
    @property
    def dedot(self) -> _Plugin_dedot_Bound:
        """
        Temporal dotcrawl and rainbow remover
        """
# end instance
# instance_bound: delogo
    @property
    def delogo(self) -> _Plugin_delogo_Bound:
        """
        VapourSynth Delogo Filter v005a.0.4
        """
# end instance
# instance_bound: delogohd
    @property
    def delogohd(self) -> _Plugin_delogohd_Bound:
        """
        VapourSynth DelogoHD Filter r9
        """
# end instance
# instance_bound: depan
    @property
    def depan(self) -> _Plugin_depan_Bound:
        """
        Tools for estimation and compensation of global motion (pan)
        """
# end instance
# instance_bound: descale
    @property
    def descale(self) -> _Plugin_descale_Bound:
        """
        Undo linear interpolation
        """
# end instance
# instance_bound: descale_getnative
    @property
    def descale_getnative(self) -> _Plugin_descale_getnative_Bound:
        """
        Undo linear interpolation
        """
# end instance
# instance_bound: dfttest
    @property
    def dfttest(self) -> _Plugin_dfttest_Bound:
        """
        2D/3D frequency domain denoiser
        """
# end instance
# instance_bound: dgdecodenv
    @property
    def dgdecodenv(self) -> _Plugin_dgdecodenv_Bound:
        """
        DGDecodeNV for VapourSynth
        """
# end instance
# instance_bound: dghdrtosdr
    @property
    def dghdrtosdr(self) -> _Plugin_dghdrtosdr_Bound:
        """
        Convert HDR to SDR
        """
# end instance
# instance_bound: dgm
    @property
    def dgm(self) -> _Plugin_dgm_Bound:
        """
        Spatio-temporal limited median denoiser
        """
# end instance
# instance_bound: dotkill
    @property
    def dotkill(self) -> _Plugin_dotkill_Bound:
        """
        VapourSynth DotKill
        """
# end instance
# instance_bound: eedi2
    @property
    def eedi2(self) -> _Plugin_eedi2_Bound:
        """
        EEDI2
        """
# end instance
# instance_bound: eedi3
    @property
    def eedi3(self) -> _Plugin_eedi3_Bound:
        """
        EEDI3
        """
# end instance
# instance_bound: eedi3m
    @property
    def eedi3m(self) -> _Plugin_eedi3m_Bound:
        """
        Enhanced Edge Directed Interpolation 3
        """
# end instance
# instance_bound: f3kdb
    @property
    def f3kdb(self) -> _Plugin_f3kdb_Bound:
        """
        flash3kyuu_deband
        """
# end instance
# instance_bound: fb
    @property
    def fb(self) -> _Plugin_fb_Bound:
        """
        FillBorders plugin for VapourSynth
        """
# end instance
# instance_bound: ffms2
    @property
    def ffms2(self) -> _Plugin_ffms2_Bound:
        """
        FFmpegSource 2 for VapourSynth
        """
# end instance
# instance_bound: fft3dfilter
    @property
    def fft3dfilter(self) -> _Plugin_fft3dfilter_Bound:
        """
        FFT3DFilter
        """
# end instance
# instance_bound: fftspectrum
    @property
    def fftspectrum(self) -> _Plugin_fftspectrum_Bound:
        """
        FFT Spectrum plugin
        """
# end instance
# instance_bound: fh
    @property
    def fh(self) -> _Plugin_fh_Bound:
        """
        FieldHint Plugin
        """
# end instance
# instance_bound: flux
    @property
    def flux(self) -> _Plugin_flux_Bound:
        """
        FluxSmooth plugin for VapourSynth
        """
# end instance
# instance_bound: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_Bound:
        """
        Format converter, r20
        """
# end instance
# instance_bound: focus
    @property
    def focus(self) -> _Plugin_focus_Bound:
        """
        VapourSynth Pixel Restoration Filters
        """
# end instance
# instance_bound: focus2
    @property
    def focus2(self) -> _Plugin_focus2_Bound:
        """
        VapourSynth TemporalSoften Filter v1
        """
# end instance
# instance_bound: ftf
    @property
    def ftf(self) -> _Plugin_ftf_Bound:
        """
        Fix Telecined Fades
        """
# end instance
# instance_bound: grad
    @property
    def grad(self) -> _Plugin_grad_Bound:
        """
        Adjustment of contrast, brightness, gamma and a wide range of color manipulations through gradation curves is possible.
        """
# end instance
# instance_bound: grain
    @property
    def grain(self) -> _Plugin_grain_Bound:
        """
        Add some correlated color gaussian noise
        """
# end instance
# instance_bound: hist
    @property
    def hist(self) -> _Plugin_hist_Bound:
        """
        VapourSynth Histogram Plugin
        """
# end instance
# instance_bound: hqdn3d
    @property
    def hqdn3d(self) -> _Plugin_hqdn3d_Bound:
        """
        HQDn3D port as used in avisynth/mplayer
        """
# end instance
# instance_bound: imwri
    @property
    def imwri(self) -> _Plugin_imwri_Bound:
        """
        VapourSynth ImageMagick 7 HDRI Writer/Reader
        """
# end instance
# instance_bound: it
    @property
    def it(self) -> _Plugin_it_Bound:
        """
        VapourSynth IVTC Filter v0103.1.1
        """
# end instance
# instance_bound: knlm
    @property
    def knlm(self) -> _Plugin_knlm_Bound:
        """
        KNLMeansCL for VapourSynth
        """
# end instance
# instance_bound: lsmas
    @property
    def lsmas(self) -> _Plugin_lsmas_Bound:
        """
        LSMASHSource for VapourSynth
        """
# end instance
# instance_bound: median
    @property
    def median(self) -> _Plugin_median_Bound:
        """
        Median of clips
        """
# end instance
# instance_bound: minideen
    @property
    def minideen(self) -> _Plugin_minideen_Bound:
        """
        Spatial convolution with thresholds
        """
# end instance
# instance_bound: minsrp
    @property
    def minsrp(self) -> _Plugin_minsrp_Bound:
        """
        VapourSynth MinSRP Filter
        """
# end instance
# instance_bound: misc
    @property
    def misc(self) -> _Plugin_misc_Bound:
        """
        Miscellaneous filters
        """
# end instance
# instance_bound: morpho
    @property
    def morpho(self) -> _Plugin_morpho_Bound:
        """
        Simple morphological filters.
        """
# end instance
# instance_bound: motionmask
    @property
    def motionmask(self) -> _Plugin_motionmask_Bound:
        """
        MotionMask creates a mask of moving pixels
        """
# end instance
# instance_bound: msmoosh
    @property
    def msmoosh(self) -> _Plugin_msmoosh_Bound:
        """
        MSmooth and MSharpen
        """
# end instance
# instance_bound: mv
    @property
    def mv(self) -> _Plugin_mv_Bound:
        """
        MVTools v20
        """
# end instance
# instance_bound: mvsf
    @property
    def mvsf(self) -> _Plugin_mvsf_Bound:
        """
        MVTools Single Precision
        """
# end instance
# instance_bound: neo_f3kdb
    @property
    def neo_f3kdb(self) -> _Plugin_neo_f3kdb_Bound:
        """
        Neo f3kdb
        """
# end instance
# instance_bound: nnedi3
    @property
    def nnedi3(self) -> _Plugin_nnedi3_Bound:
        """
        Neural network edge directed interpolation (3rd gen.), v12
        """
# end instance
# instance_bound: nnedi3cl
    @property
    def nnedi3cl(self) -> _Plugin_nnedi3cl_Bound:
        """
        An intra-field only deinterlacer
        """
# end instance
# instance_bound: noisegen
    @property
    def noisegen(self) -> _Plugin_noisegen_Bound:
        """
        VapourSynth Noise Generator
        """
# end instance
# instance_bound: placebo
    @property
    def placebo(self) -> _Plugin_placebo_Bound:
        """
        libplacebo plugin for VapourSynth
        """
# end instance
# instance_bound: pp7
    @property
    def pp7(self) -> _Plugin_pp7_Bound:
        """
        Postprocess 7 from MPlayer
        """
# end instance
# instance_bound: qr
    @property
    def qr(self) -> _Plugin_qr_Bound:
        """
        QR code filter
        """
# end instance
# instance_bound: raws
    @property
    def raws(self) -> _Plugin_raws_Bound:
        """
        Raw-format file Reader for VapourSynth 0.3.5
        """
# end instance
# instance_bound: rdvs
    @property
    def rdvs(self) -> _Plugin_rdvs_Bound:
        """
        RemoveDirt VapourSynth Port
        """
# end instance
# instance_bound: reduceflicker
    @property
    def reduceflicker(self) -> _Plugin_reduceflicker_Bound:
        """
        ReduceFlicker rev-
        """
# end instance
# instance_bound: resize
    @property
    def resize(self) -> _Plugin_resize_Bound:
        """
        VapourSynth Resize
        """
# end instance
# instance_bound: retinex
    @property
    def retinex(self) -> _Plugin_retinex_Bound:
        """
        Implementation of Retinex algorithm for VapourSynth.
        """
# end instance
# instance_bound: rf
    @property
    def rf(self) -> _Plugin_rf_Bound:
        """
        VapourSynth Replace Frames Tool
        """
# end instance
# instance_bound: rgsf
    @property
    def rgsf(self) -> _Plugin_rgsf_Bound:
        """
        RemoveGrain Single Precision
        """
# end instance
# instance_bound: rgvs
    @property
    def rgvs(self) -> _Plugin_rgvs_Bound:
        """
        RemoveGrain VapourSynth Port
        """
# end instance
# instance_bound: sangnom
    @property
    def sangnom(self) -> _Plugin_sangnom_Bound:
        """
        VapourSynth Single Field Deinterlacer
        """
# end instance
# instance_bound: scd
    @property
    def scd(self) -> _Plugin_scd_Bound:
        """
        Scene change detect filter for VapourSynth v0.2.0
        """
# end instance
# instance_bound: scrawl
    @property
    def scrawl(self) -> _Plugin_scrawl_Bound:
        """
        Simple text output plugin for VapourSynth
        """
# end instance
# instance_bound: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_Bound:
        """
        VapourSynth Scxvid Plugin
        """
# end instance
# instance_bound: smoothuv
    @property
    def smoothuv(self) -> _Plugin_smoothuv_Bound:
        """
        SmoothUV is a spatial derainbow filter
        """
# end instance
# instance_bound: ssiq
    @property
    def ssiq(self) -> _Plugin_ssiq_Bound:
        """
        SSIQ plugin for VapourSynth
        """
# end instance
# instance_bound: std
    @property
    def std(self) -> _Plugin_std_Bound:
        """
        VapourSynth Core Functions
        """
# end instance
# instance_bound: sub
    @property
    def sub(self) -> _Plugin_sub_Bound:
        """
        A subtitling filter based on libass and ffmpeg.
        """
# end instance
# instance_bound: surfaceblur
    @property
    def surfaceblur(self) -> _Plugin_surfaceblur_Bound:
        """
        surface blur
        """
# end instance
# instance_bound: svp1
    @property
    def svp1(self) -> _Plugin_svp1_Bound:
        """
        SVPFlow1
        """
# end instance
# instance_bound: svp2
    @property
    def svp2(self) -> _Plugin_svp2_Bound:
        """
        SVPFlow2
        """
# end instance
# instance_bound: tcanny
    @property
    def tcanny(self) -> _Plugin_tcanny_Bound:
        """
        Build an edge map using canny edge detection
        """
# end instance
# instance_bound: tcm
    @property
    def tcm(self) -> _Plugin_tcm_Bound:
        """
        TColorMask plugin for VapourSynth.
        """
# end instance
# instance_bound: tcomb
    @property
    def tcomb(self) -> _Plugin_tcomb_Bound:
        """
        Dotcrawl and rainbow remover
        """
# end instance
# instance_bound: tdm
    @property
    def tdm(self) -> _Plugin_tdm_Bound:
        """
        A bi-directionally motion adaptive deinterlacer
        """
# end instance
# instance_bound: tedgemask
    @property
    def tedgemask(self) -> _Plugin_tedgemask_Bound:
        """
        Edge detection plugin
        """
# end instance
# instance_bound: text
    @property
    def text(self) -> _Plugin_text_Bound:
        """
        VapourSynth Text
        """
# end instance
# instance_bound: timecube
    @property
    def timecube(self) -> _Plugin_timecube_Bound:
        """
        TimeCube 4D
        """
# end instance
# instance_bound: tivtc
    @property
    def tivtc(self) -> _Plugin_tivtc_Bound:
        """
        Field matching and decimation
        """
# end instance
# instance_bound: tla
    @property
    def tla(self) -> _Plugin_tla_Bound:
        """
        VapourSynth Temporal Linear Approximation plugin
        """
# end instance
# instance_bound: tmap
    @property
    def tmap(self) -> _Plugin_tmap_Bound:
        """
        Hable Tonemapping
        """
# end instance
# instance_bound: tmc
    @property
    def tmc(self) -> _Plugin_tmc_Bound:
        """
        A really simple mask cleaning plugin for VapourSynth based on mt_hysteresis.
        """
# end instance
# instance_bound: tmedian
    @property
    def tmedian(self) -> _Plugin_tmedian_Bound:
        """
        Calculates temporal median
        """
# end instance
# instance_bound: tnlm
    @property
    def tnlm(self) -> _Plugin_tnlm_Bound:
        """
        TNLMeans rev39-22a40af
        """
# end instance
# instance_bound: tonemap
    @property
    def tonemap(self) -> _Plugin_tonemap_Bound:
        """
        Simple tone mapping for VapourSynth
        """
# end instance
# instance_bound: trans
    @property
    def trans(self) -> _Plugin_trans_Bound:
        """
        VapourSynth transition plugin
        """
# end instance
# instance_bound: ttmpsm
    @property
    def ttmpsm(self) -> _Plugin_ttmpsm_Bound:
        """
        A basic, motion adaptive, temporal smoothing filter
        """
# end instance
# instance_bound: vcfreq
    @property
    def vcfreq(self) -> _Plugin_vcfreq_Bound:
        """
        VapourSynth  Frequency Domain Filters
        """
# end instance
# instance_bound: vcmod
    @property
    def vcmod(self) -> _Plugin_vcmod_Bound:
        """
        VapourSynth Pixel Amplitude modification 
        """
# end instance
# instance_bound: vcmove
    @property
    def vcmove(self) -> _Plugin_vcmove_Bound:
        """
        VapourSynth pixel repositioning Plugin
        """
# end instance
# instance_bound: vd
    @property
    def vd(self) -> _Plugin_vd_Bound:
        """
        A wavelet based denoiser
        """
# end instance
# instance_bound: vfrtocfr
    @property
    def vfrtocfr(self) -> _Plugin_vfrtocfr_Bound:
        """
        VFR to CFR Video Converter
        """
# end instance
# instance_bound: vinverse
    @property
    def vinverse(self) -> _Plugin_vinverse_Bound:
        """
        A simple filter to remove residual combing.
        """
# end instance
# instance_bound: vivtc
    @property
    def vivtc(self) -> _Plugin_vivtc_Bound:
        """
        VFM
        """
# end instance
# instance_bound: vscope
    @property
    def vscope(self) -> _Plugin_vscope_Bound:
        """
        Videoscope plugin for VapourSynth
        """
# end instance
# instance_bound: vsf
    @property
    def vsf(self) -> _Plugin_vsf_Bound:
        """
        VSFilter
        """
# end instance
# instance_bound: vsfm
    @property
    def vsfm(self) -> _Plugin_vsfm_Bound:
        """
        VSFilterMod
        """
# end instance
# instance_bound: w2xc
    @property
    def w2xc(self) -> _Plugin_w2xc_Bound:
        """
        Image Super-Resolution using Deep Convolutional Neural Networks
        """
# end instance
# instance_bound: w2xnvk
    @property
    def w2xnvk(self) -> _Plugin_w2xnvk_Bound:
        """
        VapourSynth Waifu2x NCNN Vulkan Plugin
        """
# end instance
# instance_bound: w3fdif
    @property
    def w3fdif(self) -> _Plugin_w3fdif_Bound:
        """
        Weston 3 Field Deinterlacing Filter
        """
# end instance
# instance_bound: waifu2x
    @property
    def waifu2x(self) -> _Plugin_waifu2x_Bound:
        """
        Waifu2x-opt image restoration filter for VapourSynth.
        """
# end instance
# instance_bound: warp
    @property
    def warp(self) -> _Plugin_warp_Bound:
        """
        Sharpen images by warping
        """
# end instance
# instance_bound: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_Bound:
        """
        Scene change detection approximately like Xvid's
        """
# end instance
# instance_bound: xyvsf
    @property
    def xyvsf(self) -> _Plugin_xyvsf_Bound:
        """
        xy-VSFilter
        """
# end instance
# instance_bound: yadifmod
    @property
    def yadifmod(self) -> _Plugin_yadifmod_Bound:
        """
        Modification of Fizick's yadif avisynth filter
        """
# end instance
# instance_bound: znedi3
    @property
    def znedi3(self) -> _Plugin_znedi3_Bound:
        """
        Neural network edge directed interpolation (3rd gen.)
        """
# end instance

    format: typing.Optional[Format]

    fps: fractions.Fraction
    fps_den: int
    fps_num: int

    height: int
    width: int

    num_frames: int
    flags: int

    def get_frame(self, n: int) -> VideoFrame: ...
    def get_frame_async_raw(self, n: int, cb: _Future[VideoFrame], future_wrapper: typing.Optional[typing.Callable[..., None]]=None) -> _Future[VideoFrame]: ...
    def get_frame_async(self, n: int) -> _Future[VideoFrame]: ...

    def set_output(self, index: int=0, alpha: typing.Optional['VideoNode']=None) -> None: ...
    def output(self, fileobj: typing.BinaryIO, y4m: bool = False, progress_update: typing.Optional[typing.Callable[[int, int], None]]=None, prefetch: int = 0) -> None: ...

    def frames(self) -> typing.Iterator[VideoFrame]: ...

    def __add__(self, other: 'VideoNode') -> 'VideoNode': ...
    def __mul__(self, other: int) -> 'VideoNode': ...
    def __getitem__(self, other: typing.Union[int, slice]) -> 'VideoNode': ...
    def __len__(self) -> int: ...


class _PluginMeta(typing.TypedDict):
    namespace: str
    identifier: str
    name: str
    functions: typing.Dict[str, str]


class Core:
# instance_unbound: DGMVC
    @property
    def DGMVC(self) -> _Plugin_DGMVC_Unbound:
        """
        DGMVCSource for VapourSynth
        """
# end instance
# instance_unbound: acrop
    @property
    def acrop(self) -> _Plugin_acrop_Unbound:
        """
        VapourSynth Auto Crop
        """
# end instance
# instance_unbound: adg
    @property
    def adg(self) -> _Plugin_adg_Unbound:
        """
        Adaptive grain
        """
# end instance
# instance_unbound: average
    @property
    def average(self) -> _Plugin_average_Unbound:
        """
        vs-average
        """
# end instance
# instance_unbound: avisource
    @property
    def avisource(self) -> _Plugin_avisource_Unbound:
        """
        VapourSynth AVISource Port
        """
# end instance
# instance_unbound: avs
    @property
    def avs(self) -> _Plugin_avs_Unbound:
        """
        VapourSynth Avisynth Compatibility
        """
# end instance
# instance_unbound: avsr
    @property
    def avsr(self) -> _Plugin_avsr_Unbound:
        """
        AviSynth Script Reader for VapourSynth v1.0.0
        """
# end instance
# instance_unbound: avsw
    @property
    def avsw(self) -> _Plugin_avsw_Unbound:
        """
        avsproxy
        """
# end instance
# instance_unbound: bezier
    @property
    def bezier(self) -> _Plugin_bezier_Unbound:
        """
        vapoursynth bezier curve test
        """
# end instance
# instance_unbound: bifrost
    @property
    def bifrost(self) -> _Plugin_bifrost_Unbound:
        """
        Bifrost plugin for VapourSynth
        """
# end instance
# instance_unbound: bilateral
    @property
    def bilateral(self) -> _Plugin_bilateral_Unbound:
        """
        Bilateral filter and Gaussian filter for VapourSynth.
        """
# end instance
# instance_unbound: bm3d
    @property
    def bm3d(self) -> _Plugin_bm3d_Unbound:
        """
        Implementation of BM3D denoising filter for VapourSynth.
        """
# end instance
# instance_unbound: caffe
    @property
    def caffe(self) -> _Plugin_caffe_Unbound:
        """
        Image Super-Resolution using Deep Convolutional Neural Networks
        """
# end instance
# instance_unbound: cnr2
    @property
    def cnr2(self) -> _Plugin_cnr2_Unbound:
        """
        Chroma noise reducer
        """
# end instance
# instance_unbound: colorbars
    @property
    def colorbars(self) -> _Plugin_colorbars_Unbound:
        """
        SMPTE RP 219-2:2016 and ITU-BT.2111 color bar generator for VapourSynth
        """
# end instance
# instance_unbound: comb
    @property
    def comb(self) -> _Plugin_comb_Unbound:
        """
        comb filters v0.0.1
        """
# end instance
# instance_unbound: ctmf
    @property
    def ctmf(self) -> _Plugin_ctmf_Unbound:
        """
        Constant Time Median Filtering
        """
# end instance
# instance_unbound: d2v
    @property
    def d2v(self) -> _Plugin_d2v_Unbound:
        """
        D2V Source
        """
# end instance
# instance_unbound: damb
    @property
    def damb(self) -> _Plugin_damb_Unbound:
        """
        Audio file reader and writer
        """
# end instance
# instance_unbound: dct
    @property
    def dct(self) -> _Plugin_dct_Unbound:
        """
        DCT filtering plugin rev10-b311e2e
        """
# end instance
# instance_unbound: dctf
    @property
    def dctf(self) -> _Plugin_dctf_Unbound:
        """
        DCT/IDCT Frequency Suppressor
        """
# end instance
# instance_unbound: deblock
    @property
    def deblock(self) -> _Plugin_deblock_Unbound:
        """
        It does a deblocking of the picture, using the deblocking filter of h264
        """
# end instance
# instance_unbound: decross
    @property
    def decross(self) -> _Plugin_decross_Unbound:
        """
        Spatio-temporal derainbow filter
        """
# end instance
# instance_unbound: dedot
    @property
    def dedot(self) -> _Plugin_dedot_Unbound:
        """
        Temporal dotcrawl and rainbow remover
        """
# end instance
# instance_unbound: delogo
    @property
    def delogo(self) -> _Plugin_delogo_Unbound:
        """
        VapourSynth Delogo Filter v005a.0.4
        """
# end instance
# instance_unbound: delogohd
    @property
    def delogohd(self) -> _Plugin_delogohd_Unbound:
        """
        VapourSynth DelogoHD Filter r9
        """
# end instance
# instance_unbound: depan
    @property
    def depan(self) -> _Plugin_depan_Unbound:
        """
        Tools for estimation and compensation of global motion (pan)
        """
# end instance
# instance_unbound: descale
    @property
    def descale(self) -> _Plugin_descale_Unbound:
        """
        Undo linear interpolation
        """
# end instance
# instance_unbound: descale_getnative
    @property
    def descale_getnative(self) -> _Plugin_descale_getnative_Unbound:
        """
        Undo linear interpolation
        """
# end instance
# instance_unbound: dfttest
    @property
    def dfttest(self) -> _Plugin_dfttest_Unbound:
        """
        2D/3D frequency domain denoiser
        """
# end instance
# instance_unbound: dgdecodenv
    @property
    def dgdecodenv(self) -> _Plugin_dgdecodenv_Unbound:
        """
        DGDecodeNV for VapourSynth
        """
# end instance
# instance_unbound: dghdrtosdr
    @property
    def dghdrtosdr(self) -> _Plugin_dghdrtosdr_Unbound:
        """
        Convert HDR to SDR
        """
# end instance
# instance_unbound: dgm
    @property
    def dgm(self) -> _Plugin_dgm_Unbound:
        """
        Spatio-temporal limited median denoiser
        """
# end instance
# instance_unbound: dotkill
    @property
    def dotkill(self) -> _Plugin_dotkill_Unbound:
        """
        VapourSynth DotKill
        """
# end instance
# instance_unbound: eedi2
    @property
    def eedi2(self) -> _Plugin_eedi2_Unbound:
        """
        EEDI2
        """
# end instance
# instance_unbound: eedi3
    @property
    def eedi3(self) -> _Plugin_eedi3_Unbound:
        """
        EEDI3
        """
# end instance
# instance_unbound: eedi3m
    @property
    def eedi3m(self) -> _Plugin_eedi3m_Unbound:
        """
        Enhanced Edge Directed Interpolation 3
        """
# end instance
# instance_unbound: f3kdb
    @property
    def f3kdb(self) -> _Plugin_f3kdb_Unbound:
        """
        flash3kyuu_deband
        """
# end instance
# instance_unbound: fb
    @property
    def fb(self) -> _Plugin_fb_Unbound:
        """
        FillBorders plugin for VapourSynth
        """
# end instance
# instance_unbound: ffms2
    @property
    def ffms2(self) -> _Plugin_ffms2_Unbound:
        """
        FFmpegSource 2 for VapourSynth
        """
# end instance
# instance_unbound: fft3dfilter
    @property
    def fft3dfilter(self) -> _Plugin_fft3dfilter_Unbound:
        """
        FFT3DFilter
        """
# end instance
# instance_unbound: fftspectrum
    @property
    def fftspectrum(self) -> _Plugin_fftspectrum_Unbound:
        """
        FFT Spectrum plugin
        """
# end instance
# instance_unbound: fh
    @property
    def fh(self) -> _Plugin_fh_Unbound:
        """
        FieldHint Plugin
        """
# end instance
# instance_unbound: flux
    @property
    def flux(self) -> _Plugin_flux_Unbound:
        """
        FluxSmooth plugin for VapourSynth
        """
# end instance
# instance_unbound: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_Unbound:
        """
        Format converter, r20
        """
# end instance
# instance_unbound: focus
    @property
    def focus(self) -> _Plugin_focus_Unbound:
        """
        VapourSynth Pixel Restoration Filters
        """
# end instance
# instance_unbound: focus2
    @property
    def focus2(self) -> _Plugin_focus2_Unbound:
        """
        VapourSynth TemporalSoften Filter v1
        """
# end instance
# instance_unbound: ftf
    @property
    def ftf(self) -> _Plugin_ftf_Unbound:
        """
        Fix Telecined Fades
        """
# end instance
# instance_unbound: grad
    @property
    def grad(self) -> _Plugin_grad_Unbound:
        """
        Adjustment of contrast, brightness, gamma and a wide range of color manipulations through gradation curves is possible.
        """
# end instance
# instance_unbound: grain
    @property
    def grain(self) -> _Plugin_grain_Unbound:
        """
        Add some correlated color gaussian noise
        """
# end instance
# instance_unbound: hist
    @property
    def hist(self) -> _Plugin_hist_Unbound:
        """
        VapourSynth Histogram Plugin
        """
# end instance
# instance_unbound: hqdn3d
    @property
    def hqdn3d(self) -> _Plugin_hqdn3d_Unbound:
        """
        HQDn3D port as used in avisynth/mplayer
        """
# end instance
# instance_unbound: imwri
    @property
    def imwri(self) -> _Plugin_imwri_Unbound:
        """
        VapourSynth ImageMagick 7 HDRI Writer/Reader
        """
# end instance
# instance_unbound: it
    @property
    def it(self) -> _Plugin_it_Unbound:
        """
        VapourSynth IVTC Filter v0103.1.1
        """
# end instance
# instance_unbound: knlm
    @property
    def knlm(self) -> _Plugin_knlm_Unbound:
        """
        KNLMeansCL for VapourSynth
        """
# end instance
# instance_unbound: lsmas
    @property
    def lsmas(self) -> _Plugin_lsmas_Unbound:
        """
        LSMASHSource for VapourSynth
        """
# end instance
# instance_unbound: median
    @property
    def median(self) -> _Plugin_median_Unbound:
        """
        Median of clips
        """
# end instance
# instance_unbound: minideen
    @property
    def minideen(self) -> _Plugin_minideen_Unbound:
        """
        Spatial convolution with thresholds
        """
# end instance
# instance_unbound: minsrp
    @property
    def minsrp(self) -> _Plugin_minsrp_Unbound:
        """
        VapourSynth MinSRP Filter
        """
# end instance
# instance_unbound: misc
    @property
    def misc(self) -> _Plugin_misc_Unbound:
        """
        Miscellaneous filters
        """
# end instance
# instance_unbound: morpho
    @property
    def morpho(self) -> _Plugin_morpho_Unbound:
        """
        Simple morphological filters.
        """
# end instance
# instance_unbound: motionmask
    @property
    def motionmask(self) -> _Plugin_motionmask_Unbound:
        """
        MotionMask creates a mask of moving pixels
        """
# end instance
# instance_unbound: msmoosh
    @property
    def msmoosh(self) -> _Plugin_msmoosh_Unbound:
        """
        MSmooth and MSharpen
        """
# end instance
# instance_unbound: mv
    @property
    def mv(self) -> _Plugin_mv_Unbound:
        """
        MVTools v20
        """
# end instance
# instance_unbound: mvsf
    @property
    def mvsf(self) -> _Plugin_mvsf_Unbound:
        """
        MVTools Single Precision
        """
# end instance
# instance_unbound: neo_f3kdb
    @property
    def neo_f3kdb(self) -> _Plugin_neo_f3kdb_Unbound:
        """
        Neo f3kdb
        """
# end instance
# instance_unbound: nnedi3
    @property
    def nnedi3(self) -> _Plugin_nnedi3_Unbound:
        """
        Neural network edge directed interpolation (3rd gen.), v12
        """
# end instance
# instance_unbound: nnedi3cl
    @property
    def nnedi3cl(self) -> _Plugin_nnedi3cl_Unbound:
        """
        An intra-field only deinterlacer
        """
# end instance
# instance_unbound: noisegen
    @property
    def noisegen(self) -> _Plugin_noisegen_Unbound:
        """
        VapourSynth Noise Generator
        """
# end instance
# instance_unbound: placebo
    @property
    def placebo(self) -> _Plugin_placebo_Unbound:
        """
        libplacebo plugin for VapourSynth
        """
# end instance
# instance_unbound: pp7
    @property
    def pp7(self) -> _Plugin_pp7_Unbound:
        """
        Postprocess 7 from MPlayer
        """
# end instance
# instance_unbound: qr
    @property
    def qr(self) -> _Plugin_qr_Unbound:
        """
        QR code filter
        """
# end instance
# instance_unbound: raws
    @property
    def raws(self) -> _Plugin_raws_Unbound:
        """
        Raw-format file Reader for VapourSynth 0.3.5
        """
# end instance
# instance_unbound: rdvs
    @property
    def rdvs(self) -> _Plugin_rdvs_Unbound:
        """
        RemoveDirt VapourSynth Port
        """
# end instance
# instance_unbound: reduceflicker
    @property
    def reduceflicker(self) -> _Plugin_reduceflicker_Unbound:
        """
        ReduceFlicker rev-
        """
# end instance
# instance_unbound: resize
    @property
    def resize(self) -> _Plugin_resize_Unbound:
        """
        VapourSynth Resize
        """
# end instance
# instance_unbound: retinex
    @property
    def retinex(self) -> _Plugin_retinex_Unbound:
        """
        Implementation of Retinex algorithm for VapourSynth.
        """
# end instance
# instance_unbound: rf
    @property
    def rf(self) -> _Plugin_rf_Unbound:
        """
        VapourSynth Replace Frames Tool
        """
# end instance
# instance_unbound: rgsf
    @property
    def rgsf(self) -> _Plugin_rgsf_Unbound:
        """
        RemoveGrain Single Precision
        """
# end instance
# instance_unbound: rgvs
    @property
    def rgvs(self) -> _Plugin_rgvs_Unbound:
        """
        RemoveGrain VapourSynth Port
        """
# end instance
# instance_unbound: sangnom
    @property
    def sangnom(self) -> _Plugin_sangnom_Unbound:
        """
        VapourSynth Single Field Deinterlacer
        """
# end instance
# instance_unbound: scd
    @property
    def scd(self) -> _Plugin_scd_Unbound:
        """
        Scene change detect filter for VapourSynth v0.2.0
        """
# end instance
# instance_unbound: scrawl
    @property
    def scrawl(self) -> _Plugin_scrawl_Unbound:
        """
        Simple text output plugin for VapourSynth
        """
# end instance
# instance_unbound: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_Unbound:
        """
        VapourSynth Scxvid Plugin
        """
# end instance
# instance_unbound: smoothuv
    @property
    def smoothuv(self) -> _Plugin_smoothuv_Unbound:
        """
        SmoothUV is a spatial derainbow filter
        """
# end instance
# instance_unbound: ssiq
    @property
    def ssiq(self) -> _Plugin_ssiq_Unbound:
        """
        SSIQ plugin for VapourSynth
        """
# end instance
# instance_unbound: std
    @property
    def std(self) -> _Plugin_std_Unbound:
        """
        VapourSynth Core Functions
        """
# end instance
# instance_unbound: sub
    @property
    def sub(self) -> _Plugin_sub_Unbound:
        """
        A subtitling filter based on libass and ffmpeg.
        """
# end instance
# instance_unbound: surfaceblur
    @property
    def surfaceblur(self) -> _Plugin_surfaceblur_Unbound:
        """
        surface blur
        """
# end instance
# instance_unbound: svp1
    @property
    def svp1(self) -> _Plugin_svp1_Unbound:
        """
        SVPFlow1
        """
# end instance
# instance_unbound: svp2
    @property
    def svp2(self) -> _Plugin_svp2_Unbound:
        """
        SVPFlow2
        """
# end instance
# instance_unbound: tcanny
    @property
    def tcanny(self) -> _Plugin_tcanny_Unbound:
        """
        Build an edge map using canny edge detection
        """
# end instance
# instance_unbound: tcm
    @property
    def tcm(self) -> _Plugin_tcm_Unbound:
        """
        TColorMask plugin for VapourSynth.
        """
# end instance
# instance_unbound: tcomb
    @property
    def tcomb(self) -> _Plugin_tcomb_Unbound:
        """
        Dotcrawl and rainbow remover
        """
# end instance
# instance_unbound: tdm
    @property
    def tdm(self) -> _Plugin_tdm_Unbound:
        """
        A bi-directionally motion adaptive deinterlacer
        """
# end instance
# instance_unbound: tedgemask
    @property
    def tedgemask(self) -> _Plugin_tedgemask_Unbound:
        """
        Edge detection plugin
        """
# end instance
# instance_unbound: text
    @property
    def text(self) -> _Plugin_text_Unbound:
        """
        VapourSynth Text
        """
# end instance
# instance_unbound: timecube
    @property
    def timecube(self) -> _Plugin_timecube_Unbound:
        """
        TimeCube 4D
        """
# end instance
# instance_unbound: tivtc
    @property
    def tivtc(self) -> _Plugin_tivtc_Unbound:
        """
        Field matching and decimation
        """
# end instance
# instance_unbound: tla
    @property
    def tla(self) -> _Plugin_tla_Unbound:
        """
        VapourSynth Temporal Linear Approximation plugin
        """
# end instance
# instance_unbound: tmap
    @property
    def tmap(self) -> _Plugin_tmap_Unbound:
        """
        Hable Tonemapping
        """
# end instance
# instance_unbound: tmc
    @property
    def tmc(self) -> _Plugin_tmc_Unbound:
        """
        A really simple mask cleaning plugin for VapourSynth based on mt_hysteresis.
        """
# end instance
# instance_unbound: tmedian
    @property
    def tmedian(self) -> _Plugin_tmedian_Unbound:
        """
        Calculates temporal median
        """
# end instance
# instance_unbound: tnlm
    @property
    def tnlm(self) -> _Plugin_tnlm_Unbound:
        """
        TNLMeans rev39-22a40af
        """
# end instance
# instance_unbound: tonemap
    @property
    def tonemap(self) -> _Plugin_tonemap_Unbound:
        """
        Simple tone mapping for VapourSynth
        """
# end instance
# instance_unbound: trans
    @property
    def trans(self) -> _Plugin_trans_Unbound:
        """
        VapourSynth transition plugin
        """
# end instance
# instance_unbound: ttmpsm
    @property
    def ttmpsm(self) -> _Plugin_ttmpsm_Unbound:
        """
        A basic, motion adaptive, temporal smoothing filter
        """
# end instance
# instance_unbound: vcfreq
    @property
    def vcfreq(self) -> _Plugin_vcfreq_Unbound:
        """
        VapourSynth  Frequency Domain Filters
        """
# end instance
# instance_unbound: vcmod
    @property
    def vcmod(self) -> _Plugin_vcmod_Unbound:
        """
        VapourSynth Pixel Amplitude modification 
        """
# end instance
# instance_unbound: vcmove
    @property
    def vcmove(self) -> _Plugin_vcmove_Unbound:
        """
        VapourSynth pixel repositioning Plugin
        """
# end instance
# instance_unbound: vd
    @property
    def vd(self) -> _Plugin_vd_Unbound:
        """
        A wavelet based denoiser
        """
# end instance
# instance_unbound: vfrtocfr
    @property
    def vfrtocfr(self) -> _Plugin_vfrtocfr_Unbound:
        """
        VFR to CFR Video Converter
        """
# end instance
# instance_unbound: vinverse
    @property
    def vinverse(self) -> _Plugin_vinverse_Unbound:
        """
        A simple filter to remove residual combing.
        """
# end instance
# instance_unbound: vivtc
    @property
    def vivtc(self) -> _Plugin_vivtc_Unbound:
        """
        VFM
        """
# end instance
# instance_unbound: vscope
    @property
    def vscope(self) -> _Plugin_vscope_Unbound:
        """
        Videoscope plugin for VapourSynth
        """
# end instance
# instance_unbound: vsf
    @property
    def vsf(self) -> _Plugin_vsf_Unbound:
        """
        VSFilter
        """
# end instance
# instance_unbound: vsfm
    @property
    def vsfm(self) -> _Plugin_vsfm_Unbound:
        """
        VSFilterMod
        """
# end instance
# instance_unbound: w2xc
    @property
    def w2xc(self) -> _Plugin_w2xc_Unbound:
        """
        Image Super-Resolution using Deep Convolutional Neural Networks
        """
# end instance
# instance_unbound: w2xnvk
    @property
    def w2xnvk(self) -> _Plugin_w2xnvk_Unbound:
        """
        VapourSynth Waifu2x NCNN Vulkan Plugin
        """
# end instance
# instance_unbound: w3fdif
    @property
    def w3fdif(self) -> _Plugin_w3fdif_Unbound:
        """
        Weston 3 Field Deinterlacing Filter
        """
# end instance
# instance_unbound: waifu2x
    @property
    def waifu2x(self) -> _Plugin_waifu2x_Unbound:
        """
        Waifu2x-opt image restoration filter for VapourSynth.
        """
# end instance
# instance_unbound: warp
    @property
    def warp(self) -> _Plugin_warp_Unbound:
        """
        Sharpen images by warping
        """
# end instance
# instance_unbound: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_Unbound:
        """
        Scene change detection approximately like Xvid's
        """
# end instance
# instance_unbound: xyvsf
    @property
    def xyvsf(self) -> _Plugin_xyvsf_Unbound:
        """
        xy-VSFilter
        """
# end instance
# instance_unbound: yadifmod
    @property
    def yadifmod(self) -> _Plugin_yadifmod_Unbound:
        """
        Modification of Fizick's yadif avisynth filter
        """
# end instance
# instance_unbound: znedi3
    @property
    def znedi3(self) -> _Plugin_znedi3_Unbound:
        """
        Neural network edge directed interpolation (3rd gen.)
        """
# end instance

    num_threads: int
    max_cache_size: int
    add_cache: bool

    def set_max_cache_size(self, mb: int) -> int: ...
    def get_plugins(self) -> typing.Dict[str, _PluginMeta]: ...
    def list_functions(self) -> str: ...

    def register_format(self, color_family: ColorFamily, sample_type: SampleType, bits_per_sample: int, subsampling_w: int, subsampling_h: int) -> Format: ...
    def get_format(self, id: typing.Union[Format, int, PresetFormat]) -> Format: ...

    def version(self) -> str: ...
    def version_number(self) -> int: ...


def get_core(threads: typing.Optional[int]=None, add_cache: typing.Optional[bool]=None) -> Core: ...


class _CoreProxy(Core):
    core: Core


core: _CoreProxy
