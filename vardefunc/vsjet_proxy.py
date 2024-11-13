import inspect
import logging

from fractions import Fraction
from functools import lru_cache
from typing import Any, Iterable, Protocol, Sequence, TypeVar, Union, overload

from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import Point
from vsmasktools import DeferredMask as vsmasktools_DeferredMask
from vsmasktools import GenericMaskT
from vsmasktools import HardsubASS as vsmasktools_HardsubASS
from vsmasktools import HardsubLine as vsmasktools_HardsubLine
from vsmasktools import HardsubLineFade as vsmasktools_HardsubLineFade
from vsmasktools import HardsubMask as vsmasktools_HardsubMask
from vsmasktools import HardsubSign as vsmasktools_HardsubSign
from vsmasktools import HardsubSignFades as vsmasktools_HardsubSignFades
from vsmasktools import Morpho, SobelStd, XxpandMode, normalize_mask
from vsrgtools.util import mean_matrix
from vstools import ColorRange, FrameRangeN, FrameRangesN, Keyframes, KwargsT, core, flatten
from vstools import replace_ranges as vstools_replace_ranges
from vstools import scale_value, to_arr, vs

from .types import AnyPath, Range, RangeN
from .util import normalise_ranges

__all__ = [
    "is_preview", "set_output", "replace_ranges", "BestestSource",
    "DeferredMask", "HardsubASS", "HardsubLine", "HardsubLineFade", "HardsubMask",
    "HardsubSign", "HardsubSignFades"
]


@lru_cache
def is_preview() -> bool:
    try:
        import vspreview.api
    except ImportError:
        is_preview = False
    else:
        is_preview = vspreview.api.is_preview()
    return is_preview


TimecodesT = AnyPath | dict[RangeN, float | Range | Fraction] | list[Fraction] | None
ScenesT = Keyframes | list[Range] | list[Keyframes | list[Range]] | None


# VideoNode signature
@overload
def set_output(
    node: vs.VideoNode,
    index: int = ...,
    /,
    *,
    alpha: vs.VideoNode | None = ...,
    timecodes: TimecodesT = None, denominator: int = 1001, scenes: ScenesT = None,
    **kwargs: Any
) -> None:
    ...


@overload
def set_output(
    node: vs.VideoNode,
    name: str | bool | None = ...,
    /,
    *,
    alpha: vs.VideoNode | None = ...,
    timecodes: TimecodesT = None, denominator: int = 1001, scenes: ScenesT = None,
    **kwargs: Any
) -> None:
    ...


@overload
def set_output(
    node: vs.VideoNode,
    index: int = ..., name: str | bool | None = ...,
    /,
    alpha: vs.VideoNode | None = ...,
    *,
    timecodes: TimecodesT = None, denominator: int = 1001, scenes: ScenesT = None,
    **kwargs: Any
) -> None:
    ...


# AudioNode signature
@overload
def set_output(
    node: vs.AudioNode,
    index: int = ...,
    /,
    **kwargs: Any
) -> None:
    ...

@overload
def set_output(
    node: vs.AudioNode,
    name: str | bool | None = ...,
    /,
    **kwargs: Any
) -> None:
    ...

@overload
def set_output(
    node: vs.AudioNode,
    index: int = ..., name: str | bool | None = ...,
    /,
    **kwargs: Any
) -> None:
    ...


# Iterable of VideoNode signature
@overload
def set_output(
    node: Iterable[vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]],
    index: int | Sequence[int] = ...,
    /,
    **kwargs: Any
) -> None:
    ...


@overload
def set_output(
    node: Iterable[vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]],
    name: str | bool | None = ...,
    /,
    **kwargs: Any
) -> None:
    ...


@overload
def set_output(
    node: Iterable[vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]],
    index: int | Sequence[int] = ..., name: str | bool | None = ...,
    /,
    **kwargs: Any
) -> None:
    ...

# Iterable of AudioNode signature
@overload
def set_output(
    node: Iterable[vs.AudioNode | Iterable[vs.AudioNode | Iterable[vs.AudioNode]]],
    index: int | Sequence[int] = ...,
    /,
    **kwargs: Any
) -> None:
    ...


@overload
def set_output(
    node: Iterable[vs.AudioNode | Iterable[vs.AudioNode | Iterable[vs.AudioNode]]],
    name: str | bool | None = ...,
    /,
    **kwargs: Any
) -> None:
    ...


@overload
def set_output(
    node: Iterable[vs.AudioNode | Iterable[vs.AudioNode | Iterable[vs.AudioNode]]],
    index: int | Sequence[int] = ..., name: str | bool | None = ...,
    /,
    **kwargs: Any
) -> None:
    ...


def set_output(
    node: vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode]]],
    index_or_name: int | Sequence[int] | str | bool | None = None, name: str | bool | None = None,
    /,
    alpha: vs.VideoNode | None = None,
    *,
    timecodes: TimecodesT = None, denominator: int = 1001, scenes: ScenesT = None,
    **kwargs: Any
) -> None:
    if not is_preview():
        return None

    from vspreview.api import set_scening, set_timecodes, update_node_info

    if isinstance(index_or_name, (str, bool)):
        index = None
        name = index_or_name
    else:
        index = index_or_name

    ouputs = vs.get_outputs()
    nodes = list(flatten(node))

    index = to_arr(index) if index is not None else [max(ouputs, default=-1) + 1]

    while len(index) < len(nodes):
        index.append(index[-1] + 1)

    for i, n in zip(index[:len(nodes)], nodes):
        if i in ouputs:
            logging.warn(f"Index n° {i} has been already used!")
        if isinstance(n, vs.VideoNode):
            n.set_output(i, alpha)
            title = 'Clip'
        else:
            n.set_output(i)
            title = 'Audio' if isinstance(n, vs.AudioNode) else 'Node'

        if (not name and name is not False) or name is True:
            name = f"{title} {i}"

            current_frame = inspect.currentframe()

            assert current_frame
            assert current_frame.f_back

            ref_id = str(id(n))
            for vname, val in reversed(current_frame.f_back.f_locals.items()):
                if (str(id(val)) == ref_id):
                    name = vname
                    break

            del current_frame

        update_node_info(
            type(n), i,
            **KwargsT(cache=True, disable_comp=False) | (KwargsT(name=name) if name else {}) | kwargs
        )

        if isinstance(n, vs.VideoNode):
            if timecodes:
                timecodes = str(timecodes) if not isinstance(timecodes, (dict, list)) else timecodes
                set_timecodes(i, timecodes, n, denominator)

            if scenes:
                set_scening(scenes, n, name or f'Clip {i}')


_VideoFrameT_contra = TypeVar("_VideoFrameT_contra", vs.VideoFrame, list[vs.VideoFrame], contravariant=True)

class RangesCallBack(Protocol):
    def __call__(self, n: int) -> bool:
        ...

class RangesCallBackF(Protocol[_VideoFrameT_contra]):
    def __call__(self, f: _VideoFrameT_contra) -> bool:
        ...

class RangesCallBackNF(Protocol[_VideoFrameT_contra]):
    def __call__(self, n: int, f: _VideoFrameT_contra) -> bool:
        ...

RangesCallBackT = Union[
    RangesCallBack,
    RangesCallBackF[vs.VideoFrame],
    RangesCallBackNF[vs.VideoFrame],
    RangesCallBackF[list[vs.VideoFrame]],
    RangesCallBackNF[list[vs.VideoFrame]],
]

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN,
    *,
    exclusive: bool = True, mismatch: bool = False,
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: RangesCallBack,
    *,
    mismatch: bool = False,
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: RangesCallBackF[vs.VideoFrame] | RangesCallBackNF[vs.VideoFrame],
    *,
    mismatch: bool = False,
    prop_src: vs.VideoNode
) -> vs.VideoNode:
    ...

@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: RangesCallBackF[list[vs.VideoFrame]] | RangesCallBackNF[list[vs.VideoFrame]],
    *,
    mismatch: bool = False,
    prop_src: list[vs.VideoNode]
) -> vs.VideoNode:
    ...

def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | RangesCallBackT | None,
    *,
    exclusive: bool = True, mismatch: bool = False,
    prop_src: vs.VideoNode | list[vs.VideoNode] | None = None
) -> vs.VideoNode:
    """
    Replaces frames in a clip, either with pre-calculated indices or on-the-fly with a callback.
    Frame ranges are by default exclusive. This behaviour can be changed by setting `exclusive=False`.

    Examples with clips ``black`` and ``white`` of equal length:
        * ``replace_ranges(black, white, [(0, 1)])``: replace frames 0 with ``white``
        * ``replace_ranges(black, white, [(0, 2)])``: replace frames 0 and 1 with ``white``
        * ``replace_ranges(black, white, [(None, None)])``: replace the entire clip with ``white``
        * ``replace_ranges(black, white, [(0, None)])``: same as previous
        * ``replace_ranges(black, white, [(200, None)])``: replace 200 until the end with ``white``
        * ``replace_ranges(black, white, [(200, -1)])``: replace 200 until the end with ``white``,
                                                         leaving 1 frame of ``black``

    Optional Dependencies:
        * Either of the following two plugins:
            * `VS Julek Plugin <https://github.com/dnjulek/vapoursynth-julek-plugin>`_ (recommended!)
            * `VSRemapFrames <https://github.com/Irrational-Encoding-Wizardry/Vapoursynth-RemapFrames>`_

    :param clip_a:      Original clip.
    :param clip_b:      Replacement clip.
    :param ranges:      Ranges to replace clip_a (original clip) with clip_b (replacement clip).
                        Integer values in the list indicate single frames,
                        Tuple values indicate inclusive ranges.
                        Callbacks must return true to replace a with b.
                        Negative integer values will be wrapped around based on clip_b's length.
                        None values are context dependent:
                            * None provided as sole value to ranges: no-op
                            * Single None value in list: Last frame in clip_b
                            * None as first value of tuple: 0
                            * None as second value of tuple: Last frame in clip_b
    :param exclusive:   Use exclusive ranges (Default: True).
    :param mismatch:    Accept format or resolution mismatch between clips.

    :return:            Clip with ranges from clip_a replaced with clip_b.
    """
    if exclusive and not callable(ranges):
        return vstools_replace_ranges(clip_a, clip_b, normalise_ranges(clip_b, ranges), exclusive, mismatch, prop_src=prop_src)
    return vstools_replace_ranges(clip_a, clip_b, ranges, exclusive, mismatch, prop_src=prop_src)

try:
    from vssource import BestSource
except ImportError:
    pass
else:
    class BestestSource(BestSource):
        def __init__(self, *, force: bool = True, **kwargs: Any) -> None:
            super().__init__(force=force, **kwargs)

            def handler_func(m_type: vs.MessageType, msg: str) -> None:
                if all([
                    m_type == vs.MESSAGE_TYPE_INFORMATION,
                    msg.startswith(("VideoSource ", "AudioSource ")),
                    logging.getLogger().level <= logging.WARNING,
                    is_preview()
                ]):
                    print(msg, end="\r")

            self._log_handle = core.add_log_handler(handler_func)

        def __del__(self) -> None:
            core.remove_log_handler(self._log_handle)


class DeferredMask(vsmasktools_DeferredMask):
    _incl_excl_ranges: FrameRangesN

    @property
    def ranges(self) -> FrameRangesN:
        return [
            (s, (e - 1) if e is not None else e)
            for (s, e) in normalise_ranges(None, self._incl_excl_ranges, norm_dups=True)
        ]

    @ranges.setter
    def ranges(self, value: FrameRangesN) -> None:
        self._incl_excl_ranges = value


class HardsubMask(vsmasktools_HardsubMask, DeferredMask): ...


# Waiting for https://github.com/Jaded-Encoding-Thaumaturgy/vs-masktools/pull/29
class HardsubSignFades(vsmasktools_HardsubSignFades, HardsubMask):
    def __init__(
        self, *args: Any, highpass: float = 0.0763, expand: int = 8, edgemask: GenericMaskT = SobelStd,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, highpass=highpass, expand=expand, edgemask=edgemask, **kwargs)
        self.expand_mode = expand_mode

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clipedge, refedge = (
            normalize_mask(self.edgemask, x, **kwargs).std.Convolution(mean_matrix)
            for x in (clip, ref)
        )

        highpass = scale_value(self.highpass, 32, clip, ColorRange.FULL)

        mask = norm_expr(
            [clipedge, refedge], f'x y - {highpass} < 0 {ExprToken.RangeMax} ?'
        ).std.Median()

        return Morpho.inflate(Morpho.expand(mask, self.expand, mode=self.expand_mode), iterations=4)


# Waiting for https://github.com/Jaded-Encoding-Thaumaturgy/vs-masktools/pull/29
class HardsubSign(vsmasktools_HardsubSign, HardsubMask):
    def __init__(
        self, *args: Any, thr: float = 0.06, minimum: int = 1, expand: int = 8, inflate: int = 7,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, thr=thr, minimum=minimum, expand=expand, inflate=inflate, **kwargs)
        self.expand_mode = expand_mode

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        hsmf = norm_expr([clip, ref], 'x y - abs')
        hsmf = Point.resample(hsmf, clip.format.replace(subsampling_w=0, subsampling_h=0))  # type: ignore

        hsmf = ExprOp.MAX(hsmf, split_planes=True)

        hsmf = Morpho.binarize(hsmf, self.thr)
        hsmf = Morpho.minimum(hsmf, iterations=self.minimum)
        hsmf = Morpho.expand(hsmf, self.expand, mode=self.expand_mode)
        hsmf = Morpho.inflate(hsmf, iterations=self.inflate)

        return hsmf.std.Limiter()


class HardsubLine(vsmasktools_HardsubLine, HardsubMask): ...


class HardsubLineFade(vsmasktools_HardsubLineFade, HardsubMask): ...


class HardsubASS(vsmasktools_HardsubASS, HardsubMask): ...
