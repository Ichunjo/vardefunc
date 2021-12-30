"""Miscellaneous functions and wrappers that didn't really have a place in any other submodules."""
from __future__ import annotations

__all__ = [
    'DebugOutput', 'Thresholds', 'thresholding',
    'fade_filter', 'merge_chroma',
    'Planes', 'YUVPlanes', 'RGBPlanes',
    'get_chroma_shift', 'get_bicubic_params',
    'set_ffms2_log_level'
]

import math
import warnings
from abc import ABC
from contextlib import AbstractContextManager
from functools import partial, wraps
from itertools import count
from operator import ilshift, imatmul, ior
from types import TracebackType
from typing import (
    Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Literal, MutableMapping, NamedTuple,
    Optional, Sequence, Tuple, Type, TypeVar, Union, cast, overload
)

import vapoursynth as vs
from lvsfunc.comparison import Stack
from vsutil import depth, get_depth, get_w, insert_clip, join, plane

from .types import F_OpInput, FormatError, OpInput, Output

core = vs.core


OpDebug = Callable[["DebugOutput", OpInput], "DebugOutput"]
_OPS = {
    '<<=': cast(OpDebug, ilshift),
    '@=': cast(OpDebug, imatmul),
    '|=': cast(OpDebug, ior)
}


class DebugOutputMMap(MutableMapping[int, vs.VideoNode], ABC):
    """Abstract Debug Output interface implementing the mutable mapping methods"""
    outputs: ClassVar[Dict[int, vs.VideoNode]] = {}

    _props: int
    _num: int
    _scale: int

    _min_idx: int
    _max_idx: int

    def __getitem__(self, index: int) -> vs.VideoNode:
        return self.outputs[index]

    def __setitem__(self, index: int, clip: vs.VideoNode) -> None:
        self.outputs[index] = clip

        self._update_minmax()

        if self._props:
            clip = clip.text.FrameProps(alignment=self._props, scale=self._scale)
        if self._num:
            clip = clip.text.FrameNum(self._num, self._scale)

        clip.set_output(index)

    def __delitem__(self, index: int) -> None:
        del self.outputs[index]
        self._update_minmax()
        vs.clear_output(index)

    def __len__(self) -> int:
        return len(self.outputs)

    def __iter__(self) -> Iterator[int]:
        yield from self.outputs.keys()

    def __str__(self) -> str:
        string = ''
        for idx, clip in sorted(self.items()):
            string += f'Index N° {idx}\n' + str(clip) + '---------------\n'
        return string

    def __repr__(self) -> str:
        return repr(self.outputs)

    def __del__(self) -> None:
        """
        Deleting an item will effectively freed the memory since we're invoking vs.clear_output.
        Indexes are also updated.

        However, we can't clear the outputs in the destructor of the DebugOutput instance.
        Previewers won't be able to get the outputs because they run after the end of the script,
        and the destructor is already used.

        So ``del debug[0]`` will clear the output 0 but ``del debug`` won't.
        If you want to clear outputs just do: ``debug.clear()`` and ``del debug``
        """
        self.outputs.clear()
        for name in set(self.__dict__):
            delattr(self, name)

    def _update_minmax(self) -> None:
        try:
            self._min_idx, self._max_idx = min(self.outputs.keys()), max(self.outputs.keys())
        except ValueError:
            del self._min_idx, self._max_idx


class DebugOutput(DebugOutputMMap):
    """Utility class to ouput multiple clips"""

    def __init__(self, *clips: Output, props: int = 0, num: int = 0, scale: int = 1,
                 clear_outputs: bool = False, check_curr_env: bool = True, **named_clips: Output) -> None:
        """
        Args:
            clips (vs.VideoNode | List[vs.VideoNode] | Tuple[int, vs.VideoNode] | Tuple[int, List[vs.VideoNode]]):
                `clips` can be a VideoNode, a list of planes,
                a tuple of an index and VideoNode or a tuple of an index and a list of planes.
                If a list of planes is passed, DebugOutput will try to stack the planes for previewing.
                Only 444 and 420 format are allowed. Otherwise a warning will be raise and a garbage clip will be displayed.

            named_clips (Dict[str, vs.VideoNode | List[vs.VideoNode] | Tuple[int, vs.VideoNode] | Tuple[int, List[vs.VideoNode]]]):
                Same as clips except it's Keyword arguments.
                Location of named_clips's names are hardcoded to 8.

            props (int, optional):
                Location of the displayed FrameProps. 0 means no display.
                Defaults to 0.

            num (int, optional):
                Location of the displayed FrameNum. 0 means no display.
                Defaults to 0.

            scale (int, optional):
                Global integer scaling factor for the bitmap font.
                Defaults to 1.

            clear_outputs (bool, optional):
                Clears all clips set for output in the current environment.
                Defaults to False.

            check_curr_env (bool, optional):
                Check all clips set for output in the current environment.
                Defaults to True.
        """
        self._props = props
        self._num = num
        self._scale = scale
        self._max_idx = 0
        self._min_idx = 0
        self._load_clips(*clips, clear_outputs=clear_outputs, check_curr_env=check_curr_env, **named_clips)

    def _load_clips(self, *clips: Output, clear_outputs: bool = False, check_curr_env: bool = True, **named_clips: Output) -> None:
        rclips = [
            self._resolve_clips(i, clip, None) for i, clip in enumerate(clips)
        ]
        rclips += [
            self._resolve_clips(i, clip, name)
            for i, (name, clip) in enumerate(named_clips.items(), start=len(rclips))
        ]

        if len(all_idx := [idx for idx, _ in rclips]) != len(set(all_idx)):
            raise ValueError('DebugOutput: there are shared indexes!')

        if clear_outputs:
            self.clear()
            self.update(rclips)
        else:
            if check_curr_env:
                self._check_curr_env(all_idx)
            self.update(self._get_outputs() | dict(rclips))

    def __ilshift__(self, clips: OpInput) -> DebugOutput:
        """Adds from the biggest index <<="""
        return self._resolve_input_operator(self._index_gen(self._max_idx + 1), clips, True)

    def __imatmul__(self, clips: OpInput) -> DebugOutput:
        """Fills unused indexes @="""
        return self._resolve_input_operator(self._index_not_used_gen(), clips, True)

    def __ior__(self, clips: OpInput) -> DebugOutput:
        """Fills and replaces existing indexes |="""
        return self._resolve_input_operator(self._index_gen(self._min_idx), clips, False)

    def _resolve_clips(self, i: int, clip: Output, name: Optional[str]) -> Tuple[int, vs.VideoNode]:
        if isinstance(clip, vs.VideoNode):
            out = i, clip
        elif isinstance(clip, list):
            out = i, self._stack_planes(clip)
        else:
            idx, clp = clip
            if isinstance(clp, list):
                out = idx, self._stack_planes(clp)
            else:
                out = idx, clp

        if name:
            idx, c = out
            out = idx, c.text.Text(name, 8, self._scale)

        return out

    def _resolve_input_operator(self, yield_func: Iterable[int], clips: OpInput, env: bool = True) -> DebugOutput:
        if isinstance(clips, dict):
            self._load_clips(
                clear_outputs=False, check_curr_env=env,
                **{name: cast(Output, (i, clip)) for i, (name, clip) in zip(yield_func, clips.items())}
            )
        elif isinstance(clips, tuple):
            if isinstance(clips[0], vs.VideoNode):
                self._load_clips(
                    *zip(yield_func, (c for c in clips if isinstance(c, vs.VideoNode))), check_curr_env=env,
                )
            else:
                self._load_clips(*zip(yield_func, (c for c in clips if isinstance(c, list))), check_curr_env=env,)
        elif isinstance(clips, list):
            self._load_clips(*zip(yield_func, [clips]), check_curr_env=env,)
        else:
            self._load_clips(*zip(yield_func, [clips]), check_curr_env=env)
        return self

    def _index_not_used_gen(self) -> Iterable[int]:
        for i in self._index_gen(self._min_idx):
            if i not in self.keys():
                yield i

    @overload
    def catch(self, func: Optional[F_OpInput], /) -> F_OpInput:
        ...

    @overload
    def catch(self, /, *, op: Union[OpDebug, str] = '<<=') -> Callable[[F_OpInput], F_OpInput]:
        ...

    def catch(self, func: Optional[F_OpInput] = None, /, *, op: Union[OpDebug, str] = '<<='
              ) -> Union[Callable[[F_OpInput], F_OpInput], F_OpInput]:
        """Decorator to catch the output of the function decorated"""
        if func is None:
            return cast(
                Callable[[F_OpInput], F_OpInput],
                partial(self.catch, op=op)
            )

        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> OpInput:
            assert func
            out = func(*args, **kwargs)
            opera = _OPS[op] if isinstance(op, str) else op
            opera(self, out)
            return out

        return cast(F_OpInput, _wrapper)

    @staticmethod
    def _index_gen(start: int) -> Iterable[int]:
        yield from count(start=start)

    @staticmethod
    def _stack_planes(planes: List[vs.VideoNode]) -> vs.VideoNode:
        if len(planes) > 3:
            warnings.warn('DebugOutput: output list out of range', Warning)
            out = core.std.BlankClip(
                format=vs.GRAY8, color=128
            ).text.Text('Problematic output: \noutput list out of range', 5, 2)
        else:
            if len({c.width for c in planes}) == len({c.height for c in planes}) == 1:
                out = Stack(planes).clip
            else:
                try:
                    out = Stack([planes[0], Stack(planes[1:], direction=Direction.VERTICAL).clip]).clip
                except ValueError:
                    warnings.warn('DebugOutput: unexpected subsampling')
                    out = core.std.BlankClip(
                        format=vs.GRAY8, color=128
                    ).text.Text('Problematic output: \nunexpected subsampling', 5, 2)
        return out

    @staticmethod
    def _check_curr_env(idx: Iterable[int]) -> None:
        for i in idx:
            if i in vs.get_outputs().keys():
                raise ValueError(f'DebugOutput: index {i} is already used in current environment!')

    @staticmethod
    def _get_outputs() -> Dict[int, vs.VideoNode]:
        outputs: Dict[int, vs.VideoNode] = {}
        for idx, output in vs.get_outputs().items():
            if isinstance(output, vs.VideoOutputTuple):
                outputs[idx] = output.clip
                if output.alpha:
                    warnings.warn('DebugOutput: VideoOutputTuple.alpha detected; this is not supported', Warning)
        return outputs


class Thresholds(NamedTuple):
    """
    [soft_bound_min, [hard_bound_min, hard_bound_max], soft_bound_max)
    """
    clip: vs.VideoNode
    soft_bound_min: int | float | Sequence[int] | Sequence[float]
    hard_bound_min: int | float | Sequence[int] | Sequence[float]
    hard_bound_max: int | float | Sequence[int] | Sequence[float]
    soft_bound_max: int | float | Sequence[int] | Sequence[float]
    coef_min: int | float | Sequence[int] | Sequence[float] | None = None
    coef_max: int | float | Sequence[int] | Sequence[float] | None = None


def thresholding(*thrs: Thresholds, base: Optional[vs.VideoNode] = None, guidance: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    """
    General function for applying specific filtering on specific thresholds
    with gradation support before and after the hard thresholds

    Args:
        thrs (Thresholds):
            Positional arguments of Thresholds.

        base (vs.VideoNode, optional):
            Base clip on which the first application will be made.
            If not specified, a blank clip is made from the first ``thrs``.

        guidance (VideoNode, optional):
            Guidance clip on which the threshold references are taken.
            If not specified, the guidance clip is made from the first ``thrs``.

    Returns:
        vs.VideoNode:
            Thresholded clip.
    """
    if not base:
        base = thrs[0].clip.std.BlankClip()
    if not guidance:
        guidance = thrs[0].clip

    if not base.format or not guidance.format:
        raise ValueError('thresholding: variable format not allowed')

    for i, thr in enumerate(thrs):
        if thr.clip.format != base.format:
            raise ValueError(f'thresholding: threshold {i} has a different format than base clip')

    def _normalise_thr(thr: int | float | Sequence[int] | Sequence[float], num_planes: int) -> List[int | float]:
        thr = [thr] if isinstance(thr, (float, int)) else thr
        return (list(thr) + [thr[-1]] * (num_planes - len(thr)))[:num_planes]

    pclip = base

    for thr in thrs:
        soft_bound_min, hard_bound_min, hard_bound_max, soft_bound_max = (_normalise_thr(t, base.format.num_planes) for t in thr[1:5])
        coef_min = _normalise_thr(thr.coef_min, base.format.num_planes) if thr.coef_min else None
        coef_max = _normalise_thr(thr.coef_max, base.format.num_planes) if thr.coef_max else None

        exprs: List[str] = []
        for i in range(base.format.num_planes):
            if_in_min = f'x {soft_bound_min[i]} >= x {hard_bound_min[i]} < and'
            if_in_max = f'x {hard_bound_max[i]} >= x {soft_bound_max[i]} < and'
            if_in_hard = f'x {hard_bound_min[i]} >= x {hard_bound_max[i]} < and'

            str_min = f'x {soft_bound_min[i]} - {hard_bound_min[i]} {soft_bound_min[i]} - /'
            if coef_min:
                str_min += f' {coef_min[i]} pow'

            str_max = f'x {hard_bound_max[i]} - {soft_bound_max[i]} {hard_bound_max[i]} - /'
            if coef_max:
                str_max += f' {coef_max} pow'

            exprs.append(
                if_in_min + f' z {str_min} * y 1 {str_min} - * + '
                + if_in_max + f' y {str_max} * z 1 {str_max} - * + '
                + if_in_hard + ' z y ? ? ?'
            )

        pclip = core.std.Expr([guidance, pclip, thr.clip], exprs)

    return pclip



def fade_filter(clip: vs.VideoNode, clip_a: vs.VideoNode, clip_b: vs.VideoNode,
                start_f: int, end_f: int) -> vs.VideoNode:
    """Applies a filter by fading clip_a to clip_b.

    Args:
        clip (vs.VideoNode): Source clip

        clip_a (vs.VideoNode): Fade in clip.

        clip_b (vs.VideoNode): Fade out clip.

        start_f (int): Start frame.

        end_f (int): End frame.

    Returns:
        vs.VideoNode: Faded clip.
    """
    length = end_f - start_f

    def _fade(n: int, clip_a: vs.VideoNode, clip_b: vs.VideoNode, length: int) -> vs.VideoNode:
        return core.std.Merge(clip_a, clip_b, n / length)

    func = partial(_fade, clip_a=clip_a[start_f:end_f + 1], clip_b=clip_b[start_f:end_f + 1], length=length)
    clip_fad = core.std.FrameEval(clip[start_f:end_f + 1], func)

    return insert_clip(clip, clip_fad, start_f)


def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    """Merges chroma from ref with luma.

    Args:
        luma (vs.VideoNode): Source luma clip.
        ref (vs.VideoNode): Source chroma clip.

    Returns:
        vs.VideoNode:
    """
    return core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)



PlanesT = TypeVar('PlanesT', bound='Planes')


class Planes(AbstractContextManager[vs.VideoNode], Sequence[vs.VideoNode]):
    """General context manager for easier planes management"""

    __slots__ = ('_clip', '_family', '_final_clip', '_planes')

    def __init__(self, clip: vs.VideoNode, bits: Optional[int] = None, family: vs.ColorFamily = vs.YUV) -> None:
        """
        Args:
            clip (vs.VideoNode):
                Source clip

            bits (Optional[int], optional):
                Target bitdepth. Defaults to None.

            family (vs.ColorFamily, optional):
                Colour family. Defaults to vs.YUV.
        """
        self._clip = depth(clip, bits) if bits else clip
        self._family = family
        # Initialisation
        self._final_clip: vs.VideoNode
        self._planes: List[vs.VideoNode]
        super().__init__()

    def __enter__(self: PlanesT) -> PlanesT:
        if isinstance(planes := self._clip.std.SplitPlanes(), Sequence):
            self._planes = list(planes)
        else:
            raise FormatError(f'{self.__class__.__name__}: GRAY colour family isn\'t supported!')
        return self

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        self._final_clip = join(self._planes, self._family)
        return super().__exit__(__exc_type, __exc_value, __traceback)

    @overload
    def __getitem__(self, i: int) -> vs.VideoNode:
        ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[vs.VideoNode]:
        ...

    def __getitem__(self, i: int | slice) -> vs.VideoNode | Sequence[vs.VideoNode]:
        return self._planes[i]

    def __setitem__(self, index: int, gray: vs.VideoNode) -> None:
        try:
            self._planes[index] = gray
        except IndexError as i_err:
            raise ValueError(f'{self.__class__.__name__}: plane number out of range') from i_err
        if get_depth(gray) != (bits := get_depth(self._clip)):
            # 32 bits float in YUV and doing on chroma planes
            if bits == 32 and self._family == vs.YUV and index in {1, 2}:
                gray = plane(depth(join([gray] * 3, self._family), bits), index)
            else:
                gray = depth(gray, bits)
            self._planes[index] = depth(gray, bits)

    def __delitem__(self, index: int) -> None:
        self[index] = self[index].std.BlankClip()

    def __len__(self) -> Literal[3]:
        return 3

    @property
    def clip(self) -> vs.VideoNode:
        """Get final merged clip"""
        try:
            out = self._final_clip
        except AttributeError as attr_err:
            raise ValueError(
                f'{self.__class__.__name__}: you can only get "clip" outside of the context manager and once'
            ) from attr_err
        else:
            del self._clip, self._family, self._final_clip, self._planes
            return out


class YUVPlanes(Planes):
    def __init__(self, clip: vs.VideoNode, bits: Optional[int] = None) -> None:
        super().__init__(clip, bits, vs.YUV)

    @property
    def Y(self) -> vs.VideoNode:
        return self[0]

    @Y.setter
    def Y(self, _x: vs.VideoNode) -> None:
        self[0] = _x

    @Y.deleter
    def Y(self) -> None:
        del self[0]

    @property
    def U(self) -> vs.VideoNode:
        return self[1]

    @U.setter
    def U(self, _x: vs.VideoNode) -> None:
        self[1] = _x

    @U.deleter
    def U(self) -> None:
        del self[1]

    @property
    def V(self) -> vs.VideoNode:
        return self[2]

    @V.setter
    def V(self, _x: vs.VideoNode) -> None:
        self[2] = _x

    @V.deleter
    def V(self) -> None:
        del self[2]


class RGBPlanes(Planes):
    def __init__(self, clip: vs.VideoNode, bits: Optional[int] = None) -> None:
        super().__init__(clip, bits, vs.RGB)

    @property
    def R(self) -> vs.VideoNode:
        return self[0]

    @R.setter
    def R(self, _x: vs.VideoNode) -> None:
        self[0] = _x

    @R.deleter
    def R(self) -> None:
        del self[0]

    @property
    def G(self) -> vs.VideoNode:
        return self[1]

    @G.setter
    def G(self, _x: vs.VideoNode) -> None:
        self[1] = _x

    @G.deleter
    def G(self) -> None:
        del self[1]

    @property
    def B(self) -> vs.VideoNode:
        return self[2]

    @B.setter
    def B(self, _x: vs.VideoNode) -> None:
        self[2] = _x

    @B.deleter
    def B(self) -> None:
        del self[2]



def get_chroma_shift(src_h: int, dst_h: int, aspect_ratio: float = 16 / 9) -> float:
    """Intended to calculate the right value for chroma shifting when doing subsampled scaling.

    Args:
        src_h (int): Source height.
        dst_h (int): Destination height.
        aspect_ratio (float, optional): Defaults to 16/9.

    Returns:
        float:
    """
    src_w = get_w(src_h, aspect_ratio)
    dst_w = get_w(dst_h, aspect_ratio)

    ch_shift = 0.25 - 0.25 * (src_w / dst_w)
    ch_shift = float(round(ch_shift, 5))
    return ch_shift


def get_bicubic_params(cubic_filter: str) -> Tuple[float, float]:
    """Return the parameter b and c for the bicubic filter
       Source: https://www.imagemagick.org/discourse-server/viewtopic.php?f=22&t=19823
               https://www.imagemagick.org/Usage/filter/#mitchell

    Args:
        cubic_filter (str): Can be: Spline, B-Spline, Hermite, Mitchell-Netravali, Mitchell,
                            Catmull-Rom, Catrom, Sharp Bicubic, Robidoux soft, Robidoux, Robidoux Sharp.

    Returns:
        Tuple: b/c combo
    """
    sqrt = math.sqrt

    def _get_robidoux_soft() -> Tuple[float, float]:
        b = (9 - 3 * sqrt(2)) / 7
        c = (1 - b) / 2
        return b, c

    def _get_robidoux() -> Tuple[float, float]:
        sqrt2 = sqrt(2)
        b = 12 / (19 + 9 * sqrt2)
        c = 113 / (58 + 216 * sqrt2)
        return b, c

    def _get_robidoux_sharp() -> Tuple[float, float]:
        sqrt2 = sqrt(2)
        b = 6 / (13 + 7 * sqrt2)
        c = 7 / (2 + 12 * sqrt2)
        return b, c

    cubic_filter = cubic_filter.lower().replace(' ', '_').replace('-', '_')
    cubic_filters = {
        'spline': (1.0, 0.0),
        'b_spline': (1.0, 0.0),
        'hermite': (0.0, 0.0),
        'mitchell_netravali': (1 / 3, 1 / 3),
        'mitchell': (1 / 3, 1 / 3),
        'catmull_rom': (0.0, 0.5),
        'catrom': (0.0, 0.5),
        'bicubic_sharp': (0.0, 1.0),
        'sharp_bicubic': (0.0, 1.0),
        'robidoux_soft': _get_robidoux_soft(),
        'robidoux': _get_robidoux(),
        'robidoux_sharp': _get_robidoux_sharp()
    }
    return cubic_filters[cubic_filter]


def set_ffms2_log_level(level: Union[str, int] = 0) -> None:
    """A more friendly set of log level in ffms2

    Args:
        level (int, optional): The target level in ffms2.
                               Valid choices are "quiet" or 0, "panic" or 1, "fatal" or 2, "error" or 3,
                               "warning" or 4, "info" or 5, "verbose" or 6, "debug" or 7 and "trace" or 8.
                               Defaults to 0.
    """
    levels = {
        'quiet': -8,
        'panic': 0,
        'fatal': 8,
        'error': 16,
        'warning': 24,
        'info': 32,
        'verbose': 40,
        'debug': 48,
        'trace': 56,
        0: -8,
        1: 0,
        2: 8,
        3: 16,
        4: 24,
        5: 32,
        6: 40,
        7: 48,
        8: 56
    }
    core.ffms2.SetLogLevel(levels[level])
