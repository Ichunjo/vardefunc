"""Miscellaneous functions and wrappers that didn't really have a place in any other submodules."""
from __future__ import annotations

import math
import warnings
from abc import ABC
from functools import partial, wraps
from itertools import count
from operator import ilshift, imatmul, ior
from typing import (Any, Callable, ClassVar, Dict, Iterable, Iterator, List,
                    MutableMapping, Optional, Tuple, Union, cast, overload)

import vapoursynth as vs
from lvsfunc.comparison import Stack
from vsutil import get_w, insert_clip

from .types import F_OpInput, OpInput, Output

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
            self.update(dict(rclips))
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
            ws, hs = {c.width for c in planes}, {c.height for c in planes}
            if len(ws) == len(hs) == 1:
                out = Stack(planes).clip
            else:
                try:
                    out = Stack([planes[0], Stack(planes[1:]).clip]).clip
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
