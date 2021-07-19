"""Miscellaneous functions and wrappers that didn't really have a place in any other submodules."""
import math
import warnings
from functools import partial
from typing import Iterable, List, Optional, Tuple, Union, overload

import vapoursynth as vs
from lvsfunc.comparison import Stack
from vsutil import get_w, insert_clip

core = vs.core


OutputClip = Union[
    vs.VideoNode,
    List[vs.VideoNode],
    Tuple[int, vs.VideoNode],
    Tuple[int, List[vs.VideoNode]]
]


class DebugOutput:
    """Utility class to ouput multiple clips"""
    @overload
    def __init__(self, *clips: OutputClip) -> None:
        ...

    @overload
    def __init__(self, *clips: OutputClip, clear_outputs: bool = False) -> None:
        ...

    @overload
    def __init__(self, *clips: OutputClip, clear_outputs: bool = False, **named_clips: OutputClip) -> None:
        ...

    def __init__(self, *clips: OutputClip, clear_outputs: bool = False, **named_clips: OutputClip) -> None:
        """
        Utility class to ouput multiple clips.
        Either `clips` or `named_clips` can be a VideoNode, a list of planes,
        a tuple of an index and VideoNode or a tuple of an index and a list of planes.

        If a list of planes is passed, DebugOutput will try to stack the planes for previewing.
        Only 444 and 420 format are allowed. Otherwise a warning will be raise and a garbage clip will be displayed.

        Args:
            clips (vs.VideoNode | List[vs.VideoNode] | Tuple[int, vs.VideoNode] | Tuple[int, List[vs.VideoNode]]):
                Output clips.

            named_clips (Dict[str, vs.VideoNode | List[vs.VideoNode] | Tuple[int, vs.VideoNode] | Tuple[int, List[vs.VideoNode]]]):
                Keyword arguments for all output clips.

            clear_outputs (bool, optional):
                Clears all clips set for output in the current environment.
                Defaults to False.
        """

        rclips = [
            self._resolve_clips(i, clip) for i, clip in enumerate(clips)
        ]
        rclips += [
            self._resolve_clips(i, clip, name)
            for i, (name, clip) in enumerate(named_clips.items(), start=len(rclips))
        ]

        if len(all_idx := [idx for idx, _ in rclips]) != len(set(all_idx)):
            raise ValueError('DebugOutput: there are shared indexes!')

        if clear_outputs:
            vs.clear_outputs()
        else:
            self._check_curr_env(all_idx)

        for idx, clip in rclips:
            clip.set_output(idx)

    def _resolve_clips(self, i: int, clip: OutputClip, name: Optional[str] = None) -> Tuple[int, vs.VideoNode]:
        if isinstance(clip, vs.VideoNode):
            out = (i, clip)
        elif isinstance(clip, list):
            out = (i, self._stack_planes(clip))
        else:
            idx, clp = clip
            if isinstance(clp, list):
                out = (idx, self._stack_planes(clp))
            else:
                out = (idx, clp)

        if name:
            idx, c = out
            out = idx, c.text.Text(name)

        return out

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
                finally:
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


def generate_keyframes(clip: vs.VideoNode, out_path: str) -> None:
    """Generate qp filename for keyframes to pass the file into the encoder
       to force I frames. Use both scxvid and wwxd. Original function stolen from kagefunc.

    Args:
        clip (vs.VideoNode): Source clip
        out_path (str): output path
    """
    clip = core.resize.Bilinear(clip, 640, 360)
    clip = core.scxvid.Scxvid(clip)
    clip = core.wwxd.WWXD(clip)
    out_txt = ""
    for i in range(clip.num_frames):
        if clip.get_frame(i).props["_SceneChangePrev"] == 1 \
                or clip.get_frame(i).props["Scenechange"] == 1:
            out_txt += "%d I -1\n" % i
        if i % 2000 == 0:
            print(i)
    text_file = open(out_path, "w")
    text_file.write(out_txt)
    text_file.close()


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
