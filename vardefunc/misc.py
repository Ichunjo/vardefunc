"""Miscellaneous functions and wrappers that didn't really have a place in any other submodules."""
import math
import subprocess
from functools import partial
from typing import Tuple, Union

from vsutil import get_w
import vapoursynth as vs

core = vs.core


def fade_filter(source: vs.VideoNode, clip_a: vs.VideoNode, clip_b: vs.VideoNode,
                start_f: int, end_f: int) -> vs.VideoNode:
    """Apply a filter with a fade

    Args:
        source (vs.VideoNode): Source clip
        clip_a (vs.VideoNode): Fade in clip
        clip_b (vs.VideoNode): Fade out clip
        start_f (int): Start frame.
        end_f (int): End frame.

    Returns:
        vs.VideoNode:
    """
    length = end_f - start_f

    def _fade(n, clip_a, clip_b, length):
        return core.std.Merge(clip_a, clip_b, n / length)

    function = partial(_fade, clip_a=clip_a[start_f:end_f + 1], clip_b=clip_b[start_f:end_f + 1], length=length)
    clip_fad = core.std.FrameEval(source[start_f:end_f + 1], function)

    final = clip_fad

    if start_f != 0:
        final = source[:start_f] + final
    if end_f + 1 < source.num_frames:
        final = final + source[end_f + 1:]

    return final


def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    """Merge chroma from ref with luma

    Args:
        luma (vs.VideoNode): Source luma clip
        ref (vs.VideoNode): Source chroma clip

    Returns:
        vs.VideoNode:
    """
    return core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)


def get_chroma_shift(src_h: int, dst_h: int, aspect_ratio: float = 16 / 9) -> float:
    """Intended to calculate the right value for chroma shifting

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


def get_bicubic_params(cubic_filter: str) -> Tuple:
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

    def _get_robidoux_soft() -> Tuple:
        b = (9 - 3 * sqrt(2)) / 7
        c = (1 - b) / 2
        return b, c

    def _get_robidoux() -> Tuple:
        sqrt2 = sqrt(2)
        b = 12 / (19 + 9 * sqrt2)
        c = 113 / (58 + 216 * sqrt2)
        return b, c

    def _get_robidoux_sharp() -> Tuple:
        sqrt2 = sqrt(2)
        b = 6 / (13 + 7 * sqrt2)
        c = 7 / (2 + 12 * sqrt2)
        return b, c

    cubic_filter = cubic_filter.lower().replace(' ', '_').replace('-', '_')
    cubic_filters = {
        'spline': (1, 0),
        'b_spline': (1, 0),
        'hermite': (0, 0),
        'mitchell_netravali': (1 / 3, 1 / 3),
        'mitchell': (1 / 3, 1 / 3),
        'catmull_rom': (0, 1 / 2),
        'catrom': (0, 1 / 2),
        'bicubic_sharp': (0, 1),
        'sharp_bicubic': (0, 1),
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


def encode(clip: vs.VideoNode, binary: str, output_file: str, **args) -> None:
    """Stolen from lyfunc
    Args:
        clip (vs.VideoNode): Source filtered clip
        binary (str): Path to x264 binary.
        output_file (str): Path to the output file.
    """
    cmd = [binary,
           "--demuxer", "y4m",
           "--frames", f"{clip.num_frames}",
           "--sar", "1:1",
           "--output-depth", "10",
           "--output-csp", "i420",
           "--colormatrix", "bt709",
           "--colorprim", "bt709",
           "--transfer", "bt709",
           "--no-fast-pskip",
           "--no-dct-decimate",
           "--partitions", "all",
           "-o", output_file,
           "-"]
    for i, v in args.items():
        i = "--" + i if i[:2] != "--" else i
        i = i.replace("_", "-")
        if i in cmd:
            cmd[cmd.index(i) + 1] = str(v)
        else:
            cmd.extend([i, str(v)])

    print("Encoder command: ", " ".join(cmd), "\n")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    process.communicate()


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
