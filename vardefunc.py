"""
Various functions I use.
"""
import math
import subprocess
from pathlib import Path
from typing import Union, NoReturn, Tuple
from functools import partial
from vsutil import core, vs, depth, get_depth, get_y, get_w, split

import fvsfunc as fvf
import havsfunc as hvf


def fade_filter(source: vs.VideoNode, clip_a: vs.VideoNode, clip_b: vs.VideoNode,
                start_f: int = None, end_f: int = None)-> vs.VideoNode:
    """Apply a filter with a fade

    Args:
        source (vs.VideoNode): Source clip
        clip_a (vs.VideoNode): Fade in clip
        clip_b (vs.VideoNode): Fade out clip
        start_f (int, optional): Start frame. Defaults to None.
        end_f (int, optional): End frame. Defaults to None.

    Returns:
        vs.VideoNode:
    """
    length = end_f - start_f

    def _fade(n, clip_a, clip_b, length):
        return core.std.Merge(clip_a, clip_b, n / length)

    clip_fad = core.std.FrameEval(source[start_f:end_f+1],
                                  partial(_fade, clip_a=clip_a[start_f:end_f+1],
                                          clip_b=clip_b[start_f:end_f+1], length=length))
    return source[:start_f] + clip_fad + source[end_f+1:]


def knlmcl(source: vs.VideoNode, h_y: float = 1.2, h_uv: float = 0.5,
           device_id: int = 0, bits: int = None)-> vs.VideoNode:
    """Denoise both luma and chroma with KNLMeansCL

    Args:
        source (vs.VideoNode): Source clip
        h_y (float, optional): h parameter in KNLMeansCL for the luma. Defaults to 1.2.
        h_uv (float, optional): h parameter in KNLMeansCL for the chroma. Defaults to 0.5.
        device_id (int, optional): Device id in KNLMeansCL. Defaults to 0.
        bits (int, optional): Output bitdepth. Defaults to None.

    Returns:
        vs.VideoNode:
    """
    if get_depth(source) != 32:
        clip = depth(source, 32)
    else:
        clip = source

    denoise = core.knlm.KNLMeansCL(clip, a=2, h=h_y, d=3, device_type='gpu',
                                   device_id=device_id, channels='Y')
    denoise = core.knlm.KNLMeansCL(denoise, a=2, h=h_uv, d=3, device_type='gpu',
                                   device_id=device_id, channels='UV')
    if depth is not None:
        denoise = depth(denoise, bits)

    return denoise


def diff_rescale_mask(source: vs.VideoNode, height: int = 720, kernel: str = 'bicubic',
                      b: float = 0, c: float = 1/2, mthr: int = 55,
                      mode: str = 'rectangle', sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """Modified version of Atomchtools for generate a mask with a rescaled difference

    Args:
        source (vs.VideoNode): Source clip
        height (int, optional): Defaults to 720.
        kernel (str, optional): Defaults to 'bicubic'.
        b (float, optional): Defaults to 0.
        c (float, optional): Defaults to 1/2.
        mthr (int, optional): Defaults to 55.
        mode (str, optional): Can be 'rectangle', 'losange' or 'ellipse' . Defaults to 'rectangle'.
        sw (int, optional): Growing/shrinking shape width. 0 is allowed. Defaults to 2.
        sh (int, optional): Growing/shrinking shape height. 0 is allowed. Defaults to 2.

    Returns:
        vs.VideoNode:
    """
    only_luma = source.format.num_planes == 1

    if not only_luma:
        clip = get_y(source)
    else:
        clip = source

    if get_depth(source) != 8:
        clip = depth(clip, 8)

    width = get_w(height)
    desc = fvf.Resize(clip, width, height, kernel=kernel, a1=b, a2=c, invks=True)
    upsc = depth(fvf.Resize(desc, source.width, source.height, kernel=kernel, a1=b, a2=c), 8)

    diff = core.std.MakeDiff(clip, upsc)
    mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
    mask = mask.std.Expr(f'x {mthr} < 0 x ?')
    mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    mask = hvf.mt_expand_multi(mask, mode=mode, sw=sw, sh=sh)

    if get_depth(source) != 8:
        mask = depth(mask, get_depth(source))
    return mask


def diff_creditless_mask(source: vs.VideoNode, titles: vs.VideoNode, nc: vs.VideoNode,
                         start: int = None, end: int = None,
                         sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """Modified version of Atomchtools for generate a mask with with a NC

    Args:
        source (vs.VideoNode): Source clip
        titles (vs.VideoNode): Credit clip
        nc (vs.VideoNode): Non credit clip
        start (int, optional): Start frame. Defaults to None.
        end (int, optional): End frame. Defaults to None.
        sw (int, optional): Growing/shrinking shape width. 0 is allowed. Defaults to 2.
        sh (int, optional): Growing/shrinking shape height. 0 is allowed. Defaults to 2.

    Returns:
        vs.VideoNode:
    """
    if get_depth(titles) != 8:
        titles = depth(titles, 8)
    if get_depth(nc) != 8:
        nc = depth(nc, 8)

    diff = core.std.MakeDiff(titles, nc, [0])
    diff = get_y(diff)
    diff = diff.std.Prewitt().std.Expr('x 25 < 0 x ?').std.Expr('x 2 *')
    diff = core.rgvs.RemoveGrain(diff, 4).std.Expr('x 30 > 255 x ?')

    credit_m = hvf.mt_expand_multi(diff, sw=sw, sh=sh)

    blank = core.std.BlankClip(source, format=vs.GRAY8)

    if start == 0:
        credit_m = credit_m+blank[end+1:]
    elif end == source.num_frames-1:
        credit_m = blank[:start]+credit_m
    else:
        credit_m = blank[:start]+credit_m+blank[end+1:]

    if get_depth(source) != 8:
        credit_m = depth(credit_m, get_depth(source))
    return credit_m


def nnedi3cl_double(clip: vs.VideoNode, znedi: bool = True, **args)-> vs.VideoNode:
    """Double the clip using nnedi3 for even frames and nnedi3cl for odd frames
       Intended to speed up encoding speed without hogging the GPU either

    Args:
        clip (vs.VideoNode): Source clip
        znedi (bool, optional): Use znedi3 or not. Defaults to True.

    Returns:
        vs.VideoNode:
    """
    args = args or dict(nsize=0, nns=4, qual=2, pscrn=2)

    def _nnedi3(clip):
        if znedi:
            clip = clip.std.Transpose().znedi3.nnedi3(0, True, **args) \
                .std.Transpose().znedi3.nnedi3(0, True, **args)
        else:
            clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **args) \
                .std.Transpose().nnedi3.nnedi3(0, True, **args)
        return clip

    def _nnedi3cl(clip):
        return clip.nnedi3cl.NNEDI3CL(0, True, True, **args)

    clip = core.std.Interleave([_nnedi3(clip[::2]), _nnedi3cl(clip[1::2])])
    return core.resize.Spline36(clip, src_top=.5, src_left=.5)


def to444(clip: vs.VideoNode, width: int = None, height: int = None, join_planes: bool = True)-> vs.VideoNode:
    """Zastin’s nnedi3 chroma upscaler

    Args:
        clip ([type]): Source clip
        width (int, optional): Upscale width. Defaults to None.
        height (int, optional): Upscale height. Defaults to None.
        join_planes (bool, optional): Defaults to True.

    Returns:
        vs.VideoNode: [description]
    """
    def _nnedi3x2(clip):
        if hasattr(core, 'znedi3'):
            clip = clip.std.Transpose().znedi3.nnedi3(1, 1, 0, 0, 4, 2) \
                .std.Transpose().znedi3.nnedi3(0, 1, 0, 0, 4, 2)
        else:
            clip = clip.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 1) \
                .std.Transpose().nnedi3.nnedi3(0, 1, 0, 0, 3, 1)
        return clip

    chroma = [_nnedi3x2(c) for c in split(clip)[1:]]

    if width in (None, clip.width) and height in (None, clip.height):
        chroma = [core.resize.Spline36(c, src_top=0.5) for c in chroma]
    else:
        chroma = [core.resize.Spline36(c, width, height, src_top=0.5) for c in chroma]

    return core.std.ShufflePlanes([clip] + chroma, [0]*3, vs.YUV) if join_planes else chroma


def region_mask(clip: vs.VideoNode,
                left: int = None, right: int = None,
                top: int = None, bottom: int = None)-> vs.VideoNode:
    """Crop your mask

    Args:
        clip (vs.VideoNode): Source clip
        left (int, optional): Left crop. Defaults to None.
        right (int, optional): Right crop. Defaults to None.
        top (int, optional): Top crop. Defaults to None.
        bottom (int, optional): Bottom crop. Defaults to None.

    Returns:
        vs.VideoNode:
    """
    crop = core.std.Crop(clip, left, right, top, bottom)
    borders = core.std.AddBorders(crop, left, right, top, bottom)
    return borders


def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode)-> vs.VideoNode:
    """Merge chroma from ref with luma

    Args:
        luma (vs.VideoNode): Source luma clip
        ref (vs.VideoNode): Source chroma clip

    Returns:
        vs.VideoNode:
    """
    return core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)


def get_chroma_shift(src_h: int = None, dst_h: int = None,
                     aspect_ratio: float = 16/9)-> float:
    """Intended to calculate the right value for chroma shifting

    Args:
        src_h (int, optional): Source height. Defaults to None.
        dst_h (int, optional): Destination height. Defaults to None.
        aspect_ratio (float, optional): Defaults to 16/9.

    Returns:
        float:
    """
    src_w = get_w(src_h, aspect_ratio)
    dst_w = get_w(dst_h, aspect_ratio)

    ch_shift = 0.25 - 0.25 * (src_w / dst_w)
    ch_shift = float(round(ch_shift, 5))
    return ch_shift


def get_bicubic_params(cubic_filter: str)-> Tuple:
    """Return the parameter b and c for the bicubic filter
       Source: https://www.imagemagick.org/discourse-server/viewtopic.php?f=22&t=19823
               https://www.imagemagick.org/Usage/filter/#mitchell

    Args:
        cubic_filter (str): Can be: Spline, B-Spline, Hermite, Mitchell-Netravali, Mitchell,
                            Catmull-Rom, Catrom, Sharp Bicubic, Robidoux soft, Robidoux, Robidoux Sharp.

    Returns:
        Tuple: b/c combo
    """
    def _sqrt(var):
        return math.sqrt(var)

    def _get_robidoux_soft()-> Tuple:
        b = (9-3*_sqrt(2))/7
        c = (1-b)/2
        return b, c

    def _get_robidoux()-> Tuple:
        sqrt2 = _sqrt(2)
        b = 12/(19+9*sqrt2)
        c = 113/(58+216*sqrt2)
        return b, c

    def _get_robidoux_sharp()-> Tuple:
        sqrt2 = _sqrt(2)
        b = 6/(13+7*sqrt2)
        c = 7/(2+12*sqrt2)
        return b, c

    cubic_filter = cubic_filter.lower().replace(' ', '_').replace('-', '_')
    cubic_filters = {
        'spline': (1, 0),
        'b_spline': (1, 0),
        'hermite': (0, 0),
        'mitchell_netravali': (1/3, 1/3),
        'mitchell': (1/3, 1/3),
        'catmull_rom': (0, 1/2),
        'catrom': (0, 1/2),
        'bicubic_sharp': (0, 1),
        'sharp_bicubic': (0, 1),
        'robidoux_soft': _get_robidoux_soft(),
        'robidoux': _get_robidoux(),
        'robidoux_sharp': _get_robidoux_sharp()
    }
    return cubic_filters[cubic_filter]


def generate_keyframes(clip: vs.VideoNode, out_path: str = None) -> NoReturn:
    """Generate qp filename for keyframes to pass the file into the encoder
       to force I frames. Use both scxvid and wwxd. Original function stolen from kagefunc.

    Args:
        clip (vs.VideoNode): Source clip
        out_path (str, optional): Defaults to None.
    """
    clip = core.resize.Bilinear(clip, 640, 360)
    clip = core.scxvid.Scxvid(clip)
    clip = core.wwxd.WWXD(clip)
    out_txt = ""
    for i in range(clip.num_frames):
        if clip.get_frame(i).props["_SceneChangePrev"] == 1 \
            or clip.get_frame(i).props["Scenechange"] == 1:
            out_txt += "%d I -1\n" % i
        if i % 500 == 0:
            print(i)
    text_file = open(out_path, "w")
    text_file.write(out_txt)
    text_file.close()


def encode(clip: vs.VideoNode, x264: Union[str, Path] = None,
           output_file: str = None, **args) -> NoReturn:
    """Stolen from lyfunc
    Args:
        clip (vs.VideoNode): Source filtered clip
        x264 (Union[str, Path]): Path to x264 build. Defaults to None.
        output_file (str): Path to output file. Defaults to None.
    """
    x264_cmd = [x264,
                "--demuxer", "y4m",
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
        if i in x264_cmd:
            x264_cmd[x264_cmd.index(i)+ 1] = str(v)
        else:
            x264_cmd.extend([i, str(v)])

    print("x264 command: ", " ".join(x264_cmd), "\n")
    process = subprocess.Popen(x264_cmd, stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue: print(
        f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || x264: ", end=""))
    process.communicate()


drm = diff_rescale_mask
dcm = diff_creditless_mask
gk = generate_keyframes
