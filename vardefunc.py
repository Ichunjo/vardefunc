"""
Various functions I use.
"""
import math
import subprocess
from typing import Tuple, Callable, Dict, Any, cast, Union
from functools import partial

import fvsfunc as fvf
import havsfunc as hvf
import placebo

from vsutil import depth, get_depth, get_y, get_w, split, iterate
import vapoursynth as vs

core = vs.core



# # # # # # # # # #
# Noise functions #
# # # # # # # # # #

def adaptative_regrain(denoised: vs.VideoNode, new_grained: vs.VideoNode, original_grained: vs.VideoNode,
                       range_avg: Tuple[float, float] = (0.5, 0.4), luma_scaling: int = 28)-> vs.VideoNode:
    """Merge back the original grain below the lower range_avg value,
       apply the new grain clip above the higher range_avg value
       and weight both of them between the range_avg values for a smooth merge.
       Intended for use in applying a static grain in higher PlaneStatsAverage values
       to decrease the file size since we can't see a dynamic grain on that level.
       However, in dark scenes, it's more noticeable so we apply the original grain.

    Args:
        denoised (vs.VideoNode): The denoised clip.
        new_grained (vs.VideoNode): The new regrained clip.
        original_grained (vs.VideoNode): The original regrained clip.
        range_avg (Tuple[float, float], optional): Range used in PlaneStatsAverage. Defaults to (0.5, 0.4).
        luma_scaling (int, optional): Parameter in adg.Mask. Defaults to 28.

    Returns:
        vs.VideoNode: The new adaptative grained clip.

    Example:
        import vardefunc as vdf

        denoise = denoise_filter(src, ...)
        diff = core.std.MakeDiff(src, denoise)
        ...
        some filters
        ...
        new_grained = core.neo_f3kdb.Deband(last, preset='depth', grainy=32, grainc=32)
        original_grained = core.std.MergeDiff(last, diff)
        adapt_regrain = vdf.adaptative_regrain(last, new_grained, original_grained, range_avg=(0.5, 0.4), luma_scaling=28)
    """

    avg = core.std.PlaneStats(denoised)
    adapt_mask = core.adg.Mask(get_y(avg), luma_scaling)
    adapt_grained = core.std.MaskedMerge(new_grained, original_grained, adapt_mask)

    avg_max = max(range_avg)
    avg_min = min(range_avg)

    # pylint: disable=unused-argument
    def _diff(n: int, f: vs.VideoFrame, avg_max: float, avg_min: float,
              new: vs.VideoNode, adapt: vs.VideoNode) -> vs.VideoNode:
        psa = cast(float, f.props['PlaneStatsAverage'])
        if psa > avg_max:
            clip = new
        elif psa < avg_min:
            clip = adapt
        else:
            weight = (psa - avg_min) / (avg_max - avg_min)
            clip = core.std.Merge(adapt, new, [weight])
        return clip

    diff_function = partial(_diff, avg_max=avg_max, avg_min=avg_min, new=new_grained, adapt=adapt_grained)

    return core.std.FrameEval(denoised, diff_function, [avg])




# # # # # # # # # # # #
# Upscaling functions #
# # # # # # # # # # # #

def nnedi3cl_double(clip: vs.VideoNode, znedi: bool = True, **nnedi3_args)-> vs.VideoNode:
    """Double the clip using nnedi3 for even frames and nnedi3cl for odd frames
       Intended to speed up encoding speed without hogging the GPU either

    Args:
        clip (vs.VideoNode): Source clip.
        znedi (bool, optional): Use znedi3 or not. Defaults to True.

    Returns:
        vs.VideoNode:
    """
    nnargs: Dict[str, Any] = dict(nsize=0, nns=4, qual=2, pscrn=2)
    nnargs.update(nnedi3_args)

    def _nnedi3(clip):
        if znedi:
            clip = clip.std.Transpose().znedi3.nnedi3(0, True, **nnargs) \
                .std.Transpose().znedi3.nnedi3(0, True, **nnargs)
        else:
            clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **nnargs) \
                .std.Transpose().nnedi3.nnedi3(0, True, **nnargs)
        return clip

    def _nnedi3cl(clip):
        return clip.nnedi3cl.NNEDI3CL(0, True, True, **nnargs)

    clip = core.std.Interleave([_nnedi3(clip[::2]), _nnedi3cl(clip[1::2])])
    return core.resize.Spline36(clip, src_top=.5, src_left=.5)


def nnedi3_upscale(clip: vs.VideoNode, scaler: Callable[[vs.VideoNode, Any], vs.VideoNode] = None,
                   correct_shift: bool = True, **nnedi3_args)-> vs.VideoNode:
    """Classic based nnedi3 upscale.

    Args:
        clip (vs.VideoNode): Source clip.
        scaler (Callable[[vs.VideoNode, Any], vs.VideoNode], optional): Resizer used to correct the shift. Defaults to core.resize.Spline36.
        correct_shift (bool, optional): Defaults to True.

    Returns:
        vs.VideoNode: Upscaled clip.
    """
    nnargs: Dict[str, Any] = dict(nsize=4, nns=4, qual=2, pscrn=2)
    nnargs.update(nnedi3_args)
    clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **nnargs).std.Transpose().nnedi3.nnedi3(0, True, **nnargs)

    if scaler is None:
        scaler = core.resize.Spline36

    return scaler(clip, src_top=.5, src_left=.5) if correct_shift else clip


def fsrcnnx_upscale(source: vs.VideoNode, width: int = None, height: int = 1080, shader_file: str = None,
                    downscaler: Callable[[vs.VideoNode, int, int], vs.VideoNode] = core.resize.Bicubic,
                    upscaler_smooth: Callable[[vs.VideoNode, Any], vs.VideoNode] = partial(nnedi3_upscale, nsize=4, nns=4, qual=2, pscrn=2),
                    strength: float = 100.0, profile: str = 'slow',
                    lmode: int = 1, overshoot: int = None, undershoot: int = None)-> vs.VideoNode:
    """Upscale the given luma source clip with FSRCNNX to a given width / height while preventing FSRCNNX artifacts by limiting them.

    Args:
        source (vs.VideoNode): Source clip, assuming this one is perfectly descaled.
        width (int): Target resolution width. Defaults to None.
        height (int): Target resolution height. Defaults to 1080.
        shader_file (str): Path to the FSRCNNX shader file. Defaults to None.
        downscaler (Callable[[vs.VideoNode, int, int], vs.VideoNode], optional): Resizer used to downscale the upscaled clip.
                                                                                 Defaults to core.resize.Bicubic.
        upscaler_smooth (Callable[[vs.VideoNode, Any], vs.VideoNode], optional): Resizer used to replace the smoother nnedi3 upscale.
                                                                                 Defaults to partial(nnedi3_upscale, nsize=4, nns=4, qual=2, pscrn=2).
        strength (float): Only for profile='slow'. Strength between the smooth upscale and the fsrcnnx upscale where 0.0 means the full smooth clip
                          and 100.0 means the full fsrcnnx clip. Negative and positive values are possible, but not recommended.
        profile (str): Profile settings. Possible strings: "fast", "old" or "slow". "fast" is the old draft mode (the plain fsrcnnx clip returned),
                       "old" is the old mode to deal with the bright pixels and "slow" is the new mode, more efficient.
        lmode (int): Only for profile='slow':
                     – (< 0): Limit with rgvs.Repair (ex: lmode=-1 --> rgvs.Repair(1), lmode=-5 --> rgvs.Repair(5) ...)
                     – (= 0): No limit.
                     – (= 1): Limit to over/undershoot.
        overshoot (int): Limit for pixels that get brighter during upscaling.
        undershoot (int): Limit for pixels that get darker during upscaling.

    Returns:
        vs.VideoNode: Upscaled luma clip.
    """
    if source.format.num_planes > 1:
        source = get_y(source)

    if (depth_src := get_depth(source)) != 16:
        clip = depth(source, 16)
    else:
        clip = source

    if width is None:
        width = get_w(height, clip.width/clip.height)
    if overshoot is None:
        overshoot = strength / 100
    if undershoot is None:
        undershoot = overshoot

    try:
        num = ['fast', 'old', 'slow'].index(profile.lower())
    except:
        raise vs.Error('fsrcnnx_upscale: profile must be "fast" or "old" or "slow"')




    fsrcnnx = placebo.shader(clip, clip.width*2, clip.height*2, shader_file)


    if num >= 1:
        # old or slow profile
        smooth = upscaler_smooth(clip)
        if num == 1:
            # old profile
            limit = core.std.Expr([fsrcnnx, smooth], 'x y min')
        else:
            # slow profile
            upscaled = core.std.Expr([fsrcnnx, smooth], 'x {strength} * y 1 {strength} - * +'.format(strength=strength/100))
            if lmode < 0:
                limit = core.rgvs.Repair(upscaled, smooth, abs(lmode))
            elif lmode == 0:
                limit = upscaled
            elif lmode == 1:
                dark_limit = core.std.Minimum(smooth)
                bright_limit = core.std.Maximum(smooth)

                limit = hvf.Clamp(upscaled, bright_limit, dark_limit, hvf.scale(overshoot, (1 << 16) - 1), hvf.scale(undershoot, (1 << 16) - 1))
            else:
                raise vs.Error('fsrcnnx_upscale: "lmode" must be < 0, 0 or 1')
    else:
        limit = fsrcnnx



    if downscaler:
        scaled = downscaler(limit, width, height)
    else:
        scaled = limit


    if get_depth(scaled) != depth_src:
        out = depth(scaled, depth_src)
    else:
        out = scaled

    return out


def to_444(clip: vs.VideoNode, width: int = None, height: int = None, join_planes: bool = True)-> vs.VideoNode:
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




# # # # # # # # # # #
# Masking functions #
# # # # # # # # # # #

def diff_rescale_mask(source: vs.VideoNode, height: int = 720, kernel: str = 'bicubic',
                      b: float = 0, c: float = 1/2, mthr: int = 55,
                      mode: str = 'ellipse', sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """Modified version of Atomchtools for generate a mask with a rescaled difference.
       Its alias is vardefunc.drm

    Args:
        source (vs.VideoNode): Source clip.
        height (int, optional): Defaults to 720.
        kernel (str, optional): Defaults to bicubic.
        b (float, optional): Defaults to 0.
        c (float, optional): Defaults to 1/2.
        mthr (int, optional): Defaults to 55.
        mode (str, optional): Can be 'rectangle', 'losange' or 'ellipse' . Defaults to 'ellipse'.
        sw (int, optional): Growing/shrinking shape width. 0 is allowed. Defaults to 2.
        sh (int, optional): Growing/shrinking shape height. 0 is allowed. Defaults to 2.

    Returns:
        vs.VideoNode:
    """
    if not source.format.num_planes == 1:
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
                         start: int, end: int, sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """Modified version of Atomchtools for generate a mask with with a NC
       Its alias is vardefunc.dcm

    Args:
        source (vs.VideoNode): Source clip.
        titles (vs.VideoNode): Credit clip.
        nc (vs.VideoNode): Non credit clip.
        start (int, optional): Start frame.
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


def luma_credit_mask(source: vs.VideoNode, thr: int = 230, mode: str = 'prewitt', draft: bool = False)-> vs.VideoNode:
    """Creates a mask based on luma value and edges.

    Args:
        source (vs.VideoNode): Source clip.
        thr (int, optional): Luma value assuming 8 bit input. Defaults to 230.
        mode (str, optional): Chooses a predefined kernel used for the mask computing.
                              Valid choices are "sobel", "prewitt", "scharr", "kirsch",
                              "robinson", "roberts", "cartoon", "min/max", "laplace",
                              "frei-chen", "kayyali", "LoG", "FDOG" and "TEdge".
                              Defaults to 'prewitt'.
        draft (bool, optional): Allow to output the mask without growing. Defaults to False.

    Returns:
        vs.VideoNode: Credit mask.
    """
    try:
        import G41Fun as gf
    except ModuleNotFoundError:
        raise ModuleNotFoundError("luma_credit_mask: missing dependency 'G41Fun'")

    if not source.format.num_planes == 1:
        clip = get_y(source)
    else:
        clip = source

    edge_mask = gf.EdgeDetect(clip, mode).std.Maximum()
    luma_mask = core.std.Expr(clip, f'x {thr} > x 0 ?')

    credit_mask = core.std.Expr([edge_mask, luma_mask], 'x y min')

    if not draft:
        credit_mask = iterate(credit_mask, core.std.Maximum, 4)
        credit_mask = iterate(credit_mask, core.std.Inflate, 2)

    return credit_mask


def edge_detect(clip: vs.VideoNode, mode: str, thr: int, mpand: Tuple[int, int],
                **kwargs)-> vs.VideoNode:
    """Generates edge mask based on convolution kernel.

    Args:
        clip (vs.VideoNode): Source clip.
        mode (str): Chooses a predefined kernel used for the mask computing.
                    Valid choices are "sobel", "prewitt", "scharr", "kirsch",
                    "robinson", "roberts", "cartoon", "min/max", "laplace",
                    "frei-chen", "kayyali", "LoG", "FDOG" and "TEdge".
        thr (int): Binarize threshold.
        mpand (Tuple[int, int]): Iterate numbers of expand and inpand.

    Returns:
        vs.VideoNode: Return edge mask
    """
    try:
        import G41Fun as gf
    except ModuleNotFoundError:
        raise ModuleNotFoundError("edge_detect: missing dependency 'G41Fun'")


    mask = gf.EdgeDetect(clip, mode, **kwargs).std.Binarize(thr)

    coord = [1, 2, 1, 2, 2, 1, 2, 1]
    mask = iterate(mask, partial(core.std.Maximum, coordinates=coord), mpand[0])
    mask = iterate(mask, partial(core.std.Minimum, coordinates=coord), mpand[1])

    return mask.std.Inflate().std.Deflate()


def region_mask(clip: vs.VideoNode,
                left: int = 0, right: int = 0,
                top: int = 0, bottom: int = 0)-> vs.VideoNode:
    """Crop your mask

    Args:
        clip (vs.VideoNode): Source clip.
        left (int, optional): Left parameter in std.CropRel or std.Crop. Defaults to 0.
        right (int, optional): Right parameter in std.CropRel or std.Crop. Defaults to 0.
        top (int, optional): Top parameter in std.CropRel or std.Crop. Defaults to 0.
        bottom (int, optional): Bottom parameter in std.CropRel or std.Crop. Defaults to 0.

    Returns:
        vs.VideoNode: Cropped clip
    """
    crop = core.std.Crop(clip, left, right, top, bottom)
    return core.std.AddBorders(crop, left, right, top, bottom)




# # # # # # # # # # # # # # # # # # #
# Utility / calculation functions  #
# # # # # # # # # # # # # # # # # # #

def fade_filter(source: vs.VideoNode, clip_a: vs.VideoNode, clip_b: vs.VideoNode,
                start_f: int, end_f: int)-> vs.VideoNode:
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
        return core.std.Merge(clip_a, clip_b, n/length)

    function = partial(_fade, clip_a=clip_a[start_f:end_f+1], clip_b=clip_b[start_f:end_f+1], length=length)
    clip_fad = core.std.FrameEval(source[start_f:end_f+1], function)

    final = clip_fad

    if start_f != 0:
        final = source[:start_f] + final
    if end_f+1 < source.num_frames:
        final = final + source[end_f+1:]

    return final


def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode)-> vs.VideoNode:
    """Merge chroma from ref with luma

    Args:
        luma (vs.VideoNode): Source luma clip
        ref (vs.VideoNode): Source chroma clip

    Returns:
        vs.VideoNode:
    """
    return core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)


def get_chroma_shift(src_h: int, dst_h: int, aspect_ratio: float = 16/9)-> float:
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
            cmd[cmd.index(i)+ 1] = str(v)
        else:
            cmd.extend([i, str(v)])

    print("Encoder command: ", " ".join(cmd), "\n")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    process.communicate()

def set_ffms2_log_level(level: Union[str, int] = 0)-> None:
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


# # # # # #
# Aliases #
# # # # # #

drm = diff_rescale_mask
dcm = diff_creditless_mask
lcm = luma_credit_mask
gk = generate_keyframes
