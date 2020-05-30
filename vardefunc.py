"""
Various functions I use. Most of them are bad though.
"""
from functools import partial
from vsutil import *

import fvsfunc as fvf
import havsfunc as hvf

import vapoursynth as vs

core = vs.core

def fade_filter(source: vs.VideoNode, clip_a: vs.VideoNode, clip_b: vs.VideoNode,
                start_f: int = None, end_f: int = None)-> vs.VideoNode:
    """
    Apply a filter with a fade
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
    """
    Denoise both luma and chroma with KNLMeansCL
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
                      b: float = 1/3, c: float = 1/3, mthr: int = 55,
                      mode: str = 'rectangle', sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """
    Modified version of Atomchtools for generate a mask with a rescaled difference
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
    """
    Modified version of Atomchtools for generate a mask with with a NC
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

def nnedi3cl_double(clip: vs.VideoNode, znedi: bool = True, **args) -> vs.VideoNode:
    """
    Double the clip using nnedi3 for even frames and nnedi3cl for odd frames
    Intended to speed up encoding speed without hogging the GPU either
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

def to444(clip, width: int = None, height: int = None, join_planes: bool = True)-> vs.VideoNode:
    """
    Zastin’s nnedi3 chroma upscaler
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
    """
    Crop your mask
    """
    crop = core.std.Crop(clip, left, right, top, bottom)
    borders = core.std.AddBorders(crop, left, right, top, bottom)
    return borders

def get_chroma_shift(src_h: int = None, dst_h: int = None,
                     aspect_ratio: float = 16/9) -> float:
    """
    Intended to calculate the right value for chroma shifting
    """
    src_w = get_w(src_h, aspect_ratio)
    dst_w = get_w(dst_h, aspect_ratio)

    ch_shift = 0.25 - 0.25 * (src_w / dst_w)
    ch_shift = float(round(ch_shift, 5))
    return ch_shift


drm = diff_rescale_mask
dcm = diff_creditless_mask
