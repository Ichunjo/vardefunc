"""
Various functions I use. Most of them are bad though.
"""
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
import vapoursynth as vs

core = vs.core

def fade_filter(source: vs.VideoNode, clipa: vs.VideoNode, clipb: vs.VideoNode,
                start: int = None, end: int = None, length: int = None)-> vs.VideoNode:
    """
    Apply a filter with a fade
    """

    if not length:
        length = end - start + 1

    length = length + 3

    black = core.std.BlankClip(source, format=vs.GRAY8, length=length, color=0)
    white = core.std.BlankClip(source, format=vs.GRAY8, length=length, color=255)

    fadmask = kgf.crossfade(black, white, length-1)

    fadmask = fadmask[2:-1]

    if kgf.get_depth(source) != 8:
        fadmask = fvf.Depth(fadmask, bits=kgf.get_depth(source))

    merged = source[:start] + core.std.MaskedMerge(clipa[start:end+1],
                                                   clipb[start:end+1], fadmask) + source[end+1:]
    return merged

def knlmcl(source: vs.VideoNode, h_y: float = 1.2, h_uv: float = 0.5,
           device_id: int = 0, depth: int = None)-> vs.VideoNode:
    """
    Denoise both luma and chroma with KNLMeansCL
    """

    if kgf.get_depth(source) != 32:
        clip = fvf.Depth(source, 32)
    else:
        clip = source

    denoise = core.knlm.KNLMeansCL(clip, a=2, h=h_y, d=3, device_type='gpu',
                                   device_id=device_id, channels='Y')
    denoise = core.knlm.KNLMeansCL(denoise, a=2, h=h_uv, d=3, device_type='gpu',
                                   device_id=device_id, channels='UV')
    if depth is not None:
        denoise = fvf.Depth(denoise, depth)

    return denoise

def diff_rescale_mask(source: vs.VideoNode, height: int = 720, kernel: str = 'bicubic',
                      b: float = 1/3, c: float = 1/3, mthr: int = 55,
                      mode: str = 'rectangle', sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """
    Modified version of Atomchtools for generate a mask with a rescaled difference
    """

    only_luma = source.format.num_planes == 1

    if not only_luma:
        clip = kgf.get_y(clip)

    width = kgf.get_w(height)
    desc = fvf.Resize(clip, width, height, kernel=kernel, a1=b, a2=c, invks=True)
    upsc = fvf.Depth(fvf.Resize(desc, source.width, source.height, kernel=kernel, a1=b, a2=c), 8)

    diff = core.std.MakeDiff(clip, upsc)
    mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
    mask = mask.std.Expr(f'x {mthr} < 0 x ?')
    mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    mask = hvf.mt_expand_multi(mask, mode=mode, sw=sw, sh=sh)

    if kgf.get_depth(source) != 8:
        mask = fvf.Depth(mask, bits=kgf.get_depth(source))
    return mask

def diff_creditless_mask(source: vs.VideoNode, titles: vs.VideoNode, nc: vs.VideoNode,
                         start: int = None, end: int = None,
                         sw: int = 2, sh: int = 2)-> vs.VideoNode:
    """
    Modified version of Atomchtools for generate a mask with with a NC
    """

    if kgf.get_depth(titles) != 8:
        titles = fvf.Depth(titles, 8)
    if kgf.get_depth(nc) != 8:
        nc = fvf.Depth(nc, 8)

    diff = core.std.MakeDiff(titles, nc, [0])
    diff = kgf.get_y(diff)
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

    if kgf.get_depth(source) != 8:
        credit_m = fvf.Depth(credit_m, bits=kgf.get_depth(source))
    return credit_m


# def F3kdbSep(src_y: vs.VideoNode, src_uv: vs.VideoNode,
#             range: int = None, y: int = None, c: int = None,
#             grainy: int = None, grainc: int = None,
#             mask: vs.VideoNode = None, neo_f3kdb: bool = True)-> List[vs.VideoNode]:

#     only_luma = src_y.format.num_planes == 1

#     if not only_luma:
#         src_y = kgf.get_y(src_y)

#     if kgf.get_depth(src_y) != 16:
#         src_y = fvf.Depth(src_y, 16)
#     if kgf.get_depth(src_uv) != 16:
#         src_uv = fvf.Depth(src_uv, 16)

#     if neo_f3kdb:
#         db_y = core.neo_f3kdb.Deband(src_y, range, y, grainy=grainy, sample_mode=4, preset='luma')
#         db_c = core.neo_f3kdb.Deband(src_uv, range, cb=c, cr=c, grainc=grainc, sample_mode=4, preset='chroma')
#     else:
#         db_y = core.f3kdb.Deband(src_y, range, y, grainy=grainy, output_depth=16, preset='luma')
#         db_c = core.f3kdb.Deband(src_uv, range, cb=c, cr=c, grainc=grainc, output_depth=16, preset='chroma')

#     if mask is not None:
#         if kgf.get_depth(mask) != 16:
#             mask = fvf.Depth(mask, 16)
#         if mask.height != src_y.height:
#             mask_y = core.resize.Bicubic(mask, src_y.width, src_y.height)
#         else:
#             mask_y = mask
#         db_y = core.std.MaskedMerge(db_y, src_y, mask_y, 0)

#         if mask.height != src_uv.height:
#             mask_c = core.resize.Bicubic(mask, src_uv.width, src_uv.height)
#         else:
#             mask_c = mask
#         db_c = core.std.MaskedMerge(db_c, src_uv, mask_c, [1, 2])

#     return db_y, db_c

def to444(clip, width: int = None, height: int = None, join: bool = True)-> vs.VideoNode:
    """
    Zastin’s nnedi3 chroma upscaler
    """

    def _nnedi3x2(clip):
        if hasattr(core, 'znedi3'):
            clip = clip.std.Transpose().znedi3.nnedi3(1, 1, 0, 0, 4, 2).std.Transpose().znedi3.nnedi3(0, 1, 0, 0, 4, 2)
        else:
            clip = clip.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 1).std.Transpose().nnedi3.nnedi3(0, 1, 0, 0, 3, 1)
        return clip

    chroma = [_nnedi3x2(c) for c in kgf.split(clip)[1:]]

    if width in (None, clip.width) and height in (None, clip.height):
        chroma = [core.fmtc.resample(c, sy=0.5, flt=0) for c in chroma]
    else:
        chroma = [core.resize.Spline36(c, width, height, src_top=0.5) for c in chroma]

    return core.std.ShufflePlanes([clip] + chroma, [0]*3, vs.YUV) if join else chroma

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
    src_w = kgf.get_w(src_h, aspect_ratio)
    dst_w = kgf.get_w(dst_h, aspect_ratio)

    ch_shift = 0.25 - 0.25 * (src_w / dst_w)
    ch_shift = float(round(ch_shift, 5))
    return ch_shift


drm = diff_rescale_mask
dcm = diff_creditless_mask
