import vapoursynth as vs
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
import descale as dsc
from functools import partial
from vsutil import *

core = vs.core

def fade_filter(source: vs.VideoNode, clipa: vs.VideoNode, clipb: vs.VideoNode, 
                start: int = None, end: int = None, length: int = None)-> vs.VideoNode:

    if not length:
        length = end - start + 1

    length = length + 3

    black = core.std.BlankClip(source, format=vs.GRAY8, length=length, color=0)
    white = core.std.BlankClip(source, format=vs.GRAY8, length=length, color=255)
    
    fadmask = kgf.crossfade(black, white, length-1)
    
    fadmask = fadmask[2:-1]
    
    if get_depth(source) != 8:
        fadmask = fvf.Depth(fadmask, bits=get_depth(source))

    merged = source[:start]+core.std.MaskedMerge(clipa[start:end+1], clipb[start:end+1], fadmask)+source[end+1:]
    return merged


#It's basically adaptive_grain of kagefunc with show_mask=True
def adaptive_mask(source: vs.VideoNode, luma_scaling: int = 12)-> vs.VideoNode:
    import numpy as np
    if get_depth(source) != 8:
        clip = fvf.Depth(source, bits=8)
    else:
        clip = source
    def fill_lut(y):
        x = np.arange(0, 1, 1 / (1 << 8))
        z = (1 - (x * (1.124 + x * (-9.466 + x * (36.624 + x * (-45.47 + x * 18.188)))))) ** ((y ** 2) * luma_scaling)
        if clip.format.sample_type == vs.INTEGER:
            z = z * 255
            z = np.rint(z).astype(int)
        return z.tolist()

    def generate_mask(n, f, clip):
        frameluma = round(f.props.PlaneStatsAverage * 999)
        table = lut[int(frameluma)]
        return core.std.Lut(clip, lut=table)

    lut = [None] * 1000
    for y in np.arange(0, 1, 0.001):
        lut[int(round(y * 1000))] = fill_lut(y)

    luma = get_y(fvf.Depth(clip, 8)).std.PlaneStats()

    mask = core.std.FrameEval(luma, partial(generate_mask, clip=luma), prop_src=luma)
    mask = core.resize.Spline36(mask, clip.width, clip.height)

    if get_depth(source) != 8:
        mask = fvf.Depth(mask, bits=get_depth(source))
    return mask

def KNLMCL(source: vs.VideoNode, h_Y: float = 1.2, h_UV: float = 0.5, device_id: int = 0, depth: int = None)-> vs.VideoNode:
  
    if get_depth(source) != 32:
        clip = fvf.Depth(source, 32)
    
    denoise = core.knlm.KNLMeansCL(clip, a=2, h=h_Y, d=3, device_type='gpu', device_id=device_id, channels='Y')
    denoise = core.knlm.KNLMeansCL(denoise, a=2, h=h_UV, d=3, device_type='gpu', device_id=device_id, channels='UV')

    if depth is not None:
        denoise = fvf.Depth(denoise, depth)
    
    return denoise

def DiffRescaleMask(source: vs.VideoNode, h: int = 720, kernel: str = 'bicubic', 
                    b:float = 1/3, c:float = 1/3, mthr: int = 55, 
                    mode: str = 'rectangle', sw: int = 2, sh: int = 2)-> vs.VideoNode:

    only_luma = source.format.num_planes == 1

    if get_depth(source) != 8:
        clip = fvf.Depth(source, 8)
    else:
        clip = source

    if not only_luma:
        clip = get_y(clip)

    w = get_w(h)
    desc = fvf.Resize(clip, w, h, kernel=kernel, a1=b, a2=c, invks=True)
    upsc = fvf.Depth(fvf.Resize(desc, w, h, kernel=kernel, a1=b, a2=c), 8)
    
    diff = core.std.MakeDiff(clip, upsc)
    mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
    mask = mask.std.Expr('x {thr} < 0 x ?'.format(thr=mthr))
    mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    mask = hvf.mt_expand_multi(mask, mode=mode, sw=sw, sh=sh)
    if get_depth(source) != 8:
        mask = fvf.Depth(mask, bits=get_depth(source))
    return mask

DRM = DiffRescaleMask

def F3kdbSep(src_l: vs.VideoNode, src_c: vs.VideoNode, 
            range: int = None, y: int = None, c: int = None,
            grainy: int = None, grainc: int = None,
            mask: vs.VideoNode = None)-> List[vs.VideoNode]:

    only_luma = src_l.format.num_planes == 1

    if not only_luma:
        src_l = get_y(src_l)

    if get_depth(src_l) != 16:
        src_l = fvf.Depth(src_l, 16)
    if get_depth(src_c) != 16:
        src_c = fvf.Depth(src_c, 16)

    db_y = core.f3kdb.Deband(src_l, range, y, grainy=grainy, output_depth=16, preset='luma')
    db_c = core.f3kdb.Deband(src_c, range, cb=c, cr=c, grainc=grainc, output_depth=16, preset='chroma')

    if mask is not None:
        if get_depth(mask) != 16:
            mask = fvf.Depth(mask, 16)
        if mask.height != src_l.height:
            mask_y = core.resize.Bicubic(mask, src_l.width, src_l.height)
        else:
            mask_y = mask
        db_y = core.std.MaskedMerge(db_y, src_l, mask_y, 0)

        if mask.height != src_c.height:
            mask_c = core.resize.Bicubic(mask, src_c.width, src_c.height)
        else:
            mask_c = mask
        db_c = core.std.MaskedMerge(db_c, src_c, mask_c, [1, 2])

    return db_y, db_c
