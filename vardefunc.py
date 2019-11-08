import vapoursynth as vs
import fvsfunc as fvf
import kagefunc as kgf
from functools import partial
from vsutil import get_depth, get_y

core = vs.core

def fade_filter(source: vs.VideoNode, clipa: vs.VideoNode, clipb: vs.VideoNode, start=None, end=None, length=None)-> vs.VideoNode:

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


#It's basically adaptive_denoise with show_mask=True
def adaptive_mask(source: vs.VideoNode, luma_scaling=12) -> vs.VideoNode:
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
