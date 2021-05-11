"""(Up/De)scaling functions"""
from functools import partial
from typing import Any, Callable, Dict

from vsutil import depth, get_depth, get_w, get_y, split
import havsfunc as hvf
import vapoursynth as vs

from . import placebo
from .sharp import z4USM

core = vs.core


def nnedi3cl_double(clip: vs.VideoNode, znedi: bool = True,
                    scaler: Callable[[vs.VideoNode, Any], vs.VideoNode] = None,
                    correct_shift: bool = True, **nnedi3_args) -> vs.VideoNode:
    """Double the clip using nnedi3 for even frames and nnedi3cl for odd frames
       Intended to speed up encoding speed without hogging the GPU either

    Args:
        clip (vs.VideoNode): Source clip.
        znedi (bool, optional): Use znedi3 or not. Defaults to True.
        scaler (Callable[[vs.VideoNode, Any], vs.VideoNode], optional):
            Resizer used to correct the shift. Defaults to core.resize.Bicubic.
        correct_shift (bool, optional): Defaults to True.

    Returns:
        vs.VideoNode:
    """
    nnargs: Dict[str, Any] = dict(nsize=4, nns=4, qual=2, pscrn=2)
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

    if scaler is None:
        scaler = core.resize.Bicubic

    return scaler(clip, src_top=.5, src_left=.5) if correct_shift else clip


def nnedi3_upscale(clip: vs.VideoNode, scaler: Callable[[vs.VideoNode, Any], vs.VideoNode] = None,
                   correct_shift: bool = True, znedi: bool = False, **nnedi3_args) -> vs.VideoNode:
    """Classic based nnedi3 upscale.

    Args:
        clip (vs.VideoNode): Source clip.
        scaler (Callable[[vs.VideoNode, Any], vs.VideoNode], optional):
            Resizer used to correct the shift. Defaults to core.resize.Bicubic.
        correct_shift (bool, optional): Defaults to True.
        znedi (bool, optional): Use znedi3 or not. Defaults to False.

    Returns:
        vs.VideoNode: Upscaled clip.
    """
    nnargs: Dict[str, Any] = dict(nsize=4, nns=4, qual=2, pscrn=2)
    nnargs.update(nnedi3_args)

    if znedi:
        clip = clip.std.Transpose().znedi3.nnedi3(0, True, **nnargs).std.Transpose().znedi3.nnedi3(0, True, **nnargs)
    else:
        clip = clip.std.Transpose().nnedi3.nnedi3(0, True, **nnargs).std.Transpose().nnedi3.nnedi3(0, True, **nnargs)

    if scaler is None:
        scaler = core.resize.Bicubic

    return scaler(clip, src_top=.5, src_left=.5) if correct_shift else clip


def eedi3_upscale(clip: vs.VideoNode, scaler: Callable[[vs.VideoNode, Any], vs.VideoNode] = None,
                  correct_shift: bool = True, nnedi3_args: Dict[str, Any] = None, eedi3_args: Dict[str, Any] = None) -> vs.VideoNode:
    """
    Upscale function using the power of eedi3 and nnedi3.
    Eedi3 default values are the safest and should work for anything without introducing any artifacts
    except for very specific shrinking pattern.

    Args:
        clip (vs.VideoNode): Source clip.
        scaler (Callable[[vs.VideoNode, Any], vs.VideoNode], optional): Resizer used to correct the shift. Defaults to core.resize.Bicubic.
        correct_shift (bool, optional): Defaults to True.
        nnedi3_args (Dict[str, Any], optional): Defaults to dict(nsize=4, nns=4, qual=2, etype=1, pscrn=1).
        eedi3_args (Dict[str, Any], optional): Defaults to dict(alpha=0.2, beta=0.8, gamma=1000, nrad=1, mdis=15).

    Returns:
        vs.VideoNode: Upscaled clip.
    """
    nnargs: Dict[str, Any] = dict(nsize=4, nns=4, qual=2, etype=1, pscrn=1)
    if nnedi3_args:
        nnargs.update(nnedi3_args)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.8, gamma=1000, nrad=1, mdis=15)
    if eedi3_args:
        eeargs.update(eedi3_args)

    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)
    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)

    if scaler is None:
        scaler = core.resize.Bicubic

    return scaler(clip, src_top=.5, src_left=.5) if correct_shift else clip


def fsrcnnx_upscale(source: vs.VideoNode, width: int = None, height: int = 1080, shader_file: str = None,
                    downscaler: Callable[[vs.VideoNode, int, int], vs.VideoNode] = core.resize.Bicubic,
                    upscaler_smooth: Callable[[vs.VideoNode, Any], vs.VideoNode] = partial(nnedi3_upscale, nsize=4, nns=4, qual=2, pscrn=2),
                    strength: float = 100.0, profile: str = 'slow',
                    lmode: int = 1, overshoot: int = None, undershoot: int = None,
                    sharpener: Callable[[vs.VideoNode, Any], vs.VideoNode] = partial(z4USM, radius=2, strength=65)
                    ) -> vs.VideoNode:
    """
    Upscale the given luma source clip with FSRCNNX to a given width / height
    while preventing FSRCNNX artifacts by limiting them.

    Args:
        source (vs.VideoNode): Source clip, assuming this one is perfectly descaled.
        width (int): Target resolution width. Defaults to None.
        height (int): Target resolution height. Defaults to 1080.
        shader_file (str): Path to the FSRCNNX shader file. Defaults to None.

        downscaler (Callable[[vs.VideoNode, int, int], vs.VideoNode], optional):
            Resizer used to downscale the upscaled clip. Defaults to core.resize.Bicubic.

        upscaler_smooth (Callable[[vs.VideoNode, Any], vs.VideoNode], optional):
            Resizer used to replace the smoother nnedi3 upscale.
            Defaults to partial(nnedi3_upscale, nsize=4, nns=4, qual=2, pscrn=2).

        strength (float):
            Only for profile='slow'.
            Strength between the smooth upscale and the fsrcnnx upscale where 0.0 means the full smooth clip
            and 100.0 means the full fsrcnnx clip. Negative and positive values are possible, but not recommended.

        profile (str): Profile settings. Possible strings: "fast", "old", "slow" or "zastin".
                       – "fast" is the old draft mode (the plain fsrcnnx clip returned).
                       – "old" is the old mode to deal with the bright pixels.
                       – "slow" is the new mode, more efficient.
                       – "zastin" is a combination between a sharpened nnedi3 upscale and a fsrcnnx upscale.
                         The sharpener prevents the interior of lines from being brightened and fsrnncx
                         (as a clamping clip without nnedi3) prevents artifacting (halos) from the sharpening.

        lmode (int): Only for profile='slow':
                     – (< 0): Limit with rgvs.Repair (ex: lmode=-1 --> rgvs.Repair(1), lmode=-5 --> rgvs.Repair(5) ...)
                     – (= 0): No limit.
                     – (= 1): Limit to over/undershoot.

        overshoot (int): Limit for pixels that get brighter during upscaling.
        undershoot (int): Limit for pixels that get darker during upscaling.

        sharpener (Callable[[vs.VideoNode, Any], vs.VideoNode], optional):
            Sharpening function used to replace the sharped smoother nnedi3 upscale.
            Defaults to partial(z4USM, radius=2, strength=65)

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
        width = get_w(height, clip.width / clip.height)
    if overshoot is None:
        overshoot = strength / 100
    if undershoot is None:
        undershoot = overshoot

    profiles = ['fast', 'old', 'slow', 'zastin']
    if profile not in profiles:
        raise vs.Error('fsrcnnx_upscale: "profile" must be "fast", "old", "slow" or "zastin"')
    num = profiles.index(profile.lower())

    fsrcnnx = placebo.shader(clip, clip.width * 2, clip.height * 2, shader_file)

    if num >= 1:
        # old or slow profile
        smooth = upscaler_smooth(clip)
        if num == 1:
            # old profile
            limit = core.std.Expr([fsrcnnx, smooth], 'x y min')
        elif num == 2:
            # slow profile
            upscaled = core.std.Expr([fsrcnnx, smooth], 'x {strength} * y 1 {strength} - * +'.format(strength=strength / 100))
            if lmode < 0:
                limit = core.rgvs.Repair(upscaled, smooth, abs(lmode))
            elif lmode == 0:
                limit = upscaled
            elif lmode == 1:
                dark_limit = core.std.Minimum(smooth)
                bright_limit = core.std.Maximum(smooth)

                limit = hvf.Clamp(upscaled, bright_limit, dark_limit,
                                  hvf.scale(overshoot, (1 << 16) - 1),
                                  hvf.scale(undershoot, (1 << 16) - 1))
            else:
                raise vs.Error('fsrcnnx_upscale: "lmode" must be < 0, 0 or 1')
        else:
            # zastin profile
            smooth_sharp = sharpener(smooth)
            limit = core.std.Expr([smooth, fsrcnnx, smooth_sharp], 'x y z min max y z max min')
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


def to_444(clip: vs.VideoNode,
           width: int = None, height: int = None,
           join_planes: bool = True, znedi: bool = True,
           scaler: Callable[[vs.VideoNode, Any], vs.VideoNode] = None) -> vs.VideoNode:
    """Zastin’s nnedi3 chroma upscaler. Modified by Vardë.

    Args:
        clip ([type]): Source clip
        width (int, optional): Upscale width. Defaults to None.
        height (int, optional): Upscale height. Defaults to None.
        join_planes (bool, optional): Defaults to True.
        znedi (bool, optional): Use znedi3 or not. Defaults to True.
        scaler (Callable[[vs.VideoNode, Any], vs.VideoNode], optional):
            Resizer used to correct the shift. Defaults to core.resize.Bicubic.

    Returns:
        vs.VideoNode: 444’d clip.
    """
    def _nnedi3x2(clip):
        if znedi:
            clip = clip.std.Transpose().znedi3.nnedi3(1, 1, 0, 0, 4, 2) \
                .std.Transpose().znedi3.nnedi3(0, 1, 0, 0, 4, 2)
        else:
            clip = clip.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 1) \
                .std.Transpose().nnedi3.nnedi3(0, 1, 0, 0, 3, 1)
        return clip

    chroma = [_nnedi3x2(c) for c in split(clip)[1:]]

    if scaler is None:
        scaler = core.resize.Bicubic

    if width in (None, clip.width) and height in (None, clip.height):
        chroma = [scaler(c, src_top=0.5) for c in chroma]
    else:
        chroma = [scaler(c, width, height, src_top=0.5) for c in chroma]

    return core.std.ShufflePlanes([clip] + chroma, [0] * 3, vs.YUV) if join_planes else chroma
