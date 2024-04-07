"""(Up/De)scaling functions"""

__all__ = ['fsrcnnx_upscale']

from functools import partial
from typing import Callable, Optional

import vapoursynth as vs

from vsaa import Nnedi3
from vsscale import PlaceboShader
from vstools import depth, get_depth, get_w, get_y

from .sharp import z4usm

core = vs.core


def fsrcnnx_upscale(clip: vs.VideoNode, width: Optional[int] = None, height: int = 1080, shader_file: Optional[str] = None,
                    downscaler: Optional[Callable[[vs.VideoNode, int, int], vs.VideoNode]] = core.resize.Bicubic,
                    upscaled_smooth: Optional[vs.VideoNode] = None,
                    strength: float = 100.0, profile: str = 'slow',
                    lmode: int = 1, overshoot: Optional[float] = None, undershoot: Optional[float] = None,
                    sharpener: Callable[[vs.VideoNode], vs.VideoNode] = partial(z4usm, radius=2, strength=65)
                    ) -> vs.VideoNode:
    """
    Upscale the given luma source clip with FSRCNNX to a given width / height
    while preventing FSRCNNX artifacts by limiting them.

    Args:
        source (vs.VideoNode):
            Source clip, assuming this one is perfectly descaled.

        width (int):
            Target resolution width (if None, auto-calculated). Defaults to None.

        height (int):
            Target resolution height. Defaults to 1080.

        shader_file (str):
            Path to the FSRCNNX shader file. Defaults to None.

        downscaler (Callable[[vs.VideoNode, int, int], vs.VideoNode], optional):
            Resizer used to downscale the upscaled clip. Defaults to core.resize.Bicubic.

        upscaled_smooth (Optional[vs.VideoNode]):
            Smooth doubled clip. If not provided, will use nnedi3_upscale(source).

        strength (float):
            Only for profile='slow'.
            Strength between the smooth upscale and the fsrcnnx upscale where 0.0 means the full smooth clip
            and 100.0 means the full fsrcnnx clip. Negative and positive values are possible, but not recommended.

        profile (str): Profile settings. Possible strings: "fast", "old", "slow" or "zastin".
                       – "fast" is the old draft mode (the plain fsrcnnx clip returned).
                       – "old" is the old mode to deal with the bright pixels.
                       – "slow" is the new mode, more efficient, using clamping.
                       – "zastin" is a combination between a sharpened nnedi3 upscale and a fsrcnnx upscale.
                         The sharpener prevents the interior of lines from being brightened and fsrnncx
                         (as a clamping clip without nnedi3) prevents artifacting (halos) from the sharpening.

        lmode (int): Only for profile='slow':
                     – (< 0): Limit with rgvs.Repair (ex: lmode=-1 --> rgvs.Repair(1), lmode=-5 --> rgvs.Repair(5) ...)
                     – (= 0): No limit.
                     – (= 1): Limit to over/undershoot.

        overshoot (float):
            Only for profile='slow'.
            Limit for pixels that get brighter during upscaling.

        undershoot (float):
            Only for profile='slow'.
            Limit for pixels that get darker during upscaling.

        sharpener (Callable[[vs.VideoNode, Any], vs.VideoNode], optional):
            Only for profile='zastin'.
            Sharpening function used to replace the sharped smoother nnedi3 upscale.
            Defaults to partial(z4USM, radius=2, strength=65)

    Returns:
        vs.VideoNode: Upscaled luma clip.
    """
    bits = get_depth(clip)

    clip = depth(get_y(clip), 16)

    if width is None:
        width = get_w(height, clip.width / clip.height)
    if overshoot is None:
        overshoot = strength / 100
    if undershoot is None:
        undershoot = overshoot

    profiles = ['fast', 'old', 'slow', 'zastin']
    if profile not in profiles:
        raise ValueError('fsrcnnx_upscale: "profile" must be "fast", "old", "slow" or "zastin"')
    num = profiles.index(profile.lower())

    if not shader_file:
        raise ValueError('fsrcnnx_upscale: You must set a string path for "shader_file"')

    fsrcnnx = PlaceboShader(shader_file).multi(clip, 2)

    if num >= 1:
        # old or slow profile
        smooth = depth(get_y(upscaled_smooth), 16) if upscaled_smooth else Nnedi3.multi(clip, 2)
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

                overshoot *= 2**8
                undershoot *= 2**8
                limit = core.std.Expr(
                    [upscaled, bright_limit, dark_limit],
                    f'x y {overshoot} + > y {overshoot} + x ? z {undershoot} - < z {undershoot} - x y {overshoot} + > y {overshoot} + x ? ?'
                )
            else:
                raise ValueError('fsrcnnx_upscale: "lmode" must be < 0, 0 or 1')
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

    return depth(scaled, bits)
