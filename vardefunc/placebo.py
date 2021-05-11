"""Placebo wrapper"""
from typing import List, Union
from vsutil import depth, join, split, get_y

import vapoursynth as vs
core = vs.core


def deband(clip: vs.VideoNode, radius: float = 16.0,
           threshold: Union[float, List[float]] = 4.0, iterations: int = 1,
           grain: Union[float, List[float]] = 6.0, chroma: bool = True, **kwargs) -> vs.VideoNode:
    """Wrapper for placebo.Deband
       https://github.com/Lypheo/vs-placebo#vs-placebo

    Args:
        clip (vs.VideoNode): Source clip.

        radius (float, optional):
            The debanding filter's initial radius. The radius increases linearly for each iteration.
            A higher radius will find more gradients, but a lower radius will smooth more aggressively.
            Defaults to 16.0.

        threshold (Union[float, List[float]], optional):
            The debanding filter's cut-off threshold.
            Higher numbers increase the debanding strength dramatically,
            but progressively diminish image details. Defaults to 4.0.

        iterations (int, optional):
            The number of debanding steps to perform per sample. Each step reduces a bit more banding,
            but takes time to compute. Note that the strength of each step falls off very quickly,
            so high numbers (>4) are practically useless. Defaults to 1.

        grain (Union[float, List[float]], optional):
            Add some extra noise to the image.
            This significantly helps cover up remaining quantization artifacts.
            Higher numbers add more noise.

            Note: When debanding HDR sources, even a small amount of grain can result
            in a very big change to the brightness level.
            It's recommended to either scale this value down or disable it entirely for HDR.
            Defaults to 6.0, which is very mild.

        chroma (bool, optional): Process chroma planes or not.
            Defaults to True if the input clip has chroma planes.

    Returns:
        vs.VideoNode: Debanded clip.
    """
    threshold = [threshold] * 3 if isinstance(threshold, float) else threshold + [threshold[-1]] * (3 - len(threshold))
    grain = [grain] * 3 if isinstance(grain, float) else grain + [grain[-1]] * (3 - len(grain))

    if chroma and clip.format.num_planes > 1:
        planes = split(clip)

        for i, (thr, gra) in enumerate(zip(threshold, grain)):
            planes[i] = planes[i].placebo.Deband(1, iterations, thr, radius, gra, **kwargs)
        clip = join(planes)
    else:
        clip = clip.placebo.Deband(1, iterations, threshold[0], radius, grain[0], **kwargs)

    return clip



def shader(clip: vs.VideoNode, width: int, height: int, shader_file: str, luma_only: bool = True, **kwargs) -> vs.VideoNode:
    """Wrapper for placebo.Resample
       https://github.com/Lypheo/vs-placebo#vs-placebo

    Args:
        clip (vs.VideoNode): Source clip.

        width (int): Destination width.

        height (int): Destination height.

        shader_file (str):
            Path to shader file used into placebo.Shader.

        luma_only (bool, optional):
            If process the luma only. Defaults to True.

    Returns:
        vs.VideoNode: Shader'd clip.
    """
    clip = depth(clip, 16)

    if luma_only:
        filter_shader = 'box'
        if clip.format.num_planes == 1:
            if width > clip.width or height > clip.height:
                clip = clip.resize.Point(format=vs.YUV444P16)
            else:
                if width % 4 == 0 and height % 4 == 0:
                    blank = core.std.BlankClip(clip, int(clip.width / 4), int(clip.height / 4), vs.GRAY16)
                elif width % 2 == 0 and height % 2 == 0:
                    blank = core.std.BlankClip(clip, int(clip.width / 2), int(clip.height / 2), vs.GRAY16)
                else:
                    blank = core.std.BlankClip(clip, vs.GRAY16)
                clip = join([clip, blank, blank])
    else:
        filter_shader = 'ewa_lanczos'

    clip = core.placebo.Shader(clip, shader_file, width, height, filter=filter_shader, **kwargs)

    return get_y(clip) if luma_only else clip
