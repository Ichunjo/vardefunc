from vsutil import depth, join, split, get_depth, get_y
import vapoursynth as vs
core = vs.core


def deband(clip: vs.VideoNode, radius: int = 17, threshold: float = 4,
           iterations: int = 1, grain: float = 4, chroma: bool = True)-> vs.VideoNode:
    """Wrapper for placebo.Deband

    Args:
        clip (vs.VideoNode):
        radius (int, optional): Defaults to 17.
        threshold (float, optional): Defaults to 4.
        iterations (int, optional): Defaults to 1.
        grain (float, optional): Defaults to 4.
        chroma (bool, optional): Defaults to True.

    Returns:
        vs.VideoNode
    """
    if get_depth(clip) != 16:
        clip = depth(clip, 16)
    if chroma is True:
        clip = join([core.placebo.Deband(x, 1, iterations, threshold, radius, grain)
                     for x in split(clip)])
    else:
        clip = core.placebo.Deband(clip, 1, iterations, threshold, radius, grain)
    return clip

def shader(clip: vs.VideoNode, width: int, height: int, shader_file: str, luma_only: bool = True, **args)-> vs.VideoNode:
    """Wrapper for placebo.Resample

    Args:
        clip (vs.VideoNode): Source clip
        width (int): Destination width
        height (int): Destination height
        shader_file (str): Shader used into Resample
        luma_only (bool, optional): If process the luma only. Defaults to True.
    Returns:
        vs.VideoNode:
    """
    if get_depth(clip) != 16:
        clip = depth(clip, 16)
    if luma_only is True:
        filter_shader = 'box'
        if clip.format.num_planes == 1:
            if width > clip.width or height > clip.height:
                clip = clip.resize.Point(format=vs.YUV444P16)
            else:
                blank = core.std.BlankClip(clip, clip.width/4, clip.height/4, vs.GRAY16)
                clip = join([clip, blank, blank])
    else:
        filter_shader = 'ewa_lanczos'

    clip = core.placebo.Shader(clip, shader_file, width, height, filter=filter_shader, **args)

    return get_y(clip) if luma_only is True else clip
