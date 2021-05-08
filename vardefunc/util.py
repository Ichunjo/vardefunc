import vapoursynth as vs

core = vs.core


def get_sample_type(clip: vs.VideoNode) -> vs.SampleType:
    """[summary]

    Args:
        clip (vs.VideoNode): [description]

    Returns:
        [type]: [description]
    """
    return clip.format.sample_type
