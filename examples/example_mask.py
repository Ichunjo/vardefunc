import vapoursynth as vs
from vardefunc.mask import FreyChen

core = vs.core

# Import your clip
SOURCE = core.std.BlankClip(format=vs.YUV420P16)


def filtering() -> vs.VideoNode:
    clip = SOURCE
    # Use a EdgeDetect mask
    mask = FreyChen().get_mask(clip, lthr=4000, hthr=8000, multi=1.5)
    return mask


if __name__ == '__main__':
    pass
else:
    filtered = filtering()
    filtered.set_output(0)
