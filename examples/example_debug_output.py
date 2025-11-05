import vapoursynth as vs
from vsutil import split

from vardefunc.misc import DebugOutput

core = vs.core

# Import your clip
SOURCE = core.std.BlankClip(format=vs.YUV420P16)


# Initialise the DebugOutput object
DEBUG = DebugOutput(SOURCE, props=7, num=9, scale=1)
# DEBUG = DebugOutput((0, SOURCE))
# DEBUG = DebugOutput(source=SOURCE)
# DEBUG = DebugOutput(source=(0, SOURCE))
# DEBUG = DebugOutput()


@DEBUG.catch  # Catch the output of main_filter(). Here it's the grained clip
def main_filter() -> vs.VideoNode:
    debug = DEBUG
    src = SOURCE

    den = denoise(src)
    debug <<= den  # Add the den clip from the biggest index

    db = deband(den)
    debug <<= {"deband": db}  # Add the named den clip from the biggest index

    grained = grain(db)
    debug <<= split(grained)  # Add grained den planes from the biggest index

    return grained


def denoise(clip: vs.VideoNode) -> vs.VideoNode:
    ...


def deband(clip: vs.VideoNode) -> vs.VideoNode:
    ...


def grain(clip: vs.VideoNode) -> vs.VideoNode:
    ...



if __name__ == "__main__":
    pass
else:
    filtered = main_filter()
