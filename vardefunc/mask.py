"""Random masking functions"""

__all__ = [
    "cambi_mask",
]

from typing import Any

from vskernels import Bilinear, Scaler, ScalerLike
from vsrgtools import MeanMode, box_blur
from vstools import DitherType, VSFunction, core, depth, get_depth, vs


def cambi_mask(
    clip: vs.VideoNode,
    scale: int = 1,
    merge_previous: bool = True,
    blur_func: VSFunction = lambda clip: box_blur(clip, 2, 3, planes=0),
    scaler: ScalerLike = Bilinear,
    **cambi_args: Any,
) -> vs.VideoNode:
    """Generate a deband mask

    :param clip:            Input clip
    :param scale:           0 <= i < 5, defaults to 1
    :param merge_previous:  Will merge the GRAYS cscore frame stored as frame property for scale 0 <= i < 5, defaults to True
    :param blur_func:       A bluring function called on the mask, defaults to lambdaclip:box_blur(clip, 2, 3, 0)
    :param scaler:          Scaler used to resize the cscore frames, defaults to Bilinear
    :return:                GRAY float deband mask
    """
    if get_depth(clip) > 10:
        clip = depth(clip, 10, dither_type=DitherType.NONE)

    scores = core.akarin.Cambi(clip, scores=True, **cambi_args)
    if merge_previous:
        cscores = [
            blur_func(scores.std.PropToClip(f"CAMBI_SCALE{i}").std.Deflate().std.Deflate()) for i in range(0, scale + 1)
        ]
        scaler = Scaler.ensure_obj(scaler)

        deband_mask = MeanMode.ARITHMETIC(
            (scaler.scale(c, scores.width, scores.height) for c in cscores), func=cambi_mask
        )
    else:
        deband_mask = blur_func(scores.std.PropToClip(f"CAMBI_SCALE{scale}").std.Deflate().std.Deflate())

    return deband_mask.std.CopyFrameProps(scores)
