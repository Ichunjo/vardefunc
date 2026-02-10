"""(Up/De)scaling functions"""

from __future__ import annotations

from typing import Any, ClassVar

from jetpytools import KwargsT, SPathLike
from vskernels import Catrom, EwaLanczos, KernelLike, Placebo, Scaler, ScalerLike
from vskernels.types import LeftShift, TopShift
from vsscale import ArtCNN, PlaceboShader
from vstools import (
    ChromaLocation,
    ColorRange,
    DitherType,
    check_variable,
    core,
    depth,
    get_u,
    get_v,
    join,
    split,
    vs,
)

from .vsjet_proxy import is_preview

__all__ = ["EwaLanczosChroma", "mpv_preview"]


class PlHermite(Placebo):
    _kernel = "hermite"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(None, 0, 0, **kwargs)


class EwaLanczosChroma(EwaLanczos):
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift | list[TopShift], LeftShift | list[LeftShift]] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        assert check_variable(clip, self.__class__)
        assert clip.format.color_family is vs.YUV

        u, v = get_u(clip), get_v(clip)
        left_shift, top_shift = ChromaLocation.from_video(clip).get_offsets(clip)
        left_shift *= u.width / clip.width
        top_shift *= u.height / clip.height

        u = super().scale(u, clip.width, clip.height, (-top_shift, -left_shift), **kwargs)
        v = super().scale(v, clip.width, clip.height, (-top_shift, -left_shift), **kwargs)

        return core.std.ShufflePlanes([clip, u, v], [0, 0, 0], vs.YUV, clip)


class ArtCNNShaderBase(PlaceboShader):
    shader_file: ClassVar[SPathLike]

    def __init__(
        self,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(self.shader_file, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)


class ArtCNNShader(ArtCNNShaderBase):
    shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F16.glsl"

    class C4F16(ArtCNNShaderBase):
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F16.glsl"

    class C4F32(ArtCNNShaderBase):
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F32.glsl"

    class C4F16_Chroma(ArtCNNShaderBase):  # noqa: N801
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F16_Chroma.glsl"

    class C4F32_Chroma(ArtCNNShaderBase):  # noqa: N801
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F32_Chroma.glsl"


def mpv_preview(
    clip: vs.VideoNode,
    w: int | None = None,
    h: int | None = None,
    scale: ScalerLike = ArtCNNShader.C4F16,
    dscale: ScalerLike = PlHermite(linearize=True),
    cscale: ScalerLike = EwaLanczos(antiring=0.6),
    dither_type: DitherType = DitherType.ORDERED,
) -> vs.VideoNode:
    clip = depth(clip, 16)

    scale = Scaler.ensure_obj(scale)
    dscale = Scaler.ensure_obj(dscale)
    cscale = Scaler.ensure_obj(cscale)

    props = KwargsT()

    planes = list[vs.VideoNode](split(clip))

    if w and h and (w, h) > (clip.width, clip.height):
        if isinstance(scale, ArtCNNShader):
            planes[0] = scale.supersample(planes[0], 2)

            if (planes[0].width, planes[0].height) > (w, h):
                planes[0] = dscale.scale(planes[0], w, h)

                if is_preview():
                    props["PreviewDscale"] = dscale.__class__.__name__
        else:
            planes[0] = scale.scale(planes[0], w, h)

        if is_preview():
            props["PreviewScale"] = scale.__class__.__name__

    if isinstance(cscale, (ArtCNNShader, ArtCNN)) and (w, h) == (None, None):
        preview = cscale.scale(clip.resize.Bilinear(format=vs.YUV444P16))
    else:
        left_shift, top_shift = ChromaLocation.from_video(clip).get_offsets(clip)
        left_shift *= planes[1].width / planes[0].width
        top_shift *= planes[1].height / planes[0].height

        planes[1] = cscale.scale(planes[1], planes[0].width, planes[0].height, (-top_shift, -left_shift))
        planes[2] = cscale.scale(planes[2], planes[0].width, planes[0].height, (-top_shift, -left_shift))

        preview = core.std.CopyFrameProps(join(planes), clip)

    if is_preview():
        props["PreviewCscale"] = cscale.__class__.__name__

    return dither_type.apply(
        preview,
        preview.format.replace(color_family=vs.RGB, sample_type=vs.INTEGER, bits_per_sample=8),
        ColorRange.from_video(preview),
        ColorRange.FULL,
        False,
    ).std.SetFrameProps(**props)


# class MixedRescale:
#     upscaled: vs.VideoNode

#     def __init__(self, src: vs.VideoNode, *rescales: Rescale) -> None:
#         prop_srcs = [rs.diff(src) for rs in rescales]
#         rescales_idx = tuple(range(len(rescales)))

#         blank = core.std.BlankClip(None, 1, 1, vs.GRAY8, src.num_frames, keep=True)

#         map_prop_srcs = [
#             blank.std.CopyFrameProps(prop_src).akarin.Expr("x.PlaneStatsAverage", vs.GRAYS) for prop_src in prop_srcs
#         ]

#         base_frame = blank.get_frame(0)

#         class IdxFrame(NamedTuple):
#             idx: int
#             frame: vs.VideoFrame

#         idx_frames = list[IdxFrame]()

#         for idx in rescales_idx:
#             fcurr = base_frame.copy()

#             fcurr[0][0, 0] = idx

#             idx_frames.append(IdxFrame(idx, fcurr))

#         def _select(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
#             return min(idx_frames, key=lambda idx_frame: f[idx_frame.idx][0][0, 0]).frame

#         select_clip = blank.std.ModifyFrame(map_prop_srcs, _select)

#         def _selector(clips: list[vs.VideoNode]) -> vs.VideoNode:
#             base = next(filter(None, clips), None)

#             if base is None:
#                 raise ValueError("Requested clip was None")

#             base = base.std.BlankClip(keep=True)
#             clips = [c or base for c in clips]

#             def _eval(n: int, f: vs.VideoFrame) -> vs.VideoNode:
#                 return clips[cast(int, f[0][0, 0])]

#             return core.std.FrameEval(base, _eval, select_clip)

#         self.upscaled = _selector([rs.upscale.with_chroma() for rs in rescales])
