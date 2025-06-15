"""(Up/De)scaling functions"""
from __future__ import annotations

from abc import abstractmethod
from functools import cached_property, wraps
from math import floor
from typing import TYPE_CHECKING, Any, Callable, ClassVar, NamedTuple, TypeAlias, Union, cast

from jetpytools import SPathLike
from vsaa import NNEDI3
from vsexprtools import ExprOp, norm_expr
from vskernels import (
    Bilinear, BorderHandling, Catrom, EwaLanczos, Hermite, Kernel, KernelLike, Placebo, Scaler, ScalerLike
)
from vskernels.types import LeftShift, TopShift
from vsmasktools import FDoG, FDoGTCanny, KirschTCanny, Morpho, XxpandMode, region_abs_mask, region_rel_mask
from vsmasktools.utils import _get_region_expr
from vsrgtools import bilateral, box_blur, gauss_blur, removegrain
from vsrgtools.rgtools import RemoveGrain
from vsscale import ArtCNN, PlaceboShader
from vstools import (
    ChromaLocation, ColorRange, ConstantFormatVideoNode, DitherType, FieldBased, FieldBasedT, GenericVSFunction,
    KwargsT, VSFunction, VSFunctionNoArgs, check_variable, core, depth, expect_bits, get_peak_value, get_u, get_v,
    get_w, get_y, initialize_clip, iterate, join, mod2, scale_value, split, vs
)

from .types import Count
from .vsjet_proxy import is_preview

__all__ = [
    "EwaLanczosChroma",
    'BaseRescale', 'Rescale', 'RescaleFrac',
    'RescaleCropBase', 'RescaleCropRel', 'RescaleCropAbs',
    'RescaleInter', 'MixedRescale'
]


class PlHermite(Placebo):
    _kernel = 'hermite'

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
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)
        assert clip.format.color_family is vs.YUV

        u, v = get_u(clip), get_v(clip)
        left_shift, top_shift = ChromaLocation.from_video(clip).get_offsets(clip)
        left_shift *= u.width / clip.width
        top_shift *= u.height / clip.height

        u = super().scale(u, clip.width, clip.height, (- top_shift, - left_shift), **kwargs)  # type: ignore
        v = super().scale(v, clip.width, clip.height, (- top_shift, - left_shift), **kwargs)  # type: ignore

        return core.std.ShufflePlanes([clip, u, v], [0, 0, 0], vs.YUV, clip)


class ArtCNNShaderBase(PlaceboShader):
    shader_file: ClassVar[SPathLike]

    def __init__(
        self,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(self.shader_file, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)


class ArtCNNShader(ArtCNNShaderBase):
    shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F16.glsl"

    class C4F16(ArtCNNShaderBase):
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F16.glsl"

    class C4F32(ArtCNNShaderBase):
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F32.glsl"

    class C4F16_Chroma(ArtCNNShaderBase):
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F16_Chroma.glsl"

    class C4F32_Chroma(ArtCNNShaderBase):
        shader_file = r"C:\Users\Varde\mpv\Shaders\ArtCNN_C4F32_Chroma.glsl"


def mpv_preview(
    clip: vs.VideoNode, w: int | None = None, h: int | None = None,
    scale: ScalerLike = ArtCNNShader.C4F16,
    dscale: ScalerLike = PlHermite(linearize=True),
    cscale: ScalerLike = EwaLanczos(antiring=0.6),
    dither_type: DitherType = DitherType.ORDERED
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

        planes[1] = cscale.scale(planes[1], planes[0].width, planes[0].height, (- top_shift, - left_shift))
        planes[2] = cscale.scale(planes[2], planes[0].width, planes[0].height, (- top_shift, - left_shift))

        preview = core.std.CopyFrameProps(join(planes), clip)

    preview = cast(ConstantFormatVideoNode, preview)

    if is_preview():
        props["PreviewCscale"] = cscale.__class__.__name__

    return dither_type.apply(
        preview,
        preview.format.replace(color_family=vs.RGB, sample_type=vs.INTEGER, bits_per_sample=8),
        ColorRange.from_video(preview),
        ColorRange.FULL
    ).std.SetFrameProps(**props)


RescaleFunc = Callable[["BaseRescale", vs.VideoNode], vs.VideoNode]


class BaseRescale:
    """
    A rewritten DescaleTarget class
    """
    clip: ConstantFormatVideoNode
    clipy: ConstantFormatVideoNode

    width: int
    height: int
    src_top: TopShift
    src_left: LeftShift
    src_width: float
    src_height: float

    kernel: Kernel
    upscaler: Scaler
    downscaler: Scaler

    border_handling: BorderHandling

    if TYPE_CHECKING:
        class VideoNodeWithChromaEmbed(vs.VideoNode):
            def __init__(*args: Any, **kwargs: Any) -> None:
                ...

            def with_chroma(self, chroma: vs.VideoNode | None = None) -> vs.VideoNode:
                ...
    else:
        class VideoNodeWithChromaEmbed:
            def __init__(self, luma: vs.VideoNode, chroma: vs.VideoNode | None) -> None:
                self.luma = luma
                self.chroma = chroma

            def __getattr__(self, __name: str) -> Any:
                return getattr(self.luma, __name)

            def __add__(self, other):
                return self.luma.__add__(other)

            def __radd__(self, other):
                return self.luma.__radd__(other)

            def __mul__(self, other: int):
                return self.luma.__mul__(other)

            def __rmul__(self, other: int):
                return self.luma.__rmul__(other)

            def __getitem__(self, index: int | slice, /):
                return self.luma.__getitem__(index)

            def __len__(self) -> int:
                return self.luma.__len__()

            def with_chroma(self, chroma: vs.VideoNode | None = None) -> vs.VideoNode:
                if not (chroma or (self.chroma and self.chroma.format.color_family == vs.YUV)):
                    return self.luma

                chroma = initialize_clip(chroma or self.chroma, -1)
                withchroma = join(self.luma, chroma)
                withchroma = core.akarin.PropExpr(
                    [withchroma, chroma],
                    lambda: dict(_ChromaLocation='y._ChromaLocation', _SARNum='y._SARNum', _SARDen='y._SARDen')
                )
                return withchroma

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: int,
        kernel: KernelLike,
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        border_handling: BorderHandling = BorderHandling.MIRROR
    ) -> None:
        """
        Initialize the rescaling process.

        :param clip:                Clip to be rescaled.
        :param height:              Height to be descaled to.
        :param kernel:              Kernel used for descaling.
        :param upscaler:            Scaler that supports doubling, defaults to NNEDI3
        :param downscaler:          Scaler used for downscaling the upscaled clip back to input res, defaults to Hermite(linear=True)
        :param width:               Width to be descaled to, defaults to None
        :param shift:               Shifts to apply during descale and upscale, defaults to (0, 0)
        :param border_handling:     Adjust the way the clip is padded internally during the scaling process.
                                    Accepted values are:
                                        0: Assume the image was resized with mirror padding.
                                        1: Assume the image was resized with zero padding.
                                        2: Assume the image was resized with extend padding, where the outermost row was extended infinitely far.
                                    Defaults to 0
        """
        assert check_variable(clip, self.__class__)
        self.clip = clip
        self.clipy = get_y(clip)  #type: ignore[assignment]

        self.height = height
        if not width:
            self.width = get_w(self.height, self.clipy)
        else:
            self.width = width
        self.src_top, self.src_left = [float(x) for x in shift][:2]
        self.src_width = float(self.width)
        self.src_height = float(self.height)

        self.kernel = Kernel.ensure_obj(kernel)
        self.upscaler = Scaler.ensure_obj(upscaler)

        self.downscaler = Scaler.ensure_obj(downscaler)

        self.border_handling = self.kernel.kwargs.pop("border_handling", border_handling)

    @cached_property
    def descale(self) -> vs.VideoNode:
        """Returns the descaled clip"""
        return self._generate_descale(self.clipy)

    @cached_property
    def rescale(self) -> vs.VideoNode:
        """Returns the descaled clip upscaled back with the specified kernel"""
        return self._generate_rescale(self.descale)

    @cached_property
    def doubled(self) -> vs.VideoNode:
        """Returns the doubled clip"""
        return self._generate_doubled(self.descale)

    @cached_property
    def upscale(self) -> VideoNodeWithChromaEmbed:
        """Returns the upscaled clip"""
        return self.VideoNodeWithChromaEmbed(self._generate_upscale(self.doubled), self.clip)

    def _trydelattr(self, attr: str) -> None:
        try:
            delattr(self, attr)
        except AttributeError:
            pass

    def __delattr__(self, __name: str) -> None:
        match __name:
            case 'descale':
                self._trydelattr('rescale')
                self._trydelattr('doubled')
            case 'doubled':
                self._trydelattr('upscale')
            case _:
                pass
        delattr(self, __name)

    def diff(self, clip: vs.VideoNode, expr: str = 'x y - abs dup 0.015 > swap 0 ?') -> vs.VideoNode:
        return norm_expr((depth(self.rescale, 32), depth(get_y(clip), 32)), expr).std.Crop(5, 5, 5, 5).std.PlaneStats()

    @staticmethod
    def _add_props(function: RescaleFunc) -> RescaleFunc:
        @wraps(function)
        def wrap(self: BaseRescale, clip: vs.VideoNode) -> vs.VideoNode:
            w, h = (f"{int(d)}" if d.is_integer() else f"{d:.2f}" for d in [self.src_width, self.src_height])
            return core.std.SetFrameProp(
                function(self, clip),
                "VdfRescale" + function.__name__.split('_')[-1].capitalize() + 'From',
                data=f'{self.kernel.__class__.__name__} - {w} x {h}'
            )
        return wrap

    # generate
    @_add_props
    def _generate_descale(self: BaseRescale, clip: vs.VideoNode) -> vs.VideoNode:
        return self.kernel.descale(clip, **self.scale_args._asdict(), border_handling=self.border_handling)

    @_add_props
    def _generate_rescale(self: BaseRescale, clip: vs.VideoNode) -> vs.VideoNode:
        return self.kernel.scale(
                clip,
                self.clip.width, self.clip.height,
                src_left=self.src_left,
                src_top=self.src_top,
                src_width=self.src_width - ((clip.width - self.width) if self.src_width.is_integer() else 0),
                src_height=self.src_height - ((clip.height - self.height) if self.src_height.is_integer() else 0),
                border_handling=self.border_handling
        )

    @_add_props
    def _generate_doubled(self: BaseRescale, clip: vs.VideoNode) -> vs.VideoNode:
        return self.upscaler.multi(clip, 2)

    @_add_props
    def _generate_upscale(self: BaseRescale, clip: vs.VideoNode) -> vs.VideoNode:
        return self.downscaler.scale(
            clip,
            **{k: v * 2 for k, v in self.scale_args._asdict().items()} | KwargsT(width=self.clip.width, height=self.clip.height)
        )

    class _ScaleArgs(NamedTuple):
        width: int
        height: int
        src_top: float
        src_left: float
        src_width: float
        src_height: float

    @property
    def scale_args(self) -> _ScaleArgs:
        """Scaling arguments"""
        return self._ScaleArgs(self.width, self.height, self.src_top, self.src_left, self.src_width, self.src_height)


class Rescale(BaseRescale):
    _line_mask: vs.VideoNode | None
    _credit_mask: vs.VideoNode | None

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: int,
        kernel: KernelLike,
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        border_handling: BorderHandling = BorderHandling.MIRROR,
    ) -> None:
        """
        Initialize the rescaling process.

        :param clip:                Clip to be rescaled.
        :param height:              Height to be descaled to.
        :param kernel:              Kernel used for descaling.
        :param upscaler:            Scaler that supports doubling, defaults to NNEDI3
        :param downscaler:          Scaler used for downscaling the upscaled clip back to input res, defaults to Hermite(linear=True)
        :param width:               Width to be descaled to, defaults to None
        :param shift:               Shifts to apply during descale and upscale, defaults to (0, 0)
        :param border_handling:     Adjust the way the clip is padded internally during the scaling process.
                                    Accepted values are:
                                        0: Assume the image was resized with mirror padding.
                                        1: Assume the image was resized with zero padding.
                                        2: Assume the image was resized with extend padding, where the outermost row was extended infinitely far.
                                    Defaults to 0
        """
        super().__init__(clip, height, kernel, upscaler, downscaler, width, shift, border_handling)

        self._line_mask = None
        self._credit_mask = None
        # self.default_line_mask()

    def _generate_upscale(self, clip: vs.VideoNode) -> vs.VideoNode:
        upscale = depth(super()._generate_upscale(clip), self.clip)
        if self._line_mask or self.border_handling:
            upscale = core.std.MaskedMerge(self.clipy, upscale, self.line_mask).std.CopyFrameProps(upscale)
        if self._credit_mask:
            upscale = core.std.MaskedMerge(upscale, self.clipy, self.credit_mask)
        return upscale

    # LINEMASK
    @property
    def line_mask(self) -> vs.VideoNode:
        if self._line_mask:
            _line_mask  = self._line_mask
        else:
            _line_mask = self.clipy.std.BlankClip(color=get_peak_value(self.clipy)).std.SetFrameProps(BlankClip=1)

        if self.border_handling:
            px = (self.kernel.kernel_radius, ) * 4
            _line_mask = norm_expr(
                _line_mask,
                _get_region_expr(_line_mask, *px, replace=str(get_peak_value(_line_mask)) + " x")
            )

        self._line_mask = _line_mask

        return self._line_mask

    @line_mask.setter
    def line_mask(self, mask: vs.VideoNode | None) -> None:
        if not mask:
            self._line_mask = None
        else:
            self._line_mask = depth(mask, self.clipy, dither_type=DitherType.NONE)

    @line_mask.deleter
    def line_mask(self) -> None:
        self._line_mask = None

    def default_line_mask(self, clip: vs.VideoNode | None = None, scaler: ScalerLike = Bilinear) -> vs.VideoNode:
        """
        Load a default Kirsch line mask in the class instance. Additionnaly, it is returned.

        :param clip:    Reference clip, defaults to luma source clip if None.
        :param scaler:  Scaled used for matching the source clip format, defaults to Bilinear
        :return:        Generated mask.
        """
        line_mask = KirschTCanny.edgemask(clip if clip else self.clipy).std.Maximum().std.Minimum()
        line_mask = Scaler.ensure_obj(scaler).scale(line_mask, self.clipy.width, self.clipy.height, format=self.clipy.format)
        self.line_mask = line_mask
        return self.line_mask

    def placebo_line_mask(
        self, clip: vs.VideoNode | None = None,
        lthrs: tuple[float, float] = (0.2, 0.2),
        multis: tuple[float, float] = (1.2, 0.3),
        scaler: ScalerLike = Bilinear,
        **kwargs: Any
    ) -> vs.VideoNode:
        """
        Load a combinaison of FDoG ridge and edge masks. Additionnaly, it is returned.

        :param clip:    Reference clip, defaults to luma source clip if None.
        :param scaler:  Scaled used for matching the source clip format, defaults to Bilinear
        :return:        Generated mask.
        """
        clip = clip if clip else self.clip
        clipy = get_y(clip) if clip else self.clipy
        scaler = Scaler.ensure_obj(scaler)

        hthrs = kwargs.setdefault("hthrs", (None, None))
        clamps = kwargs.setdefault("clamps", (False, False))
        planes = kwargs.setdefault("planes", (None, None))

        edgemask = FDoGTCanny.edgemask(clip, lthrs[0], hthrs[0], multis[0], clamps[0], planes[0]).std.Maximum().std.Minimum()
        edgemask = ColorRange.FULL.apply(edgemask)
        edgemask = ExprOp.ADD.combine(
            scaler.scale(c, edgemask.width, edgemask.height, format=vs.GRAYS) for c in split(edgemask)
        )

        ridgemask = FDoG.ridgemask(depth(clipy, 32), lthrs[1], hthrs[1], multis[1], clamps[1], planes[1]).std.Maximum().std.Minimum()

        mask = core.akarin.Expr([edgemask, ridgemask], 'x y 0 max + 0 1 clamp')
        mask = scaler.scale(mask, self.clipy.width, self.clipy.height, format=self.clipy.format)

        self.line_mask = box_blur(mask)
        return self.line_mask

    def vodes_line_mask(
        self,
        clip: vs.VideoNode | None = None, scaler: ScalerLike = Bilinear,
        lthr: float | None = None, hthr: float | None = None
    ) -> vs.VideoNode:
        """
        Load DescaleTarget default mask

        :param clip:    Reference clip, defaults to luma source clip if None.
        :param scaler:  Scaled used for matching the source clip format, defaults to Bilinear
        :param lthr:    Low threshold
        :param hthr:    High threshold
        :return:        Generated mask.
        """
        scaler = Scaler.ensure_obj(scaler)
        mask = KirschTCanny.edgemask(
            get_y(clip) if clip else self.clipy,
            scale_value(80, 8, 32) if not lthr else lthr,
            scale_value(150, 8, 32) if not hthr else hthr
        )
        self.line_mask = scaler.scale(mask, self.clipy.width, self.clipy.height, format=self.clipy.format)
        return self.line_mask

    # CREDITMASK
    @property
    def credit_mask(self) -> vs.VideoNode:
        if self._credit_mask:
            return self._credit_mask
        self.credit_mask = self.clipy.std.BlankClip().std.SetFrameProps(BlankClip=1)
        return self.credit_mask

    @credit_mask.setter
    def credit_mask(self, mask: vs.VideoNode | None) -> None:
        if not mask:
            self._credit_mask = None
        else:
            self._credit_mask = depth(mask, self.clipy, dither_type=DitherType.NONE)

    @credit_mask.deleter
    def credit_mask(self) -> None:
        self._credit_mask = None

    def default_credit_mask(
        self, rescale: vs.VideoNode | None = None, src: vs.VideoNode | None = None,
        thr: float = 0.216, blur: float | KwargsT | None = None,
        prefilter: int | KwargsT | bool | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = False,
        postfilter: int | tuple[Count, RemoveGrain.Mode] | list[tuple[Count, RemoveGrain.Mode]] | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = 2,
        ampl_expr: str | None = None,
        expand: int = 2
    ) -> vs.VideoNode:
        """
        Load a credit mask based on vsmasktools.credit_mask and vsmasktools.diff_rescale

        :param rescale:     Rescaled clip, defaults to rescaled instance clip
        :param src:         Source clip, defaults to source instance clip
        :param thr:         Threshold of the amplification expr, defaults to 0.216
        :param blur:        Sigma of the gaussian blur applied before prefilter, defaults to None
        :param prefilter:   Filter applied before extracting the difference between rescale and src
                            int -> equivalent of number of taps used in the bilateral call applied to clips
                            True -> 5 taps
                            KwargsT -> Arguments passed to the bilateral function
        :param postfilter:  Filter applied to the difference clip. Default is RemoveGrainMode.MINMAX_AROUND2 applied twice.
        :param ampl_expr:   Amplification expression.
        :param expand:      Additional expand radius applied to the mask, defaults to 2
        :return:            Generated mask
        """
        if not src:
            src = self.clip
        if not rescale:
            rescale = self.rescale

        src, rescale = get_y(src), get_y(rescale)

        if blur:
            if isinstance(blur, dict):
                src, rescale = gauss_blur(src, **blur), gauss_blur(rescale, **blur)
            else:
                src, rescale = gauss_blur(src, blur), gauss_blur(rescale, blur)

        if prefilter:
            if callable(prefilter):
                src, rescale = prefilter(src), prefilter(rescale)
            else:
                if isinstance(prefilter, int):
                    sigma = 5 if prefilter is True else prefilter
                    kwargs = KwargsT(sigmaS=((sigma ** 2 - 1) / 12) ** 0.5, sigmaR=sigma / 10)
                else:
                    kwargs = prefilter

                src, rescale = bilateral(src, **kwargs), bilateral(rescale, **kwargs)

        pre, bits = expect_bits(src, 32)
        rescale = depth(rescale, 32)

        diff = ExprOp.mae(src)(pre, rescale)

        if postfilter:
            if isinstance(postfilter, int):
                mask = iterate(diff, removegrain, postfilter, RemoveGrain.Mode.MINMAX_AROUND2)
            elif isinstance(postfilter, tuple):
                mask = iterate(diff, removegrain, postfilter[0], postfilter[1])
            elif isinstance(postfilter, list):
                mask = diff
                for count, rgmode in postfilter:
                    mask = iterate(mask, removegrain, count, rgmode)
            else:
                mask = postfilter(diff)

        mask = mask.std.Expr(ampl_expr or f'x 2 4 pow * {thr} < 0 1 ?')

        mask = Morpho.expand(mask, 2 + expand, mode=XxpandMode.ELLIPSE).std.Deflate()

        mask = ColorRange.FULL.apply(mask)

        self.credit_mask = depth(mask, bits, dither_type=DitherType.NONE)
        return self.credit_mask

    def vodes_credit_mask(self, rescale: vs.VideoNode | None = None, src: vs.VideoNode | None = None, thr: float = 0.04) -> vs.VideoNode:
        """
        Load DescaleTarget default mask

        :param rescale:     Rescaled clip, defaults to rescaled instance clip
        :param src:         Source clip, defaults to source instance clip
        :param thr:         Threshold of difference, defaults to 0.01
        :return:            Generated mask.
        """
        if not src:
            src = self.clip
        if not rescale:
            rescale = self.rescale
        credit_mask = core.akarin.Expr([depth(src, 32), depth(rescale, 32)], f'x y - abs {thr} < 0 1 ?')
        credit_mask = depth(credit_mask, 16, range_in=ColorRange.FULL, range_out=ColorRange.FULL, dither_type=DitherType.NONE)
        credit_mask = credit_mask.rgvs.RemoveGrain(6).std.Maximum().std.Maximum().std.Inflate().std.Inflate()
        self.credit_mask = credit_mask
        return self.credit_mask


class RescaleFrac(Rescale):
    base_width: int
    base_height: int

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: float,
        kernel: KernelLike,
        base_height: int,
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: float | None = None,
        base_width: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        border_handling: BorderHandling = BorderHandling.MIRROR,
    ) -> None:
        """
        Initialize the rescaling process.

        :param clip:                Clip to be rescaled.
        :param height:              Float Height to be descaled to.
        :param kernel:              Kernel used for descaling.
        :param base_height:         Integer height at which the clip will be contained
        :param upscaler:            Scaler that supports doubling, defaults to NNEDI3
        :param downscaler:          Scaler used for downscaling the upscaled clip back to input res, defaults to Hermite(linear=True)
        :param width:               Float width to be descaled to, defaults to None
        :param base_width:          Integer width at which the clip will be contained, defaults to None
        :param shift:               Shifts to apply during descale and upscale, defaults to (0, 0)
        :param border_handling:     Adjust the way the clip is padded internally during the scaling process.
                                    Accepted values are:
                                        0: Assume the image was resized with mirror padding.
                                        1: Assume the image was resized with zero padding.
                                        2: Assume the image was resized with extend padding, where the outermost row was extended infinitely far.
                                    Defaults to 0
        """
        self.base_height = base_height
        if not base_width:
            self.base_width = get_w(self.base_height, clip)
        else:
            self.base_width = base_width

        if not width:
            width = height * clip.width / clip.height

        cropped_width = self.base_width - 2 * floor((self.base_width - width) / 2)
        cropped_height = self.base_height - 2 * floor((self.base_height - height) / 2)
        self.width = cropped_width
        self.height = cropped_height
        self.src_top = (cropped_height - height) / 2 + shift[0]
        self.src_left = (cropped_width - width) / 2 + shift[1]

        super().__init__(
            clip, self.height, kernel, upscaler, downscaler, self.width,
            (self.src_top, self.src_left), border_handling
        )

        self.src_width = width
        self.src_height = height

    def default_credit_mask(
        self, rescale: vs.VideoNode | None = None, src: vs.VideoNode | None = None,
        thr: float = 0.216, blur: float | KwargsT | None = None,
        prefilter: int | KwargsT | bool | VSFunction = False,
        postfilter: int | tuple[Count, RemoveGrain.Mode] | list[tuple[Count, RemoveGrain.Mode]] | VSFunction = 2,
        ampl_expr: str | None = None,
        expand: int = 2,
        use_base_height: bool = False
    ) -> vs.VideoNode:
        """
        Load a credit mask based on vsmasktools.credit_mask and vsmasktools.diff_rescale

        :param rescale:         Rescaled clip, defaults to rescaled instance clip
        :param src:             Source clip, defaults to source instance clip
        :param thr:             Threshold of the amplification expr, defaults to 0.216
        :param blur:            Sigma of the gaussian blur applied before prefilter, defaults to None
        :param prefilter:       Filter applied before extracting the difference between rescale and src
                                int -> equivalent of number taps used in the bilateral call applied to clips
                                True -> 5 taps
                                KwargsT -> Arguments passed to the bilateral function
        :param postfilter:      Filter applied to the difference clip. Default is RemoveGrainMode.MINMAX_AROUND2 applied twice.
        :param ampl_expr:       Amplification expression.
        :param expand:          Additional expand radius applied to the mask, defaults to 2
        :param use_base_height: Will use a rescaled clip based on base_height instead of height
        :return:                Generated mask
        """
        if use_base_height:
            rescale = Rescale(
                self.clipy, self.base_height, self.kernel,
                width=self.base_width, border_handling=self.border_handling
            ).rescale

        return super().default_credit_mask(rescale, src, thr, blur, prefilter, postfilter, ampl_expr, expand)


LeftCrop: TypeAlias = int
RightCrop: TypeAlias = int
TopCrop: TypeAlias = int
BottomCrop: TypeAlias = int
WidthCrop: TypeAlias = int
HeightCrop: TypeAlias = int


class RescaleCropBase(RescaleFrac):
    pre: vs.VideoNode
    crop: tuple[int, ...]

    crop_function: GenericVSFunction[vs.VideoNode]

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: float,
        kernel: KernelLike,
        crop: tuple[int, ...] | None = None,
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: float | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        border_handling: BorderHandling = BorderHandling.MIRROR,
    ) -> None:
        self.pre = clip
        self.crop = crop if crop else (0, 0, 0, 0)

        clip_cropped = self.crop_function(clip, *self.crop)

        if not width:
            if isinstance(height, int):
                width = get_w(height, get_y(clip))
            else:
                width = height * clip.width / clip.height

        height = clip_cropped.height / (self.pre.height / height)
        width = clip_cropped.width / (self.pre.width / width)

        base_height = mod2(height)
        base_width = mod2(width)

        super().__init__(clip_cropped, height, kernel, base_height, upscaler, downscaler, width, base_width, shift, border_handling)

    def _generate_upscale(self, clip: vs.VideoNode) -> vs.VideoNode:
        white = get_y(self.pre).std.BlankClip(color=get_peak_value(self.pre))

        upscale = super()._generate_upscale(clip)

        return core.std.MaskedMerge(
            upscale.std.AddBorders(*self._abs_to_rel()),
            get_y(self.pre),
            self.region_function(white, *self.crop).std.Invert()
        )

    @cached_property
    def upscale(self) -> BaseRescale.VideoNodeWithChromaEmbed:
        """Returns the upscaled clip"""
        return self.VideoNodeWithChromaEmbed(self._generate_upscale(self.doubled), self.pre)

    @abstractmethod
    def _abs_to_rel(self) -> tuple[int, ...]:
        ...

    @abstractmethod
    def region_function(self, *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...


class RescaleCropRel(RescaleCropBase):
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop]

    crop_function = core.lazy.std.CropRel

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: float,
        kernel: KernelLike,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop],
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: float | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        border_handling: BorderHandling = BorderHandling.MIRROR,
    ) -> None:
        super().__init__(clip, height, kernel, crop, upscaler, downscaler, width, shift, border_handling)

    def _abs_to_rel(self) -> tuple[int, ...]:
        return self.crop

    def region_function(self, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return region_rel_mask(*args, **kwargs)


class RescaleCropAbs(RescaleCropBase):
    crop: tuple[WidthCrop, HeightCrop, LeftCrop, TopCrop]

    crop_function = core.lazy.std.CropAbs

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: float,
        kernel: KernelLike,
        crop: Union[
            tuple[WidthCrop, HeightCrop],
            tuple[WidthCrop, HeightCrop, LeftCrop, TopCrop],
        ],
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: float | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        border_handling: BorderHandling = BorderHandling.MIRROR,
    ) -> None:

        ncrop = crop + (0, ) * (4 - len(crop))

        super().__init__(clip, height, kernel, ncrop, upscaler, downscaler, width, shift, border_handling)

    def _abs_to_rel(self) -> tuple[int, ...]:
        return (
            self.crop[2],
            self.pre.width - self.crop[0] - self.crop[2],
            self.crop[3],
            self.pre.height - self.crop[1] - self.crop[3]
        )

    def region_function(self, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return region_abs_mask(*args, **kwargs)


RescaleInterFunc = Callable[["RescaleInter", vs.VideoNode], vs.VideoNode]


class RescaleInter(Rescale):
    field_based: FieldBased

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: int,
        kernel: KernelLike,
        upscaler: ScalerLike = NNEDI3,
        downscaler: ScalerLike = Hermite(linear=True),
        width: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        field_based: FieldBasedT | None = None,
        border_handling: BorderHandling = BorderHandling.MIRROR,
    ) -> None:
        self.field_based = FieldBased.from_param(field_based) or FieldBased.from_video(clip)
        super().__init__(clip, height, kernel, upscaler, downscaler, width, shift, border_handling)

    @staticmethod
    def _apply_field_based(function: RescaleInterFunc) -> RescaleInterFunc:
        @wraps(function)
        def wrap(self: RescaleInter, clip: vs.VideoNode) -> vs.VideoNode:
            clip = self.field_based.apply(clip)
            clip = function(self, clip)
            return FieldBased.PROGRESSIVE.apply(clip)
        return wrap

    @_apply_field_based
    def _generate_descale(self: RescaleInter, clip: vs.VideoNode) -> vs.VideoNode:
        return super()._generate_descale(clip)

    @_apply_field_based
    def _generate_rescale(self: RescaleInter, clip: vs.VideoNode) -> vs.VideoNode:
        return super()._generate_rescale(clip)

    @staticmethod
    def crossconv_shift_calc_irregular(clip: vs.VideoNode, native_height: int) -> float:
        return 0.25 / (clip.height / native_height)


# class RescaleFracInter(RescaleInter, RescaleFrac):
#     ...


class MixedRescale:
    upscaled: vs.VideoNode

    def __init__(self, src: vs.VideoNode, *rescales: Rescale) -> None:
        prop_srcs = [rs.diff(src) for rs in rescales]
        rescales_idx = tuple(range(len(rescales)))

        blank = core.std.BlankClip(None, 1, 1, vs.GRAY8, src.num_frames, keep=True)

        map_prop_srcs = [blank.std.CopyFrameProps(prop_src).akarin.Expr("x.PlaneStatsAverage", vs.GRAYS) for prop_src in prop_srcs]

        base_frame = blank.get_frame(0)

        class IdxFrame(NamedTuple):
            idx: int
            frame: vs.VideoFrame

        idx_frames = list[IdxFrame]()

        for idx in rescales_idx:
            fcurr = base_frame.copy()

            fcurr[0][0, 0] = idx

            idx_frames.append(IdxFrame(idx, fcurr))

        def _select(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
            return min(idx_frames, key=lambda idx_frame: f[idx_frame.idx][0][0, 0]).frame

        select_clip = blank.std.ModifyFrame(map_prop_srcs, _select)

        def _selector(clips: list[vs.VideoNode]) -> vs.VideoNode:
            base = next(filter(None, clips), None)

            if base is None:
                raise ValueError("Requested clip was None")

            base = base.std.BlankClip(keep=True)
            clips = [c or base for c in clips]

            def _eval(n: int, f: vs.VideoFrame) -> vs.VideoNode:
                return clips[cast(int, f[0][0, 0])]

            return core.std.FrameEval(base, _eval, select_clip)

        self.upscaled = _selector([rs.upscale.with_chroma() for rs in rescales])
