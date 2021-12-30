"""Wrappers and masks for denoising, debanding, rescaling etc."""

__all__ = [
    'detail_mask',
    'Difference', 'diff_creditless_mask', 'diff_rescale_mask', 'luma_mask', 'luma_credit_mask', 'region_mask'
]

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lvsfunc
import vapoursynth as vs
from vsmask.edge import EdgeDetect as EdgeDetectVsM
from vsmask.edge import ExLaplacian4 as ExLaplacian4VsM
from vsmask.edge import FDOGTCanny as FDOGTCannyVsM
from vsmask.edge import Kirsch as KirschVsM
from vsmask.edge import MinMax as MinMaxVsM
from vsmask.util import XxpandMode
from vsmask.util import expand as expand_func
from vsutil import depth, get_depth, get_w, get_y, insert_clip, iterate, join, scale_value, split

from .types import Zimg, format_not_none
from .util import get_sample_type, mae_expr, max_expr, pick_px_op

core = vs.core


def detail_mask(clip: vs.VideoNode, brz_mm: float, brz_ed: float,
                minmax: MinMaxVsM = MinMaxVsM(rady=3, radc=2),
                edgedetect: EdgeDetectVsM = KirschVsM()) -> vs.VideoNode:
    if clip.format is None:
        raise ValueError("detail_mask: 'Variable-format clips not supported'")

    range_mask = minmax.get_mask(clip).std.Binarize(brz_mm)
    edges = edgedetect.get_mask(clip).std.Binarize(brz_ed)

    mask = core.std.Expr((range_mask, edges), 'x y max')

    removegrain = core.rgvs.RemoveGrain if clip.format.bits_per_sample < 32 else core.rgsf.RemoveGrain

    mask = removegrain(mask, 22)
    mask = removegrain(mask, 11)

    return mask


class Difference():
    """Collection of function based on differences between prediction and observation"""

    def rescale(self, clip: vs.VideoNode, height: int = 720,
                kernel: lvsfunc.kernels.Kernel = lvsfunc.kernels.Catrom(),
                thr: Union[int, float] = 55, expand: int = 2) -> vs.VideoNode:
        """Makes a mask based on rescaled difference.
           Modified version of Atomchtools.

        Args:
            clip (vs.VideoNode):
                Source clip. Can be Gray, YUV or RGB.
                Keep in mind that descale plugin will descale all planes
                after conversion to GRAYS, YUV444PS and RGBS respectively.

            height (int, optional):
                Height to descale to. Defaults to 720.

            kernel (lvsfunc.kernels.Kernel, optional):
                Kernel used to descale. Defaults to lvsfunc.kernels.Bicubic(b=0, c=0.5).

            thr (Union[int, float], optional):
                Binarization threshold. Defaults to 55.

            expand (int, optional):
                Growing/shrinking shape. 0 is allowed. Defaults to 2.

        Returns:
            vs.VideoNode: Rescaled mask.
        """
        clip = format_not_none(clip)

        bits = get_depth(clip)
        gray_only = clip.format.num_planes == 1
        thr = scale_value(thr, bits, 32, scale_offsets=True)

        pre = core.resize.Bicubic(
            clip, format=clip.format.replace(
                bits_per_sample=32, sample_type=vs.FLOAT, subsampling_w=0, subsampling_h=0
            ).id
        )
        descale = kernel.descale(pre, get_w(height), height)
        rescale = kernel.scale(descale, clip.width, clip.height)

        diff = core.std.Expr(split(pre) + split(rescale), mae_expr(gray_only))

        mask = iterate(diff, lambda x: core.rgsf.RemoveGrain(x, 2), 2)
        mask = core.std.Expr(mask, f'x 2 4 pow * {thr} < 0 1 ?')

        mask = expand_func(mask, 2 + expand, mode=XxpandMode.ELLIPSE)
        mask = mask.std.Deflate()

        return mask.resize.Point(
            format=clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
            dither_type='none'
        )


    def creditless(self, src_clip: vs.VideoNode, credit_clip: vs.VideoNode, nc_clip: vs.VideoNode,
                   start_frame: int, thr: int, expand: int = 2, *,
                   prefilter: bool = False, bilateral_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
        """Makes a mask based on difference from 2 clips.

        Args:
            src_clip (vs.VideoNode):
                Source clip. Can be Gray, YUV or RGB.

            credit_clip (vs.VideoNode): Credit clip.
                It will be resampled according to the src_clip.

            nc_clip (vs.VideoNode): Creditless clip.
                It will be resampled according to the src_clip.

            start_frame (int): Start frame.

            thr (int): Binarize threshold.
                25 is a good starting value in 8 bit.

            expand (int, optional):
                Growing/shrinking shape. 0 is allowed. Defaults to 2.

            prefilter (bool, optional):
                Blurs the credit_clip and nc_clip with Bilateral to avoid false posivive
                such as noise and compression artifacts.
                Defaults to False.

            bilateral_args (Dict[str, Any], optional):
                Additionnal and overrided Bilateral parameters if prefilter=True. Defaults to None.

        Returns:
            vs.VideoNode: Credit mask clip.

        Example:
            import vardefunc as vdf

            opstart, opend = 792, 2948

            opmask = diff_creditless_mask(clip, clip[opstart:opend+1], ncop[:opend+1-opstart], opstart, thr=25, prefilter=True)
        """
        src_clip = format_not_none(src_clip)

        gray_only = src_clip.format.num_planes == 1
        clips = [credit_clip, nc_clip]

        if prefilter:
            bilargs: Dict[str, Any] = dict(sigmaS=((5 ** 2 - 1) / 12) ** 0.5, sigmaR=0.5)
            if bilateral_args:
                bilargs |= bilateral_args
            clips = [c.bilateral.Bilateral(**bilargs) for c in clips]


        clips = [
            c.resize.Bicubic(
                format=src_clip.format.replace(
                    bits_per_sample=get_depth(src_clip),
                    subsampling_w=0, subsampling_h=0).id
            ) for c in clips]

        diff = core.std.Expr(
            sum(map(split, clips), []),
            mae_expr(gray_only),
            format=src_clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id
        )

        mask = ExLaplacian4VsM().get_mask(diff).std.Binarize(thr)
        mask = expand_func(mask, 2 + expand, mode=XxpandMode.ELLIPSE)

        blank = core.std.BlankClip(
            src_clip, format=src_clip.format.replace(
                color_family=vs.GRAY, subsampling_w=0, subsampling_h=0
            ).id
        )
        mask = insert_clip(blank, mask, start_frame)

        return mask

    def creditless_oped(self, ep: vs.VideoNode, ncop: vs.VideoNode, nced: vs.VideoNode,
                        opstart: Optional[int] = None, opend: Optional[int] = None,
                        edstart: Optional[int] = None, edend: Optional[int] = None,
                        **creditless_args: Any) -> vs.VideoNode:
        """Wrapper using Difference().creditless(...). for opening and ending.

        Args:
            ep (vs.VideoNode): Full episode clip.
            ncop (vs.VideoNode): NCOP clip.
            nced (vs.VideoNode): NCED clip.
            opstart (int): Opening start frame.
            opend (int): Opening end frame.
            edstart (int): Ending start frame.
            edend (int): Ending end frame.

        Returns:
            vs.VideoNode: Mask.
        """
        args: Dict[str, Any] = dict(thr=25, expand=4, prefilter=False)
        args |= creditless_args

        if None not in {opstart, opend, edstart, edend}:
            mask = core.std.Expr((self.creditless(ep, ep[opstart:opend+1], ncop[:opend-opstart+1], opstart, **args),  # type: ignore
                                  self.creditless(ep, ep[edstart:edend+1], nced[:edend-edstart+1], edstart, **args)),  # type: ignore
                                 'x y +')
        elif None in {edstart, edend} and None not in {opstart, opend}:
            mask = self.creditless(ep, ep[opstart:opend+1], ncop[:opend-opstart+1], opstart, **args)  # type: ignore
        elif None in {opstart, opend} and None not in {edstart, edend}:
            mask = self.creditless(ep, ep[edstart:edend+1], nced[:edend-edstart+1], edstart, **args)  # type: ignore
        else:
            raise ValueError('creditless_oped: wtf are you doing')

        return mask


def diff_creditless_mask(src_clip: vs.VideoNode, credit_clip: vs.VideoNode, nc_clip: vs.VideoNode,
                         start_frame: int, thr: int, expand: int = 2, *,
                         prefilter: bool = False, bilateral_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    """Legacy function of Difference().creditless"""
    return Difference().creditless(src_clip, credit_clip, nc_clip, start_frame, thr, expand,
                                   prefilter=prefilter, bilateral_args=bilateral_args)


def diff_rescale_mask(clip: vs.VideoNode, height: int = 720,
                      kernel: lvsfunc.kernels.Kernel = lvsfunc.kernels.Catrom(),
                      thr: Union[int, float] = 55, expand: int = 2) -> vs.VideoNode:
    """Legacy function of Difference().rescale"""
    return Difference().rescale(clip, height, kernel, thr, expand)



def luma_mask(clip: vs.VideoNode, thr_lo: float, thr_hi: float, invert: bool = True) -> vs.VideoNode:
    """Mask each pixel according to its luma value.
       From debandshit.

    Args:
        clip (vs.VideoNode):
            Source clip.

        thr_lo (float):
            All pixels below this threshold will be binary

        thr_hi (float):
            All pixels above this threshold will be binary

        All pixels in-between will be scaled from black to white

        invert (bool, optional):
            When true, masks dark areas (pixels below lo will be white, and vice versa).
            Defaults to True.

    Returns:
        vs.VideoNode: Luma mask.
    """
    bits = get_depth(clip)
    is_float = get_sample_type(clip) == vs.FLOAT
    peak = 1.0 if is_float else (1 << bits) - 1

    mask = pick_px_op(
        is_float,
        (f'x {thr_lo} < 0 x {thr_hi} > {peak} x {thr_lo} - {thr_lo} {thr_hi} - / {peak} * ? ?',
         lambda x: round(0 if x < thr_lo else peak if x > thr_hi else (x - thr_lo) / (thr_hi - thr_lo) * peak)))(get_y(clip))

    return mask.std.Invert() if invert else mask


def luma_credit_mask(clip: vs.VideoNode, thr: int = 230,
                     edgemask: EdgeDetectVsM = FDOGTCannyVsM(), draft: bool = False) -> vs.VideoNode:
    """Makes a mask based on luma value and edges.

    Args:
        clip (vs.VideoNode):
            Source clip.

        thr (int, optional):
            Luma value assuming 8 bit input. Defaults to 230.

        edgemask (EdgeDetect, optional):
            Edge mask used with thr. Defaults to FDOG().

        draft (bool, optional):
            Allow to output the mask without growing. Defaults to False.

    Returns:
        vs.VideoNode: Credit mask.
    """
    clip = get_y(clip)

    edge_mask = edgemask.get_mask(clip)

    credit_mask = core.std.Expr([edge_mask, clip], f'y {thr} > y 0 ? x min')

    if not draft:
        credit_mask = iterate(credit_mask, core.std.Maximum, 4)
        credit_mask = iterate(credit_mask, core.std.Inflate, 2)

    return credit_mask


# Depreciated stuff


class EdgeDetect(ABC):
    """Abstract edge detection interface."""

    def __init__(self) -> None:
        warnings.warn(
            'vardefunc.mask.EdgeDetect and all its subclasses are deprecated in favor of vsmask.\n'
            'Please install it at https://github.com/Irrational-Encoding-Wizardry/vsmask',
            DeprecationWarning
        )

    def get_mask(self, clip: vs.VideoNode, lthr: float = 0.0, hthr: Optional[float] = None, multi: float = 1.0) -> vs.VideoNode:
        """Makes edge mask based on convolution kernel.
           The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        Args:
            clip (vs.VideoNode):
                Source clip.

            lthr (float, optional):
                Low threshold. Anything below lthr will be set to 0. Defaults to 0.

            hthr (Optional[float], optional):
                High threshold. Anything above hthr will be set to the range max. Defaults to None.

            multi (float, optional):
                Multiply all pixels by this before thresholding. Defaults to 1.0.

        Returns:
            vs.VideoNode: Mask clip.
        """
        clip = format_not_none(clip)

        bits = get_depth(clip)
        is_float = get_sample_type(clip) == vs.FLOAT
        peak = 1.0 if is_float else (1 << bits) - 1
        hthr = peak if hthr is None else hthr


        clip_p = self._preprocess(clip)
        mask = self._compute_mask(clip_p)

        mask = depth(mask, bits, range=Zimg.PixelRange.FULL, range_in=Zimg.PixelRange.FULL)


        if multi != 1:
            mask = pick_px_op(
                is_float, (f'x {multi} *', lambda x: round(max(min(x * multi, peak), 0)))
            )(mask)


        if lthr > 0 or hthr < peak:
            mask = pick_px_op(
                is_float, (f'x {hthr} > {peak} x {lthr} <= 0 x ? ?',
                           lambda x: peak if x > hthr else 0 if x <= lthr else x)
            )(mask)


        return mask

    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        masks = [clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
                 for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())]

        expr = self._get_expr()
        mask = core.std.Expr(masks, expr) if expr else masks[0]

        return mask

    def _get_divisors(self) -> List[float]:
        return [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> List[str]:
        return ['s'] * len(self._get_matrices())

    @staticmethod
    def _get_expr() -> Optional[str]:
        return None

    @staticmethod
    def _preprocess(clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    @staticmethod
    @abstractmethod
    def _get_matrices() -> List[List[float]]:
        pass


class Laplacian1(EdgeDetect):
    """Pierre-Simon de Laplace operator 1st implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, -1, 0, -1, 4, -1, 0, -1, 0]]


class Laplacian2(EdgeDetect):
    """Pierre-Simon de Laplace operator 2nd implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, -2, 1, -2, 4, -2, 1, -2, 1]]


class Laplacian3(EdgeDetect):
    """Pierre-Simon de Laplace operator 3rd implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[2, -1, 2, -1, -4, -1, 2, -1, 2]]


class Laplacian4(EdgeDetect):
    """Pierre-Simon de Laplace operator 4th implementation. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-1, -1, -1, -1, 8, -1, -1, -1, -1]]


class ExLaplacian1(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 1st implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, 8, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]]


class ExLaplacian2(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 2nd implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 1, -1, 1, 0, 1, 1, -4, 1, 1, -1, -4, 8, -4, -1, 1, 1, -4, 1, 1, 0, 1, -1, 1, 0]]


class ExLaplacian3(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 3rd implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-1, 1, -1, 1, -1, 1, 2, -4, 2, 1, -1, -4, 8, -4, -1, 1, 2, -4, 2, 1, -1, 1, -1, 1, -1]]


class ExLaplacian4(EdgeDetect):
    """Extended Pierre-Simon de Laplace operator 4th implementation. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


class Kayyali(EdgeDetect):
    """Kayyali operator. 3x3 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[6, 0, -6, 0, 0, 0, -6, 0, 6]]


class LoG(EdgeDetect):
    """Laplacian of Gaussian. 5x5 matrix."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0]]


class Roberts(EdgeDetect):
    """Lawrence Roberts operator. 2x2 matrices computed in 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[0, 0, 0, 0, 1, 0, 0, 0, -1],
                [0, 1, 0, -1, 0, 0, 0, 0, 0]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Prewitt(EdgeDetect):
    """Judith M. S. Prewitt operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 0, -1, 1, 0, -1, 1, 0, -1],
                [1, 1, 1, 0, 0, 0, -1, -1, -1]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class PrewittStd(EdgeDetect):
    """Judith M. S. Prewitt Vapoursynth plugin operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _compute_mask(clip: vs.VideoNode) -> vs.VideoNode:
        return core.std.Prewitt(clip)


class ExPrewitt(EdgeDetect):
    """Extended Judith M. S. Prewitt operator. 5x5 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
                [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Sobel(EdgeDetect):
    """Sobel–Feldman operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 0, -1, 2, 0, -2, 1, 0, -1],
                [1, 2, 1, 0, 0, 0, -1, -2, -1]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class SobelStd(EdgeDetect):
    """Sobel–Feldman Vapoursynth plugin operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _compute_mask(clip: vs.VideoNode) -> vs.VideoNode:
        return core.std.Sobel(clip)


class ExSobel(EdgeDetect):
    """Extended Sobel–Feldman operator. 5x5 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
                [2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Scharr(EdgeDetect):
    """H. Scharr optimized operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-3, 0, 3, -10, 0, 10, -3, 0, 3],
                [-3, -10, -3, 0, 0, 0, 3, 10, 3]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class FDOG(EdgeDetect):
    """Flow-based Difference Of Gaussian operator. 5x5 matrices from G41Fun."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1],
                [1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -2, -1, -1, -2, -3, -2, -1]]

    @staticmethod
    def _get_divisors() -> List[Union[int, float]]:
        return [2, 2]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class Kroon(EdgeDetect):
    """Dirk-Jan Kroon operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-17, 0, 17, -61, 0, 61, -17, 0, 17],
                [-17, -61, -17, 0, 0, 0, 17, 61, 17]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class FreyChen(EdgeDetect):
    """Chen Frei operator. 3x3 matrices properly implemented."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        sqrt2 = math.sqrt(2)
        return [[1, sqrt2, 1, 0, 0, 0, -1, -sqrt2, -1],
                [1, 0, -1, sqrt2, 0, -sqrt2, 1, 0, -1],
                [0, -1, sqrt2, 1, 0, -1, -sqrt2, 1, 0],
                [sqrt2, -1, 0, -1, 0, 1, 0, 1, -sqrt2],
                [0, 1, 0, -1, 0, -1, 0, 1, 0],
                [-1, 0, 1, 0, 0, 0, 1, 0, -1],
                [1, -2, 1, -2, 4, -2, 1, -2, 1],
                [-2, 1, -2, 1, 4, 1, -2, 1, -2],
                [1, 1, 1, 1, 1, 1, 1, 1, 1]]

    @staticmethod
    def _get_divisors() -> List[Union[int, float]]:
        sqrt2 = math.sqrt(2)
        return [2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2, 2, 6, 6, 3]

    @staticmethod
    def _get_expr() -> Optional[str]:
        M = 'x x * y y * + z z * + a a * +'
        S = f'b b * c c * + d d * + e e * + f f * + {M} +'
        return f'{M} {S} / sqrt'

    @staticmethod
    def _preprocess(clip: vs.VideoNode) -> vs.VideoNode:
        return depth(clip, 32)


class FreyChenG41(EdgeDetect):
    """"Chen Frei" operator. 3x3 matrices from G41Fun."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[-7, 0, 7, -10, 0, 10, -7, 0, 7],
                [-7, -10, -7, 0, 0, 0, 7, 10, 7]]

    @staticmethod
    def _get_divisors() -> List[Union[int, float]]:
        return [7, 7]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class TEdge(EdgeDetect):
    """(TEdgeMasktype=2) Avisynth plugin. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[12, -74, 0, 74, -12],
                [-12, 74, 0, -74, 12]]

    @staticmethod
    def _get_divisors() -> List[Union[int, float]]:
        return [62, 62]

    @staticmethod
    def _get_mode_types() -> List[str]:
        return ['h', 'v']

    @staticmethod
    def _get_expr() -> Optional[str]:
        return 'x x * y y * + sqrt'


class TEdgeTedgemask(EdgeDetect):
    """(tedgemask.TEdgeMask(threshold=0.0, type=2)) Vapoursynth plugin. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _compute_mask(clip: vs.VideoNode) -> vs.VideoNode:
        return core.tedgemask.TEdgeMask(clip, threshold=0, type=2)


class Robinson3(EdgeDetect):
    """Robinson compass operator level 3. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 1, 1, 0, 0, 0, -1, -1, -1],
                [1, 1, 0, 1, 0, -1, 0, -1, -1],
                [1, 0, -1, 1, 0, -1, 1, 0, -1],
                [0, -1, -1, 1, 0, -1, 1, 1, 0]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(4)


class Robinson5(EdgeDetect):
    """Robinson compass operator level 5. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[1, 2, 1, 0, 0, 0, -1, -2, -1],
                [2, 1, 0, 1, 0, -1, 0, -1, -2],
                [1, 0, -1, 2, 0, -2, 1, 0, -1],
                [0, -1, -2, 1, 0, -1, 2, 1, 0]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(4)


class Kirsch(EdgeDetect):
    """Russell Kirsch compass operator. 3x3 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[5, 5, 5, -3, 0, -3, -3, -3, -3],
                [5, 5, -3, 5, 0, -3, -3, -3, -3],
                [5, -3, -3, 5, 0, -3, 5, -3, -3],
                [-3, -3, -3, 5, 0, -3, 5, 5, -3],
                [-3, -3, -3, -3, 0, -3, 5, 5, 5],
                [-3, -3, -3, -3, 0, 5, -3, 5, 5],
                [-3, -3, 5, -3, 0, 5, -3, -3, 5],
                [-3, 5, 5, -3, 0, 5, -3, -3, -3]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(8)


class ExKirsch(EdgeDetect):
    """Extended Russell Kirsch compass operator. 5x5 matrices."""
    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[9, 9, 9, 9, 9, 9, 5, 5, 5, 9, -7, -3, 0, -3, -7, -7, -3, -3, -3, -7, -7, -7, -7, -7, -7],
                [9, 9, 9, 9, -7, 9, 5, 5, -3, -7, 9, 5, 0, -3, -7, 9, -3, -3, -3, -7, -7, -7, -7, -7, -7],
                [9, 9, -7, -7, -7, 9, 5, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, -3, -3, -7, 9, 9, -7, -7, -7],
                [-7, -7, -7, -7, -7, 9, -3, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, 5, -3, -7, 9, 9, 9, 9, -7],
                [-7, -7, -7, -7, -7, -7, -3, -3, -3, -7, -7, -3, 0, -3, -7, 9, 5, 5, 5, 9, 9, 9, 9, 9, 9],
                [-7, -7, -7, -7, -7, -7, -3, -3, -3, 9, -7, -3, 0, 5, 9, -7, -3, 5, 5, 9, -7, 9, 9, 9, 9],
                [-7, -7, -7, 9, 9, -7, -3, -3, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, 5, 9, -7, -7, -7, 9, 9],
                [-7, 9, 9, 9, 9, -7, -3, 5, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, -3, 9, -7, -7, -7, -7, -7]]

    @staticmethod
    def _get_expr() -> Optional[str]:
        return max_expr(8)


class MinMax(EdgeDetect):
    """Min/max mask with separate luma/chroma radii."""
    radii: Tuple[int, int, int]

    def __init__(self, rady: int = 2, radc: int = 0) -> None:
        super().__init__()
        self.radii = (rady, radc, radc)

    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        assert clip.format
        planes = [
            core.std.Expr(
                [self._minmax(p, rad, core.std.Maximum),
                 self._minmax(p, rad, core.std.Minimum)],
                'x y -'
            )
            for p, rad in zip(split(clip), self.radii)
        ]
        return planes[0] if len(planes) == 1 else join(planes, clip.format.color_family)

    @staticmethod
    def _get_matrices() -> List[List[float]]:
        return [[]]

    @staticmethod
    def _minmax(clip: vs.VideoNode, iterations: int, morpho: Callable[..., vs.VideoNode]) -> vs.VideoNode:
        for i in range(1, iterations + 1):
            coord = [0, 1, 0, 1, 1, 0, 1, 0] if (i % 3) != 1 else [1] * 8
            clip = morpho(clip, coordinates=coord)
        return clip


def get_all_edge_detects(clip: vs.VideoNode, **kwargs: Any) -> List[vs.VideoNode]:
    """Allows you to get all masks inheriting from EdgeDetect.

    Args:
        clip (vs.VideoNode):
            Source clip.

        kwargs:
            Arguments passed to EdgeDetect().get_mask

    Returns:
        List[vs.VideoNode]: List of masks.

    Example:
        from vardefunc.mask import get_all_edge_detect

        clip.set_output(0)

        for i, mask in enumerate(get_all_edge_detect(get_y(clip)), start=1):
            mask.set_output(i)
    """
    warnings.warn(
        'vardefunc.mask.get_all_edge_detects is deprecated in favor of vsmask.\n'
        'Please install it at https://github.com/Irrational-Encoding-Wizardry/vsmask',
        DeprecationWarning
    )
    masks = [
        edge_detect().get_mask(clip, **kwargs).text.Text(edge_detect.__name__)  # type: ignore
        for edge_detect in EdgeDetect.__subclasses__()
    ]
    return masks


def region_mask(clip: vs.VideoNode,
                left: int = 0, right: int = 0,
                top: int = 0, bottom: int = 0) -> vs.VideoNode:
    """Crop your mask

    Args:
        clip (vs.VideoNode):
            Source clip.
        left (int, optional):
            Left parameter in std.CropRel or std.Crop. Defaults to 0.

        right (int, optional):
            Right parameter in std.CropRel or std.Crop. Defaults to 0.

        top (int, optional):
            Top parameter in std.CropRel or std.Crop. Defaults to 0.

        bottom (int, optional):
            Bottom parameter in std.CropRel or std.Crop. Defaults to 0.

    Returns:
        vs.VideoNode: Cropped clip.
    """
    warnings.warn(
        'vardefunc.mask.region_mask s deprecated in favor of vsmask.\n'
        'Please install it at https://github.com/Irrational-Encoding-Wizardry/vsmask',
        DeprecationWarning
    )

    crop = core.std.Crop(clip, left, right, top, bottom)
    return core.std.AddBorders(crop, left, right, top, bottom)
