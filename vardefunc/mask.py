"""Functions for masking functions"""
import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Union

import lvsfunc

from vsutil import (Range, depth, get_depth, get_w, get_y, insert_clip,
                    iterate, scale_value, split)

import vapoursynth as vs

from .util import get_sample_type, max_expr, mae_expr

core = vs.core



class EdgeDetect(ABC):
    """Abstract edge detection interface."""

    def get_mask(self, clip: vs.VideoNode, lthr: int = 0, hthr: Optional[float] = None, multi: float = 1.0) -> vs.VideoNode:
        """Makes edge mask based on convolution kernel.
           The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        Args:
            clip (vs.VideoNode):
                Source clip.

            lthr (int, optional):
                Low threshold. Anything below lthr will be set to 0. Defaults to 0.

            hthr (Optional[float], optional):
                High threshold. Anything above hthr will be set to the range max. Defaults to None.

            multi (float, optional):
                Multiply all pixels by this before thresholding. Defaults to 1.0.

        Returns:
            vs.VideoNode: Mask clip.
        """
        assert clip.format is not None

        bits = get_depth(clip)
        peak = 1.0 if get_sample_type(clip) == vs.FLOAT else (1 << bits) - 1
        hthr = peak if hthr is None else hthr


        clip_p = self._preprocess(clip)
        mask = self._compute_mask(clip_p)

        mask = depth(mask, bits, range=Range.FULL, range_in=Range.FULL)


        if multi != 1:
            mask = core.std.Expr(mask, f'x {multi} *')


        if lthr > 0 or hthr < peak:
            mask = core.std.Expr(mask, f'x {hthr} > {peak} x {lthr} <= 0 x ? ?')


        return mask

    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        masks = [clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
                 for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())]

        expr = self._get_expr()
        mask = core.std.Expr(masks, expr) if expr else masks[0]

        return mask

    def _get_divisors(self) -> List[Union[int, float]]:
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
    def _get_matrices(self) -> None:
        pass

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
    def _get_matrices(self) -> None:
        pass

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
    """Flow-based Difference Of Gaussian operator. 3x3 matrices from G41Fun."""
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
    def _get_matrices(self) -> None:
        pass

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


def diff_rescale_mask(clip: vs.VideoNode, height: int = 720,
                      kernel: lvsfunc.kernels.Kernel = lvsfunc.kernels.Bicubic(b=0, c=0.5),
                      thr: Union[int, float] = 55,
                      sw: int = 2, sh: int = 2) -> vs.VideoNode:
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

        sw (int, optional):
            Growing/shrinking shape width. 0 is allowed. Defaults to 2.

        sh (int, optional):
            Growing/shrinking shape height. 0 is allowed. Defaults to 2.

    Returns:
        vs.VideoNode: Rescaled mask.
    """
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

    diff = core.std.Expr(sum(map(split, [pre, rescale]), []), mae_expr(gray_only))

    mask = iterate(diff, lambda x: core.rgsf.RemoveGrain(x, 2), 2)
    mask = core.std.Expr(mask, f'x 2 4 pow * {thr} < 0 1 ?')

    mask = iterate(mask, partial(core.std.Maximum, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]), 2 + sw)
    mask = iterate(mask, partial(core.std.Maximum, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]), 2 + sh)
    mask = mask.std.Deflate()

    return mask.resize.Point(
        format=clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
        dither_type='none'
    )


def diff_creditless_mask(src_clip: vs.VideoNode, credit_clip: vs.VideoNode, nc_clip: vs.VideoNode,
                         start_frame: int, thr: int, sw: int = 2, sh: int = 2, *,
                         prefilter: bool = False, bilateral_args: Dict[str, Any] = {}) -> vs.VideoNode:
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

        sw (int, optional): Growing/shrinking shape width.
            0 is allowed. Defaults to 2.

        sh (int, optional): Growing/shrinking shape height.
            0 is allowed. Defaults to 2.

        prefilter (bool, optional):
            Blurs the credit_clip and nc_clip with Bilateral to avoid false posivive
            such as noise and compression artifacts.
            Defaults to False.

        bilateral_args (Dict[str, Any], optional):
            Additionnal and overrided Bilateral parameters if prefilter=True. Defaults to {}.

    Returns:
        vs.VideoNode: Credit mask clip.

    Example:
        import vardefunc as vdf

        opstart, opend = 792, 2948

        opmask = diff_creditless_mask(clip, clip[opstart:opend+1], ncop[:opend+1-opstart], opstart, thr=25, prefilter=True)
    """
    gray_only = src_clip.format.num_planes == 1
    clips = [credit_clip, nc_clip]

    if prefilter:
        bilargs: Dict[str, Any] = dict(sigmaS=((5 ** 2 - 1) / 12) ** 0.5, sigmaR=0.5)
        bilargs.update(bilateral_args)
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
        format=src_clip.format.replace(color_family=vs.GRAY).id
    )

    mask = core.std.Prewitt(diff).std.Binarize(thr)
    mask = iterate(mask, partial(core.std.Maximum, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]), sw)
    mask = iterate(mask, partial(core.std.Maximum, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]), sh)

    blank = core.std.BlankClip(
        src_clip, format=src_clip.format.replace(
            color_family=vs.GRAY, subsampling_w=0, subsampling_h=0
        ).id
    )
    mask = insert_clip(blank, mask, start_frame)

    return mask


def luma_credit_mask(clip: vs.VideoNode, thr: int = 230,
                     edgemask: EdgeDetect = Prewitt(), draft: bool = False) -> vs.VideoNode:
    """Makes a mask based on luma value and edges.

    Args:
        clip (vs.VideoNode):
            Source clip.

        thr (int, optional):
            Luma value assuming 8 bit input. Defaults to 230.

        edgemask (EdgeDetect, optional):
            Edge mask used with thr. Defaults to Prewitt().

        draft (bool, optional):
            Allow to output the mask without growing. Defaults to False.

    Returns:
        vs.VideoNode: Credit mask.
    """
    clip = get_y(clip)

    edge_mask = edgemask.get_mask(clip)
    luma_mask = core.std.Expr(clip, f'x {thr} > x 0 ?')

    credit_mask = core.std.Expr([edge_mask, luma_mask], 'x y min')

    if not draft:
        credit_mask = iterate(credit_mask, core.std.Maximum, 4)
        credit_mask = iterate(credit_mask, core.std.Inflate, 2)

    return credit_mask


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
    crop = core.std.Crop(clip, left, right, top, bottom)
    return core.std.AddBorders(crop, left, right, top, bottom)
