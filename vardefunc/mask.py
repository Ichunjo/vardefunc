"""Wrappers and masks for denoising, debanding, rescaling etc."""

__all__ = [
    'detail_mask',
    'Difference', 'diff_creditless_mask', 'diff_rescale_mask', 'luma_mask', 'luma_credit_mask', 'region_mask'
]

import warnings
from typing import Any, Dict, Optional, Union

import vapoursynth as vs
import vskernels
from vsmask.edge import EdgeDetect as EdgeDetectVsM
from vsmask.edge import ExLaplacian4 as ExLaplacian4VsM
from vsmask.edge import FDoGTCanny as FDoGTCannyVsM
from vsmask.edge import Kirsch as KirschVsM
from vsmask.edge import MinMax as MinMaxVsM
from vsmask.util import XxpandMode
from vsmask.util import expand as expand_func
from vsutil import get_depth, get_w, get_y, insert_clip, iterate, scale_value, split

from .types import format_not_none
from .util import get_sample_type, mae_expr, pick_px_op

core = vs.core


def detail_mask(clip: vs.VideoNode, brz_mm: float, brz_ed: float,
                minmax: MinMaxVsM = MinMaxVsM(rady=3, radc=2),
                edgedetect: EdgeDetectVsM = KirschVsM()) -> vs.VideoNode:
    if clip.format is None:
        raise ValueError("detail_mask: 'Variable-format clips not supported'")

    range_mask = minmax.edgemask(clip).std.Binarize(brz_mm)
    edges = edgedetect.edgemask(clip).std.Binarize(brz_ed)

    mask = core.std.Expr((range_mask, edges), 'x y max')

    removegrain = core.rgvs.RemoveGrain if clip.format.bits_per_sample < 32 else core.rgsf.RemoveGrain

    mask = removegrain(mask, 22)
    mask = removegrain(mask, 11)

    return mask


class Difference():
    """Collection of function based on differences between prediction and observation"""

    def rescale(self, clip: vs.VideoNode, height: int = 720,
                kernel: vskernels.Kernel = vskernels.Catrom(),
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

            kernel (vskernels.Kernel, optional):
                Kernel used to descale. Defaults to vskernels.Bicubic(b=0, c=0.5).

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

        mask = ExLaplacian4VsM().edgemask(diff).std.Binarize(thr)
        mask = expand_func(mask, 2 + expand, mode=XxpandMode.ELLIPSE)

        blank = core.std.BlankClip(
            src_clip, format=src_clip.format.replace(
                color_family=vs.GRAY, subsampling_w=0, subsampling_h=0
            ).id
        )
        if blank.num_frames == mask.num_frames:
            return mask
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
                      kernel: vskernels.Kernel = vskernels.Catrom(),
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
                     edgemask: EdgeDetectVsM = FDoGTCannyVsM(), draft: bool = False) -> vs.VideoNode:
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

    edge_mask = edgemask.edgemask(clip)

    credit_mask = core.std.Expr([edge_mask, clip], f'y {thr} > y 0 ? x min')

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
    warnings.warn(
        'vardefunc.mask.region_mask s deprecated in favor of vsmask.\n'
        'Please install it at https://github.com/Irrational-Encoding-Wizardry/vsmask',
        DeprecationWarning
    )

    crop = core.std.Crop(clip, left, right, top, bottom)
    return core.std.AddBorders(crop, left, right, top, bottom)
