"""Functions for masking functions"""
import math
from functools import partial
from string import ascii_lowercase
from typing import Any, Dict, List

import fvsfunc as fvf
import havsfunc as hvf

from vsutil import depth, get_depth, get_w, get_y, insert_clip, iterate, split
import vapoursynth as vs

from .util import get_sample_type

core = vs.core


def diff_rescale_mask(source: vs.VideoNode, height: int = 720, kernel: str = 'bicubic',
                      b: float = 0, c: float = 1 / 2, mthr: int = 55,
                      mode: str = 'ellipse', sw: int = 2, sh: int = 2) -> vs.VideoNode:
    """Modified version of Atomchtools for generate a mask with a rescaled difference.
       Its alias is vardefunc.drm

    Args:
        source (vs.VideoNode): Source clip.
        height (int, optional): Defaults to 720.
        kernel (str, optional): Defaults to bicubic.
        b (float, optional): Defaults to 0.
        c (float, optional): Defaults to 1/2.
        mthr (int, optional): Defaults to 55.
        mode (str, optional): Can be 'rectangle', 'losange' or 'ellipse' . Defaults to 'ellipse'.
        sw (int, optional): Growing/shrinking shape width. 0 is allowed. Defaults to 2.
        sh (int, optional): Growing/shrinking shape height. 0 is allowed. Defaults to 2.

    Returns:
        vs.VideoNode:
    """
    if not source.format.num_planes == 1:
        clip = get_y(source)
    else:
        clip = source

    if get_depth(source) != 8:
        clip = depth(clip, 8)

    width = get_w(height)
    desc = fvf.Resize(clip, width, height, kernel=kernel, a1=b, a2=c, invks=True)
    upsc = depth(fvf.Resize(desc, source.width, source.height, kernel=kernel, a1=b, a2=c), 8)

    diff = core.std.MakeDiff(clip, upsc)
    mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
    mask = mask.std.Expr(f'x {mthr} < 0 x ?')
    mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    mask = hvf.mt_expand_multi(mask, mode=mode, sw=sw, sh=sh)

    if get_depth(source) != 8:
        mask = depth(mask, get_depth(source))
    return mask


def diff_creditless_mask(src_clip: vs.VideoNode, credit_clip: vs.VideoNode, nc_clip: vs.VideoNode,
                         start_frame: int, thr: int, sw: int = 2, sh: int = 2, *,
                         prefilter: bool = False, bilateral_args: Dict[str, Any] = {}) -> vs.VideoNode:
    """Makes a mask based on difference from 2 clips.

    Args:
        src_clip (vs.VideoNode): Source clip.

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
            Blurs the credit_clip and nc_clip to avoid false posivive such as noise and compression artifacts.
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
    clips = [credit_clip, nc_clip]

    if prefilter:
        bilargs: Dict[str, Any] = dict(sigmaS=((5 ** 2 - 1) / 12) ** 0.5, sigmaR=0.5)
        bilargs.update(bilateral_args)
        clips = [c.bilateral.Bilateral(**bilargs) for c in clips]


    clips = [
        c.resize.Bicubic(
            format=src_clip.format.replace(
                bits_per_sample=get_depth(src_clip), subsampling_w=0, subsampling_h=0)
        ) for c in clips]

    diff = core.std.Expr(
        sum(map(split, clips), []),
        'x a - abs y b - abs max z c - abs max',  # MAE
        # 'x a - 2 pow sqrt y b - 2 pow sqrt max z c - 2 pow sqrt max',  # RMSE
        format=src_clip.format.replace(color_family=vs.GRAY)
    )

    mask = core.std.Prewitt(diff).std.Binarize(thr)
    mask = iterate(mask, partial(core.std.Maximum, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]), sw)
    mask = iterate(mask, partial(core.std.Maximum, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]), sh)

    blank = core.std.BlankClip(src_clip, format=src_clip.format.replace(
        color_family=vs.GRAY, subsampling_w=0, subsampling_h=0))
    mask = insert_clip(blank, mask, start_frame)

    return mask


def luma_credit_mask(source: vs.VideoNode, thr: int = 230, mode: str = 'prewitt', draft: bool = False) -> vs.VideoNode:
    """Creates a mask based on luma value and edges.

    Args:
        source (vs.VideoNode): Source clip.
        thr (int, optional): Luma value assuming 8 bit input. Defaults to 230.
        mode (str, optional): Chooses a predefined kernel used for the mask computing.
                              Valid choices are "sobel", "prewitt", "scharr", "kirsch",
                              "robinson", "roberts", "cartoon", "min/max", "laplace",
                              "frei-chen", "kayyali", "LoG", "FDOG" and "TEdge".
                              Defaults to 'prewitt'.
        draft (bool, optional): Allow to output the mask without growing. Defaults to False.

    Returns:
        vs.VideoNode: Credit mask.
    """
    try:
        import G41Fun as gf
    except ModuleNotFoundError:
        raise ModuleNotFoundError("luma_credit_mask: missing dependency 'G41Fun'")

    if not source.format.num_planes == 1:
        clip = get_y(source)
    else:
        clip = source

    edge_mask = gf.EdgeDetect(clip, mode).std.Maximum()
    luma_mask = core.std.Expr(clip, f'x {thr} > x 0 ?')

    credit_mask = core.std.Expr([edge_mask, luma_mask], 'x y min')

    if not draft:
        credit_mask = iterate(credit_mask, core.std.Maximum, 4)
        credit_mask = iterate(credit_mask, core.std.Inflate, 2)

    return credit_mask


class Operator:
    """Object representing an operator or filter used in edge detection"""
    def __init__(self, matrixes: List[List[int]],
                 divisors: List[float] = None, mode_types: List[str] = None,
                 expr: str = None) -> None:
        length = len(matrixes)
        self.matrixes = matrixes
        self.divisors = divisors if divisors else [0.0] * length
        self.mode_types = mode_types if mode_types else ['s'] * length
        self.expr = expr


def edge_detect(clip: vs.VideoNode, mode: str, lthr: int = None, hthr: int = None, multi: float = 1.0) -> vs.VideoNode:
    """Makes edge mask based on convolution kernel.
       The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

    Args:
        clip (vs.VideoNode):
            Source clip.

        mode (str):
            Operator used for the mask computing.
            Valid choices:
                – "sobel"           (Sobel–Feldman operator) 3x3 matrixes.
                – "ex_sobel"        (Extended Sobel–Feldman operator) 5x5 matrixes.
                – "prewitt"         (Judith M. S. Prewitt operator) 3x3 matrixes.
                – "ex_prewitt"      (Extended Judith M. S. Prewitt operator) 5x5 matrixes.
                – "scharr"          (H. Scharr optimized operator) 3x3 matrixes.
                – "scharr_opt"      (H. Scharr optimal operator) 3x3 matrixes.
                – "roberts"         (Lawrence Roberts operator) 2x2 matrixes computed in 3x3 matrixes.
                – "fdog"            (Flow-based Difference Of Gaussian operator) 3x3 matrixes from G41Fun.
                – "frei_chen_g41"   (Chen Frei operator) 3x3 matrixes from G41Fun.
                – "frei_chen"       (Chen Frei operator) 3x3 matrixes properly implemented.
                – "tedge"           (TEdgeMask(type=2) Avisynth plugin) 3x3 matrixes.
                – "kirsch"          (Russell Kirsch compass operator) 3x3 matrixes.
                – "ex_kirsch"       (Extended Russell Kirsch compass operator) 5x5 matrixes.
                – "robinson_3"      (Robinson compass operator level 3) 3x3 matrixes.
                – "robinson_5"      (Robinson compass operator level 5) 3x3 matrixes.
                – "laplacian_1"     (Pierre-Simon de Laplace operator 1st implementation) 3x3 matrix.
                – "laplacian_2"     (Pierre-Simon de Laplace operator 2nd implementation) 3x3 matrix.
                – "laplacian_3"     (Pierre-Simon de Laplace operator 3rd implementation) 3x3 matrix.
                – "laplacian_4"     (Pierre-Simon de Laplace operator 4th implementation) 3x3 matrix.
                – "ex_laplacian_1"  (Extended Pierre-Simon de Laplace operator 1st implementation) 5x5 matrix.
                – "ex_laplacian_2"  (Extended Pierre-Simon de Laplace operator 2nd implementation) 5x5 matrix.
                – "ex_laplacian_3"  (Extended Pierre-Simon de Laplace operator 3rd implementation) 5x5 matrix.
                – "ex_laplacian_4"  (Extended Pierre-Simon de Laplace operator 4th implementation) 5x5 matrix.
                – "log"             (Laplacian of Gaussian) 5x5 matrix.
                – "kayyali"         (Kayyali operator) 3x3 matrix
                – "kroon"           (Dirk-Jan Kroon operator) 3x3 matrixes.

        lthr (int, optional):
            Low threshold. Anything below lthr will be set to 0.

        hthr (int, optional):
            High threshold. Anything above hthr will be set to the range max.

        multi (float, optional):
            Multiply all pixels by this before thresholding. Defaults to 1.0.

    Returns:
        vs.VideoNode: Mask.
    """

    bits = get_depth(clip)
    is_float = get_sample_type(clip) == vs.FLOAT
    peak = 1.0 if is_float else (1 << bits) - 1
    lthr = 0 if not lthr else lthr
    hthr = peak if not hthr else hthr


    ope = get_edge_detect_operator(mode)


    if mode == 'frei_chen':
        clip = depth(clip, 32)


    masks = [clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
             for mat, div, mode in zip(ope.matrixes, ope.divisors, ope.mode_types)]

    if ope.expr:
        mask = core.std.Expr(masks, ope.expr)
        mask = depth(mask, bits)
    else:
        mask = masks[0]

    mask = depth(mask, bits)


    if multi != 1.0:
        mask = core.std.Expr(mask, f'x {multi} *')


    if lthr > 0 or hthr < peak:
        mask = core.std.Expr(mask, f'x {hthr} > {peak} x {lthr} <= 0 x ? ?')


    return mask


def get_edge_detect_operator(operator: str) -> Operator:
    operator = operator.lower().replace(' ', '_').replace('-', '_')

    sqrt2 = math.sqrt(2)

    abcd = list(ascii_lowercase)
    abcd = abcd[-1:] + abcd[:-3]
    M = 'x x * y y * + z z * + a a * +'
    S = f'b b * c c * + d d * + e e * + f f * + {M} +'

    expr_gradmag = 'x x * y y * + sqrt'
    expr_max4 = 'x y max ' + ' max '.join(abcd[i] for i in range(4 - 2)) + ' max'
    expr_max8 = 'x y max ' + ' max '.join(abcd[i] for i in range(8 - 2)) + ' max'
    expr_freichen = f'{M} {S} / sqrt'

    operators = {
        'sobel': Operator(
            matrixes=[[1, 0, -1, 2, 0, -2, 1, 0, -1],
                      [1, 2, 1, 0, 0, 0, -1, -2, -1]],
            expr=expr_gradmag
        ),
        'ex_sobel': Operator(
            matrixes=[[2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
                      [2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2]],
            expr=expr_gradmag
        ),
        'prewitt': Operator(
            matrixes=[[1, 0, -1, 1, 0, -1, 1, 0, -1],
                      [1, 1, 1, 0, 0, 0, -1, -1, -1]],
            expr=expr_gradmag
        ),
        'ex_prewitt': Operator(
            matrixes=[[2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
                      [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2]],
            expr=expr_gradmag
        ),
        'scharr': Operator(
            matrixes=[[-3, 0, 3, -10, 0, 10, -3, 0, 3],
                      [-3, -10, -3, 0, 0, 0, 3, 10, 3]],
            expr=expr_gradmag
        ),
        'scharr_opt': Operator(
            matrixes=[[-47, 0, 47, -162, 0, 162, -47, 0, 47],
                      [-47, -162, -47, 0, 0, 0, 47, 162, 47]],
            expr=expr_gradmag
        ),
        'roberts': Operator(
            matrixes=[[0, 0, 0, 0, 1, 0, 0, 0, -1],
                      [0, 1, 0, -1, 0, 0, 0, 0, 0]],
            expr=expr_gradmag
        ),
        'fdog': Operator(
            matrixes=[[1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1],
                      [1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -2, -1, -1, -2, -3, -2, -1]],
            divisors=[2, 2], expr=expr_gradmag
        ),
        'frei_chen_g41': Operator(
            matrixes=[[-7, 0, 7, -10, 0, 10, -7, 0, 7],
                      [-7, -10, -7, 0, 0, 0, 7, 10, 7]],
            divisors=[7, 7], expr=expr_gradmag
        ),
        'kroon': Operator(
            matrixes=[[-17, 0, 17, -61, 0, 61, -17, 0, 17],
                      [-17, -61, -17, 0, 0, 0, 17, 61, 17]],
            expr=expr_gradmag
        ),
        'tedge': Operator(
            matrixes=[[12, -74, 0, 74, -12],
                      [-12, 74, 0, -74, 12]],
            divisors=[62, 62], mode_types=['h', 'v'], expr=expr_gradmag
        ),
        'kirsch': Operator(
            matrixes=[[5, 5, 5, -3, 0, -3, -3, -3, -3],
                      [5, 5, -3, 5, 0, -3, -3, -3, -3],
                      [5, -3, -3, 5, 0, -3, 5, -3, -3],
                      [-3, -3, -3, 5, 0, -3, 5, 5, -3],
                      [-3, -3, -3, -3, 0, -3, 5, 5, 5],
                      [-3, -3, -3, -3, 0, 5, -3, 5, 5],
                      [-3, -3, 5, -3, 0, 5, -3, -3, 5],
                      [-3, 5, 5, -3, 0, 5, -3, -3, -3]],
            expr=expr_max8
        ),
        'ex_kirsch': Operator(
            matrixes=[[9, 9, 9, 9, 9, 9, 5, 5, 5, 9, -7, -3, 0, -3, -7, -7, -3, -3, -3, -7, -7, -7, -7, -7, -7],
                      [9, 9, 9, 9, -7, 9, 5, 5, -3, -7, 9, 5, 0, -3, -7, 9, -3, -3, -3, -7, -7, -7, -7, -7, -7],
                      [9, 9, -7, -7, -7, 9, 5, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, -3, -3, -7, 9, 9, -7, -7, -7],
                      [-7, -7, -7, -7, -7, 9, -3, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, 5, -3, -7, 9, 9, 9, 9, -7],
                      [-7, -7, -7, -7, -7, -7, -3, -3, -3, -7, -7, -3, 0, -3, -7, 9, 5, 5, 5, 9, 9, 9, 9, 9, 9],
                      [-7, -7, -7, -7, -7, -7, -3, -3, -3, 9, -7, -3, 0, 5, 9, -7, -3, 5, 5, 9, -7, 9, 9, 9, 9],
                      [-7, -7, -7, 9, 9, -7, -3, -3, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, 5, 9, -7, -7, -7, 9, 9],
                      [-7, 9, 9, 9, 9, -7, -3, 5, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, -3, 9, -7, -7, -7, -7, -7]],
            expr=expr_max8
        ),
        'robinson_3': Operator(
            matrixes=[[1, 1, 1, 0, 0, 0, -1, -1, -1],
                      [1, 1, 0, 1, 0, -1, 0, -1, -1],
                      [1, 0, -1, 1, 0, -1, 1, 0, -1],
                      [0, -1, -1, 1, 0, -1, 1, 1, 0]],
            expr=expr_max4
        ),
        'robinson_5': Operator(
            matrixes=[[1, 2, 1, 0, 0, 0, -1, -2, -1],
                      [2, 1, 0, 1, 0, -1, 0, -1, -2],
                      [1, 0, -1, 2, 0, -2, 1, 0, -1],
                      [0, -1, -2, 1, 0, -1, 2, 1, 0]],
            expr=expr_max4
        ),
        'frei_chen': Operator(
            matrixes=[[1, sqrt2, 1, 0, 0, 0, -1, -sqrt2, -1],
                      [1, 0, -1, sqrt2, 0, -sqrt2, 1, 0, -1],
                      [0, -1, sqrt2, 1, 0, -1, -sqrt2, 1, 0],
                      [sqrt2, -1, 0, -1, 0, 1, 0, 1, -sqrt2],
                      [0, 1, 0, -1, 0, -1, 0, 1, 0],
                      [-1, 0, 1, 0, 0, 0, 1, 0, -1],
                      [1, -2, 1, -2, 4, -2, 1, -2, 1],
                      [-2, 1, -2, 1, 4, 1, -2, 1, -2],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            divisors=[2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2, 2, 6, 6, 3],
            expr=expr_freichen
        ),
        'laplacian_1': Operator(matrixes=[[0, -1, 0, -1, 4, -1, 0, -1, 0]]),
        'laplacian_2': Operator(matrixes=[[1, -2, 1, -2, 4, -2, 1, -2, 1]]),
        'laplacian_3': Operator(matrixes=[[2, -1, 2, -1, -4, -1, 2, -1, 2]]),
        'laplacian_4': Operator(matrixes=[[-1, -1, -1, -1, 8, -1, -1, -1, -1]]),
        'ex_laplacian_1': Operator(matrixes=[[0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, 8, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]]),
        'ex_laplacian_2': Operator(matrixes=[[0, 1, -1, 1, 0, 1, 1, -4, 1, 1, -1, -4, 8, -4, -1, 1, 1, -4, 1, 1, 0, 1, -1, 1, 0]]),
        'ex_laplacian_3': Operator(matrixes=[[-1, 1, -1, 1, -1, 1, 2, -4, 2, 1, -1, -4, 8, -4, -1, 1, 2, -4, 2, 1, -1, 1, -1, 1, -1]]),
        'ex_laplacian_4': Operator(matrixes=[[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]),
        'log': Operator(matrixes=[[0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0]]),
        'kayyali': Operator(matrixes=[[6, 0, -6, 0, 0, 0, -6, 0, 6]])
    }
    return operators[operator]



def region_mask(clip: vs.VideoNode,
                left: int = 0, right: int = 0,
                top: int = 0, bottom: int = 0) -> vs.VideoNode:
    """Crop your mask

    Args:
        clip (vs.VideoNode): Source clip.
        left (int, optional): Left parameter in std.CropRel or std.Crop. Defaults to 0.
        right (int, optional): Right parameter in std.CropRel or std.Crop. Defaults to 0.
        top (int, optional): Top parameter in std.CropRel or std.Crop. Defaults to 0.
        bottom (int, optional): Bottom parameter in std.CropRel or std.Crop. Defaults to 0.

    Returns:
        vs.VideoNode: Cropped clip
    """
    crop = core.std.Crop(clip, left, right, top, bottom)
    return core.std.AddBorders(crop, left, right, top, bottom)
