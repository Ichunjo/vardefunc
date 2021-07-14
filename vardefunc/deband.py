"""Various functions used for debanding."""
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

import vapoursynth as vs
from vsutil import depth, get_depth

from .util import FormatError

core = vs.core


class SampleMode(IntEnum):
    COLUMN = 1
    SQUARE = 2
    ROW = 3
    COL_ROW_MEAN = 4


class F3kdb:
    """f3kdb object."""
    radius: int
    thy: int
    thcb: int
    thcr: int
    gry: int
    grc: int
    sample_mode: SampleMode
    use_neo: bool
    f3kdb_args: Dict[str, Any]

    _step: int

    def __init__(self,
                 radius: int = 16,
                 threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
                 sample_mode: SampleMode = SampleMode.SQUARE, use_neo: bool = False, **kwargs: Any) -> None:
        """ Handle debanding operations onto a clip using a set of configured parameters.

        Args:
            radius (int, optional):
                Banding detection range. Defaults to 16.

            threshold (Union[int, List[int]], optional):
                Banding detection threshold(s) for planes.
                If difference between current pixel and reference pixel is less than threshold,
                it will be considered as banded. Defaults to 30.

            grain (Union[int, List[int]], optional):
                Specifies amount of grains added in the last debanding stage. Defaults to 0.

            sample_mode (int, optional):
                Valid modes are:
                    – 1: Take 2 pixels as reference pixel. Reference pixels are in the same column of current pixel.
                    – 2: Take 4 pixels as reference pixel. Reference pixels are in the square around current pixel.
                    – 3: Take 2 pixels as reference pixel. Reference pixels are in the same row of current pixel.
                    – 4: Arithmetic mean of 1 and 3.
                Reference points are randomly picked within the range. Defaults to 2.

            use_neo (bool, optional):
                Use neo_f3kdb.Deband. Defaults to False.

            kwargs (optional):
                Arguments passed to f3kdb.Deband.

        """
        self.radius = radius

        self.thy, self.thcb, self.thcr = [threshold] * 3 if isinstance(threshold, int) else threshold + [threshold[-1]] * (3 - len(threshold))
        self.thy, self.thcb, self.thcr = [max(1, x) for x in [self.thy, self.thcb, self.thcr]]

        self.gry, self.grc = [grain] * 2 if isinstance(grain, int) else grain + [grain[-1]] * (2 - len(grain))

        if sample_mode > 2 and not use_neo:
            raise ValueError('F3kdb: f3kdb.Deband doesn\'t support SampleMode.ROW or SampleMode.COL_ROW_MEAN')

        self.sample_mode = sample_mode
        self.use_neo = use_neo

        self._step = 16 if sample_mode == 2 else 32

        self.f3kdb_args = dict(keep_tv_range=True, output_depth=16)
        self.f3kdb_args |= kwargs

    def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
            Main deband function.

        Args:
            clip (vs.VideoNode): Source clip.

        Returns:
            vs.VideoNode: Debanded clip.
        """
        if clip.format is None:
            raise FormatError('deband: Variable format not allowed!')


        if self.thy % self._step == 1 and self.thcb % self._step == 1 and self.thcr % self._step == 1:
            deband = self._pick_f3kdb(self.use_neo,
                                      clip, self.radius,
                                      self.thy, self.thcb, self.thcr,
                                      self.gry, self.grc,
                                      self.sample_mode, **self.f3kdb_args)
        else:
            loy, locb, locr = [(th - 1) // self._step * self._step + 1 for th in [self.thy, self.thcb, self.thcr]]
            hiy, hicb, hicr = [lo + self._step for lo in [loy, locb, locr]]

            lo_clip = self._pick_f3kdb(self.use_neo,
                                       clip, self.radius,
                                       loy, locb, locr,
                                       self.gry, self.grc,
                                       self.sample_mode, **self.f3kdb_args)
            hi_clip = self._pick_f3kdb(self.use_neo,
                                       clip, self.radius,
                                       hiy, hicb, hicr,
                                       self.gry, self.grc,
                                       self.sample_mode, **self.f3kdb_args)

            if clip.format.color_family == vs.GRAY:
                weight = [(self.thy - loy) / self._step]
            else:
                weight = [(self.thy - loy) / self._step, (self.thcb - locb) / self._step, (self.thcr - locr) / self._step]

            deband = core.std.Merge(lo_clip, hi_clip, weight)

        return deband

    def grain(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Convenience function that set thresholds to 1 (basically it doesn't deband)

        Args:
            clip (vs.VideoNode): Source clip.

        Returns:
            vs.VideoNode: Grained clip.
        """
        self.thy, self.thcr, self.thcb = (1, ) * 3
        return self.deband(clip)

    @staticmethod
    def _pick_f3kdb(neo: bool, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return core.neo_f3kdb.Deband(*args, **kwargs) if neo else core.f3kdb.Deband(*args, **kwargs)


def dumb3kdb(clip: vs.VideoNode, radius: int = 16,
             threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
             sample_mode: SampleMode = SampleMode.SQUARE, use_neo: bool = False, **kwargs: Any) -> vs.VideoNode:
    return F3kdb(radius, threshold, grain, sample_mode, use_neo, **kwargs).deband(clip)


def f3kbilateral(clip: vs.VideoNode, radius: int = 16,
                 threshold: Union[int, List[int]] = 65, grain: Union[int, List[int]] = 0,
                 f3kdb_args: Optional[Dict[str, Any]] = None,
                 limflt_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    """f3kdb multistage bilateral-esque filter from debandshit.
       This function is more of a last resort for extreme banding.

    Args:
        clip (vs.VideoNode): Source clip.

        radius (int, optional):
            Same as F3kdb constructor. Defaults to 16.

        threshold (Union[int, List[int]], optional):
            Same as F3kdb constructor. Defaults to 65.

        grain (Union[int, List[int]], optional):
            Same as F3kdb constructor.
            It happens after mvsfunc.LimitFilter and call another instance of F3kdb if != 0.
            Defaults to 0.

        f3kdb_args (Dict[str, Any], optional):
            Same as F3kdb constructor. Defaults to None.

        limflt_args (Dict[str, Any], optional):
            Arguments passed to mvsfunc.LimitFilter. Defaults to None.

    Returns:
        vs.VideoNode: Debanded clip.
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError as mod_err:
        raise ModuleNotFoundError("f3kbilateral: missing dependency 'mvsfunc'") from mod_err

    if clip.format is None:
        raise FormatError("f3kbilateral: 'Variable-format clips not supported'")

    bits = get_depth(clip)

    f3_args: Dict[str, Any] = dict()
    if f3kdb_args is not None:
        f3_args |= f3kdb_args

    lf_args: Dict[str, Any] = dict(thr=0.6, elast=3.0, thrc=None)
    if limflt_args is not None:
        lf_args |= limflt_args

    rad1 = round(radius * 4 / 3)
    rad2 = round(radius * 2 / 3)
    rad3 = round(radius / 3)

    db1 = F3kdb(rad1, threshold, 0, **f3_args)
    db2 = F3kdb(rad2, threshold, 0, **f3_args)
    db3 = F3kdb(rad3, threshold, 0, **f3_args)

    # Edit the thr of first f3kdb object
    db1.thy, db1.thcb, db1.thcr = [max(1, th // 2) for th in (db1.thy, db1.thcb, db1.thcr)]

    clip = depth(clip, 16)

    flt1 = db1.deband(clip)
    flt2 = db2.deband(flt1)
    flt3 = db3.deband(flt2)

    limit = LimitFilter(flt3, flt2, ref=clip, **lf_args)

    if grain:
        grained = F3kdb(grain=grain, **f3_args).grain(limit)
    else:
        grained = limit

    return depth(grained, bits)


def lfdeband(clip: vs.VideoNode) -> vs.VideoNode:
    """A simple debander ported from AviSynth by Zastin from debandshit

    Args:
        clip (vs.VideoNode): Source clip

    Returns:
        vs.VideoNode: Debanded clip.
    """
    if clip.format is None:
        raise ValueError("lfdeband: 'Variable-format clips not supported'")

    bits = get_depth(clip)
    wss, hss = 1 << clip.format.subsampling_w, 1 << clip.format.subsampling_h
    w, h = clip.width, clip.height
    dw, dh = round(w / 2), round(h / 2)

    clip = depth(clip, 16)
    dsc = core.resize.Spline64(clip, dw-dw % wss, dh-dh % hss)

    d3kdb = dumb3kdb(dsc, radius=30, threshold=80, grain=0)

    ddif = core.std.MakeDiff(d3kdb, dsc)

    dif = core.resize.Spline64(ddif, w, h)
    out = core.std.MergeDiff(clip, dif)
    return depth(out, bits)
