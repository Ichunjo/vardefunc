
from functools import partial
import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import vapoursynth as vs
from lvsfunc.render import clip_async_render

from .mask import region_mask
from .types import AnyPath

core = vs.core


class OCR:
    clip: vs.VideoNode
    coord: Tuple[int, int, int]
    coord_alt: Optional[Tuple[int, int, int]]
    thr_in: Sequence[int]
    thr_out: Sequence[int]
    thr_scd: float

    results: List[Tuple[int, bytes]]

    _brd_crop: int = 10

    def __init__(self, clip: vs.VideoNode, coord: Tuple[int, int, int],
                 coord_alt: Optional[Tuple[int, int, int]] = None,
                 thr_in: Union[int, Tuple[int, int, int]] = 225,
                 thr_out: Union[int, Tuple[int, int, int]] = 80) -> None:
        self.clip = clip
        assert self.clip.format

        self.coord = coord
        self.coord_alt = coord_alt

        self.thr_in = thr_in if isinstance(thr_in, tuple) else [thr_in]
        self.thr_out = thr_out if isinstance(thr_out, tuple) else [thr_out]

        if len(set([self.clip.format.num_planes, len(self.thr_in)])) > 1:
            raise ValueError


    def launch(self, **kwargs: Any) -> None:
        """http://www.vapoursynth.com/doc/plugins/ocr.html"""
        ppclip = self._cleaning(self._cropping(self.clip, self.coord, False))
        ocred = core.ocr.Recognize(ppclip, **kwargs)

        def _select_clips(n: int, f: vs.VideoFrame, clips: List[vs.VideoNode]) -> vs.VideoNode:
            return clips[1] if cast(int, f.props['PlaneStatsMax']) > 0 else clips[0]

        ocred = core.std.FrameEval(
            ppclip, partial(_select_clips, clips=[ppclip, ocred]),
            prop_src=ppclip.std.PlaneStats()
        )

        results: Set[Tuple[int, bytes]] = set()

        def _callback(n: int, f: vs.VideoFrame) -> None:
            if (prop_ocr := 'OCRString') in f.props.keys():
                results.add((n, cast(bytes, f.props[prop_ocr])))

        clip_async_render(ocred, callback=_callback)
        self.results = sorted(results)

    def write_ass(
        self, output: AnyPath,
        string_replace: List[Tuple[str, str]] = [
            ('_', '-'), ('…', '...'), ('‘', "'"), ('’', "'"),
            (" '", "'")
        ]
    ) -> None:
        """[summary]

        Args:
            output (AnyPath): [description]

            string_replace (List[Tuple[str, str]], optional):
                [description]. Defaults to [ ('_', '-'), ('…', '...'), ('‘', "'"), ('’', "'"), (" '", "'") ].
        """
        resultsd: Dict[int, Tuple[int, str]] = {}
        for frame, string in sorted(self.results):
            string = string.decode('utf-8').replace('\n', '\\N')
            for r in string_replace:
                string = string.replace(*r)
            resultsd[frame] = (frame + 1, string)

        results_s = sorted(resultsd.items(), reverse=True)

        for (start1, (end1, string1)), (start2, (end2, string2)) in zip(results_s, results_s[1:]):
            if string1 == string2 and end2 == start1:
                resultsd[start2] = (max(end1, resultsd[start1][0]), string1)
                del resultsd[start1]

        nresults = sorted(resultsd.items())
        fps = self.clip.fps

        with open(output, 'w', encoding='utf-8-sig') as ass:
            ass.write('[Events]\n')
            ass.write('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
            ass.writelines(
                f'Dialogue: 0,{self._f2assts(s, fps)},{self._f2assts(e, fps)},Default,,0,0,0,,{string}\n'
                for s, (e, string) in nresults
            )

    def _f2assts(self, f: int, fps: Fraction, /) -> str:
        s = self._f2seconds(f, fps) - fps.denominator*0.5 / fps.numerator
        s = max(0, s)
        m = s // 60
        s %= 60
        h = m // 60
        m %= 60
        return f"{h:02.0f}:{m:02.0f}:{math.trunc(s*100)/100:05.2f}"

    @staticmethod
    def _f2seconds(f: int, fps: Fraction, /) -> float:
        if f == 0:
            return 0.0

        t = round(float(10 ** 9 * f * fps ** -1))
        s = t / 10 ** 9
        return s

    def _cropping(self, clip: vs.VideoNode, c: Tuple[int, int, int], alt: bool) -> vs.VideoNode:
        cw, ch, h = c
        wcrop = (self.clip.width - cw) / 2
        hcrop = h if alt else self.clip.height - ch - h

        return clip.std.CropAbs(cw, ch, round(wcrop), hcrop)

    def _cleaning(self, clip: vs.VideoNode) -> vs.VideoNode:
        clip_black = clip.std.BlankClip(
            clip.width - self._brd_crop, clip.height - self._brd_crop
        )
        xxxx = [self._brd_crop / 2] * 4
        rectangle = core.std.AddBorders(clip_black, *xxxx, color=255)

        white_raw = clip.std.Binarize(self.thr_in)
        bright_raw = clip.std.Binarize(self.thr_out)

        bright_out = core.std.Expr([bright_raw, rectangle], 'x y min')
        bright_not = core.misc.Hysteresis(bright_out, bright_raw).std.InvertMask()
        white_txt = core.std.MaskedMerge(clip.std.BlankClip(), white_raw, bright_not)

        try:
            return white_txt.rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
        except vs.Error:
            return white_txt.rgsf.RemoveGrain(3).rgsf.RemoveGrain(3)

    @property
    def preview_cropped(self) -> vs.VideoNode:
        cmask = self._compute_preview_cropped(self.coord, False)

        if self.coord_alt:
            cmask_alt = self._compute_preview_cropped(self.coord_alt, True)
            cmask = core.std.Lut2(cmask, cmask_alt, function=lambda x, y: max(x, y))

        return core.std.MaskedMerge(
            core.std.Lut(self.clip, function=lambda x: round(x/2)),
            self.clip, cmask
        )

    def _compute_preview_cropped(self, c: Tuple[int, int, int], alt: bool) -> vs.VideoNode:
        cw, ch, h = c
        wcrop = (self.clip.width - cw) / 2
        left, right = math.ceil(wcrop), math.floor(wcrop)
        hcrop = self.clip.height - ch - h, h
        if alt:
            hcrop = hcrop[::-1]
        return region_mask(
            self.clip.std.BlankClip(format=vs.GRAY8, color=255),
            left, right, *hcrop
        )

    @property
    def preview_cleaned(self) -> vs.VideoNode:
        cclip = self._cleaning(self._cropping(self.clip, self.coord, False))

        if self.coord_alt:
            cclip_alt = self._cleaning(self._cropping(self.clip, self.coord_alt, True))
        else:
            return cclip

        try:
            return core.std.StackVertical([cclip_alt, cclip])
        except vs.Error:
            if cclip.width > cclip_alt.width:
                cclip_alt = core.std.AddBorders(cclip_alt, right=cclip.width - cclip_alt.width)
            else:
                cclip = core.std.AddBorders(cclip_alt, right=cclip_alt.width - cclip.width)
            return core.std.StackVertical([cclip_alt, cclip])
