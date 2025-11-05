__all__ = ["OCR"]

import itertools
import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import vapoursynth as vs
from pytimeconv import Convert
from vsmasktools import max_planes, region_rel_mask
from vstools import clip_async_render

from .types import AnyPath

core = vs.core


class OCR:
    """OCR Interface using ocr.Recognize"""

    clip: vs.VideoNode
    coord: Tuple[int, int, int]
    coord_alt: Optional[Tuple[int, int, int]]
    thr_in: Sequence[int]
    thr_out: Sequence[int]
    thr_scd: float

    results: List[Tuple[int, bytes]]

    _brd_crop: int = 8

    def __init__(
        self,
        clip: vs.VideoNode,
        coord: Tuple[int, int, int],
        coord_alt: Optional[Tuple[int, int, int]] = None,
        thr_in: Union[int, Tuple[int, int, int]] = 225,
        thr_out: Union[int, Tuple[int, int, int]] = 80,
    ) -> None:
        """
        Args:
            clip (vs.VideoNode):
                Source clip. If GRAY clip, `thr_in` and `thr_out` should be an integer.

            coord (Tuple[int, int, int]):
                Tuple of coordinates following the syntax: width, height, margin vertical from the bottom

            coord_alt (Optional[Tuple[int, int, int]], optional):
                Tuple of alternate coordinates following the syntax: width, height, margin vertical from the top.
                Defaults to None

            thr_in (Union[int, Tuple[int, int, int]], optional):
                Threshold for subtitles representing the minimum inline brightness.
                Defaults to 225.

            thr_out (Union[int, Tuple[int, int, int]], optional):
                Threshold for subtitles representing the maximum outline brightness.
                Defaults to 80.
        """
        assert clip.format

        self.clip = clip

        self.coord = coord
        self.coord_alt = coord_alt

        self.thr_in = thr_in if isinstance(thr_in, tuple) else [thr_in]
        self.thr_out = thr_out if isinstance(thr_out, tuple) else [thr_out]

        if len({clip.format.num_planes, len(self.thr_in), len(self.thr_out)}) > 1:
            raise ValueError("OCR: number of thr_in and thr_out values must correspond to the number of clip planes!")

    def launch(
        self, datapath: Optional[str] = None, language: Optional[str] = None, options: Optional[Sequence[str]] = None
    ) -> None:
        """http://www.vapoursynth.com/doc/plugins/ocr.html

        Args:
            datapath (Optional[str], optional):
                Path to a folder containing a “tessdata” folder, in which Tesseract’s data files must be found.
                Must have a trailing slash.
                Defaults to None.

            language (Optional[str], optional):
                An ISO 639-3 language string.
                Uses Tesseract’s default language if unset (usually eng).
                Defaults to None.

            options (Optional[Sequence], optional):
                Options to be passed to Tesseract, as a list of (key, value) pairs.
                Defaults to None.
        """
        ppclip = self._cleaning(self._cropping(self.clip, self.coord, False)).resize.Point(format=vs.GRAY8)
        ocred = core.ocr.Recognize(ppclip, datapath, language, options)
        self.results = []
        self._do_ocr(ppclip, ocred)
        del ppclip, ocred

        if self.coord_alt:
            ppclip_alt = self._cleaning(self._cropping(self.clip, self.coord_alt, True)).resize.Point(format=vs.GRAY8)
            ocred_alt = core.ocr.Recognize(ppclip_alt, datapath, language, options)
            self._do_ocr(ppclip_alt, ocred_alt)
            del ppclip_alt, ocred_alt

    def _do_ocr(self, ppclip: vs.VideoNode, ocred: vs.VideoNode) -> None:
        def _select_clips(n: int, f: vs.VideoFrame, clips: List[vs.VideoNode]) -> vs.VideoNode:
            return clips[1] if f.props["PlaneStatsMax"] > 0 else clips[0].std.BlankClip(1, 1)  # type: ignore

        ocred = core.std.FrameEval(
            core.std.Splice([ppclip[:-1], ppclip.std.BlankClip(1, 1, length=1)], True),
            partial(_select_clips, clips=[ppclip, ocred]),
            prop_src=ppclip.std.PlaneStats(),
        )

        results: Set[Tuple[int, bytes]] = set()

        def _callback(n: int, f: vs.VideoFrame) -> None:
            if (prop_ocr := "OCRString") in f.props:
                results.add((n, f.props[prop_ocr]))  # type: ignore

        clip_async_render(ocred, progress="OCRing clip...", callback=_callback)
        self.results += sorted(results)

    def write_ass(
        self,
        output: AnyPath,
        string_replace: List[Tuple[str, str]] = [("_", "-"), ("…", "..."), ("‘", "'"), ("’", "'"), (" '", "'")],
    ) -> None:
        """Write results as a readable ass file.

        Args:
            output (AnyPath): Output path

            string_replace (List[Tuple[str, str]], optional):
                List of strings you want to replace.
                Defaults to [ ('_', '-'), ('…', '...'), ('‘', "'"), ('’', "'"), (" '", "'") ].
        """
        resultsd: Dict[int, Tuple[int, str]] = {}
        for frame, string_byte in sorted(self.results):
            nstring = string_byte.decode("utf-8").replace("\n", "\\N")
            for r in string_replace:
                nstring = nstring.replace(*r)
            resultsd[frame] = (frame + 1, nstring)

        results_s = sorted(resultsd.items(), reverse=True)

        for (start1, (end1, string1)), (start2, (end2, string2)) in itertools.pairwise(results_s):
            if string1 == string2 and end2 == start1:
                resultsd[start2] = (max(end1, resultsd[start1][0]), string1)
                del resultsd[start1]

        fps = self.clip.fps

        with open(output, "w", encoding="utf-8-sig") as ass:
            ass.write("[Events]\n")
            ass.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            for s, (e, string) in sorted(resultsd.items()):
                if string:
                    ass.write(
                        f"Dialogue: 0,{Convert.f2assts(s, fps)},{Convert.f2assts(e, fps)},Default,,0,0,0,,{string}\n"
                    )

    def _cropping(self, clip: vs.VideoNode, c: Tuple[int, int, int], alt: bool) -> vs.VideoNode:
        cw, ch, h = c
        wcrop = (self.clip.width - cw) / 2
        hcrop = h if alt else self.clip.height - ch - h

        return clip.std.CropAbs(cw, ch, round(wcrop), hcrop)

    def _cleaning(self, clip: vs.VideoNode) -> vs.VideoNode:
        clip_black = clip.std.BlankClip(clip.width - self._brd_crop, clip.height - self._brd_crop)
        square = core.std.AddBorders(
            clip_black,
            *(int(self._brd_crop / 2),) * 4,
            color=[(1 << clip.format.bits_per_sample) - 1] * clip_black.format.num_planes,  # type: ignore
        )

        white_raw = clip.std.Binarize(self.thr_in)
        bright_raw = clip.std.Binarize(self.thr_out)

        bright_out = core.std.Expr([bright_raw, square], "x y min")
        bright_not = core.misc.Hysteresis(bright_out, bright_raw).std.InvertMask()
        white_txt = core.std.MaskedMerge(clip.std.BlankClip(), white_raw, bright_not)

        white_txt = max_planes(white_txt)

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

        return core.std.MaskedMerge(core.std.Lut(self.clip, function=lambda x: round(x / 2)), self.clip, cmask)

    def _compute_preview_cropped(self, c: Tuple[int, int, int], alt: bool) -> vs.VideoNode:
        cw, ch, h = c
        wcrop = (self.clip.width - cw) / 2
        left, right = math.ceil(wcrop), math.floor(wcrop)
        hcrop = self.clip.height - ch - h, h
        if alt:
            hcrop = hcrop[::-1]
        return region_rel_mask(self.clip.std.BlankClip(format=vs.GRAY8, color=255), left, right, *hcrop)

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
