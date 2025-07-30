"""Noising/denoising functions"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import cache, cached_property, partial, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Self,
    Sequence,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from jetpytools import KwargsNotNone, P, R, Singleton, T
from vsdenoise import (
    DFTTest,
    MVTools,
    SLocationT,
    bm3d,
    ccd,
    dpir,
    mc_degrain,
    nl_means,
    prefilter_to_full_range,
    wnnm,
)
from vsdenoise import mc_clamp as vsdenoise_mc_clamp
from vsmasktools import FDoGTCanny, adg_mask, range_mask
from vstools import (
    ColorRange,
    DitherType,
    PlanesT,
    VSFunctionKwArgs,
    check_variable_format,
    core,
    depth,
    get_depth,
    get_plane_sizes,
    get_y,
    join,
    normalize_planes,
    split,
    vs,
)

from .util import pick_px_op

__all__ = [
    "based_denoise",
    "BasedDenoise",
    "Grainer",
    "AddGrain",
    "F3kdbGrain",
    "Graigasm",
    "BilateralMethod",
    "decsiz",
    "adaptative_regrain",
]

VideoNodeT_contra = TypeVar("VideoNodeT_contra", bound=vs.VideoNode, contravariant=True)
VideoNodeT_co = TypeVar("VideoNodeT_co", bound=vs.VideoNode, covariant=True)


class _MCDegrainFunc(Protocol[VideoNodeT_contra, VideoNodeT_co]):
    def __call__(self, clip: VideoNodeT_contra, *args: Any, **kwargs: Any) -> tuple[VideoNodeT_co, MVTools]: ...


class FilterBase(Generic[P, R], Singleton):
    setting_name: ClassVar[str]
    __is_selected__: bool = True

    _filter_func: Callable[..., Any]

    def __init__(self, obj: BasedDenoise[P, R], filter_func: VSFunctionKwArgs[vs.VideoNode, vs.VideoNode]) -> None:
        self._bd = obj
        self._filter_func = filter_func

    def __call__(self, **kwargs: Any) -> BasedDenoise[P, R]:
        self._bd.settings[self.setting_name].update(KwargsNotNone(kwargs))
        return self._bd

    def apply_filter(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return self._filter_func(clip, **self._bd.settings[self.setting_name] | KwargsNotNone(kwargs))

    def select(self) -> Self:
        self.__is_selected__ = True
        return self

    def unselect(self) -> BasedDenoise[P, R]:
        self.__is_selected__ = False
        return self._bd

    def is_selected(self) -> bool:
        return self.__is_selected__

    @property
    def denoise(self) -> BasedDenoise[P, R]:
        return self._bd

    @property
    def frange(self) -> Filter.FullRange[P, R]:
        return self._bd.frange

    @property
    def dfttest(self) -> Filter.DFTTest[P, R]:
        return self._bd.dfttest

    @property
    def mc(self) -> Filter.MC[P, R]:
        return self._bd.mc

    @property
    def bm3d(self) -> Filter.BM3D[P, R]:
        return self._bd.bm3d

    @property
    def nl_means(self) -> Filter.NLMeans[P, R]:
        return self._bd.nl_means

    @property
    def wnnm(self) -> Filter.WNNM[P, R]:
        return self._bd.wnnm

    @property
    def dpir(self) -> Filter.DPIR[P, R]:
        return self._bd.dpir

    @property
    def ccd(self) -> Filter.CCD[P, R]:
        return self._bd.ccd


class FilterChroma(FilterBase[P, R]):
    __is_selected_chroma__: bool = False


class Filter:
    class FullRange(FilterBase[P, R]):
        setting_name = "frange"

        if TYPE_CHECKING:

            def __call__(
                self, *, slope: float | None = None, smooth: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class DFTTest(FilterBase[P, R]):
        setting_name = "dfttest"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, sloc: SLocationT | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class MC(FilterBase[P, R]):
        setting_name = "mc"

        mv: MVTools

        def __init__(self, obj: BasedDenoise[P, R], filter_func: _MCDegrainFunc[vs.VideoNode, vs.VideoNode]) -> None:
            self._bd = obj
            self._filter_func = filter_func

        def apply_filter(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
            clip, mv = self._filter_func(clip, **self._bd.settings[self.setting_name] | KwargsNotNone(kwargs))
            self.mv = mv
            return clip

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, thsad: int | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class BM3D(FilterBase[P, R]):
        setting_name = "bm3d"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, sigma: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class NLMeans(FilterChroma[P, R]):
        setting_name = "nl_means"
        __is_selected_chroma__ = True

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, h: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class WNNM(FilterChroma[P, R]):
        setting_name = "wnnm"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, sigma: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class DPIR(FilterChroma[P, R]):
        setting_name = "dpir"

        if TYPE_CHECKING:

            def DEBLOCK(
                self, *, strength: SupportsFloat | vs.VideoNode | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

            def DENOISE(
                self, *, strength: SupportsFloat | vs.VideoNode | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

            def __call__(
                self, *, strength: SupportsFloat | vs.VideoNode | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

        def __getattribute__(self, name: str) -> Any:
            if name == "DEBLOCK":
                self._filter_func = lambda clip, **kwargs: dpir.DEBLOCK(clip, **kwargs)
                return self.__call__

            if name == "DENOISE":
                self._filter_func = lambda clip, **kwargs: dpir.DENOISE(clip, **kwargs)
                return self.__call__

            return super().__getattribute__(name)

    class CCD(FilterChroma[P, R]):
        setting_name = "ccd"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, thr: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...


class BasedDenoise(Generic[P, R]):
    def __init__(self, func: Callable[P, R]) -> None:
        self.func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        res = self.func(*args, **kwargs)

        for v in self._settings_internal.values():
            v.clear()
        del self._settings_internal

        return res

    denoise = __call__

    @property
    def settings(self) -> Mapping[str, dict[str, Any]]:
        if not hasattr(self, "_settings_internal"):
            self._settings_internal: Mapping[str, dict[str, Any]] = {
                "frange": {},
                "dfttest": {"tr": 1, "sloc": [(0.0, 12.0), (0.35, 8.0), (1.0, 4.0)]},
                "mc": {"tr": 1, "thsad": 80},
                "bm3d": {"tr": 1, "sigma": 0.75},
                "nl_means": {"tr": 1, "h": 0.25, "planes": [1, 2]},
                "wnnm": {"tr": 1, "planes": [1, 2]},
                "dpir": {"planes": [1, 2]},
                "ccd": {"tr": 1, "planes": [1, 2]},
            }

        return self._settings_internal

    @cached_property
    def frange(self) -> Filter.FullRange[P, R]:
        return Filter.FullRange(self, lambda clip, **kwargs: prefilter_to_full_range(clip, **kwargs))

    @cached_property
    def dfttest(self) -> Filter.DFTTest[P, R]:
        return Filter.DFTTest(self, lambda clip, **kwargs: DFTTest(**kwargs).denoise(clip))

    @cached_property
    def mc(self) -> Filter.MC[P, R]:
        return Filter.MC(self, lambda clip, **kwargs: mc_degrain(clip, export_globals=True, **kwargs))

    @cached_property
    def bm3d(self) -> Filter.BM3D[P, R]:
        return Filter.BM3D(self, lambda clip, **kwargs: bm3d(clip, **kwargs))

    @cached_property
    def chroma_denoisers(self) -> Mapping[str, FilterChroma[P, R]]:
        return {
            "nl_means": Filter.NLMeans(self, lambda clip, **kwargs: nl_means(clip, **kwargs)),
            "wnnm": Filter.WNNM(self, lambda clip, **kwargs: wnnm(clip, **kwargs)),
            "dpir": Filter.DPIR(self, lambda clip, **kwargs: dpir.DEBLOCK(clip, **kwargs)),
            "ccd": Filter.CCD(self, lambda clip, **kwargs: ccd(clip, **kwargs)),
        }

    @property
    def chroma_denoiser(self) -> FilterChroma[P, R]:
        return next(cd for cd in self.chroma_denoisers.values() if cd.__is_selected_chroma__)

    @staticmethod
    def _select_chroma_denoiser(func: Callable[[BasedDenoise[P, R]], T]) -> Callable[[BasedDenoise[P, R]], T]:
        @wraps(func)
        def _wrapper(self: BasedDenoise[P, R]) -> T:
            for cn in self.chroma_denoisers.values():
                cn.__is_selected_chroma__ = False

            chroma_denoiser = func(self)

            setattr(chroma_denoiser, "__is_selected_chroma__", True)

            return chroma_denoiser

        return _wrapper

    @property
    @_select_chroma_denoiser
    @cache
    def nl_means(self) -> Filter.NLMeans[P, R]:
        return self.chroma_denoisers["nl_means"]  # type: ignore[return-value]

    @property
    @_select_chroma_denoiser
    @cache
    def wnnm(self) -> Filter.WNNM[P, R]:
        return self.chroma_denoisers["wnnm"]  # type: ignore[return-value]

    @property
    @_select_chroma_denoiser
    @cache
    def dpir(self) -> Filter.DPIR[P, R]:
        return self.chroma_denoisers["dpir"]  # type: ignore[return-value]

    @property
    @_select_chroma_denoiser
    @cache
    def ccd(self) -> Filter.CCD[P, R]:
        return self.chroma_denoisers["ccd"]  # type: ignore[return-value]


@BasedDenoise
def based_denoise(
    clip: vs.VideoNode,
    tr: int | None = None,
    sigma: float | None = None,
    h: float | None = None,
    sloc: SLocationT | None = None,
    thsad: float | None = None,
    mc_clamp: bool | None = None,
    planes: PlanesT = None,
    *,
    full_range_args: dict[str, Any] | None = None,
    dfttest_args: dict[str, Any] | None = None,
    mc_degrain_args: dict[str, Any] | None = None,
    bm3d_args: dict[str, Any] | None = None,
    nlmeans_args: dict[str, Any] | None = None,
    wnnm_args: dict[str, Any] | None = None,
    dpir_args: dict[str, Any] | None = None,
    ccd_args: dict[str, Any] | None = None,
    mc_clamp_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Most based denoise function you've ever seen.

    Default pipeline is prefilter_to_full_range -> dfttest -> mc_degrain -> bm3d -> nl_means -> mc_clamp

    Examples:
    - Basic
    ```
    denoised = based_denoise(clip, 1, 1.5, 0.25)
    ```
    - Without mvtools
    ```
    denoised = based_denoise.mc.unselect()(clip, 1, 1.5, 0.25)
    # Slightly more verbose
    denoised = based_denoise.mc.unselect().denoise(clip, 1, 1.5, 0.25)
    ```
    - Using WNNM
    ```
    denoised = based_denoise.wnnm.denoise(clip, 1, 1.5, wnnm_sigma=0.5)
    # Or
    denoised = based_denoise.wnnm(sigma=0.5).denoise(clip, 1, 1.5)
    ```

    Args:
        clip: Source clip.
        tr: Global temporal radius. Default is 1.
        sigma: BM3D sigma. Default is 0.75.
        h: h strength of nl_means. Defauts is 0.25.
        sloc: Slocation of DFTTest. Defauts is [(0.0, 12.0), (0.35, 8.0), (1.0, 4.0)].
        thsad: thsad of mc_degrain. Default is 80.
        mc_clamp: Applies mc_clamp or not. Default is True if mc_degrain is selected.
        planes: Which planes to process.
        full_range_args: Additional arguments.
        dfttest_args: Additional arguments.
        mc_degrain_args: Additional arguments.
        bm3d_args: Additional arguments.
        nlmeans_args: Additional arguments.
        wnnm_args: Additional arguments.
        dpir_args: Additional arguments.
        ccd_args: Additional arguments.
        mc_clamp_args: Additional arguments.
        **kwargs: Additional arguments prefixed by the name of the filter.

    Returns:
        Denoised clip.
    """

    assert check_variable_format(clip, based_denoise)

    bd = based_denoise

    full_range_args = {} if full_range_args is None else full_range_args.copy()
    dfttest_args = {} if dfttest_args is None else dfttest_args.copy()
    mc_degrain_args = {} if mc_degrain_args is None else mc_degrain_args.copy()
    bm3d_args = {} if bm3d_args is None else bm3d_args.copy()
    nlmeans_args = {} if nlmeans_args is None else nlmeans_args.copy()
    wnnm_args = {} if wnnm_args is None else wnnm_args.copy()
    dpir_args = {} if dpir_args is None else dpir_args.copy()
    ccd_args = {} if ccd_args is None else ccd_args.copy()
    mc_clamp_args = {} if mc_clamp_args is None else mc_clamp_args.copy()

    dict_args = [
        full_range_args,
        dfttest_args,
        mc_degrain_args,
        bm3d_args,
        nlmeans_args,
        wnnm_args,
        dpir_args,
        ccd_args,
        mc_clamp_args,
    ]
    prefixes = [
        "full_range_",
        "dfttest_",
        "mc_degrain_",
        "bm3d_",
        "nlmeans_",
        "wnnm_",
        "dpir_",
        "ccd_",
        "mc_clamp_",
    ]

    for k in kwargs.copy():
        for prefix, ckwargs in zip(prefixes, dict_args):
            if k.startswith(prefix):
                ckwargs[k.removeprefix(prefix)] = kwargs.pop(k)
                break

    mc_clamp = mc_clamp if isinstance(mc_clamp, bool) else bd.mc.is_selected()
    planes = normalize_planes(clip, planes)

    if bd.mc.is_selected():
        clip = (
            clip
            if (bd.chroma_denoiser.is_selected() and bd.chroma_denoiser.setting_name in ("nl_means", "wnnm"))
            else get_y(clip)
        )
        frange = bd.frange.apply_filter(clip, **full_range_args) if bd.frange.is_selected() else clip
        dft = bd.dfttest.apply_filter(frange, tr=tr, sloc=sloc, **dfttest_args) if bd.dfttest.is_selected() else frange
        mc = bd.mc.apply_filter(clip, prefilter=dft, tr=tr, thsad=thsad, **mc_degrain_args)
    else:
        mc = clip

    y = get_y(clip)

    if 0 in planes:
        bm3dd = bd.bm3d.apply_filter(y, tr=tr, sigma=sigma, ref=get_y(mc), **bm3d_args) if bd.bm3d.is_selected() else y
    else:
        bm3dd = y

    if clip.format.color_family is vs.GRAY:
        return vsdenoise_mc_clamp(bm3dd, y, bd.mc.mv, **mc_clamp_args) if mc_clamp else bm3dd

    if bd.chroma_denoiser.is_selected() and any(p in planes for p in [1, 2]):
        chroma_args_map = {
            "nl_means": nlmeans_args | {"h": h, "ref": mc},
            "wnnm": wnnm_args | {"ref": mc},
            "dpir": dpir_args,
            "ccd": ccd_args,
        }
        chromad = bd.chroma_denoiser.apply_filter(clip, **chroma_args_map[bd.chroma_denoiser.setting_name])
    else:
        chromad = clip

    out = join(bm3dd, chromad)

    return vsdenoise_mc_clamp(out, clip, bd.mc.mv, **mc_clamp_args) if mc_clamp else out


class Grainer(ABC):
    """Abstract graining interface"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        super().__init__()

    @abstractmethod
    def grain(self, clip: vs.VideoNode, /, strength: Tuple[float, float]) -> vs.VideoNode:
        """Graining function of the Grainer

        Args:
            clip (vs.VideoNode):
                Source clip.

            strength (Tuple[float, float]):
                First value is luma strength, second value is chroma strength.

        Returns:
            vs.VideoNode: Grained clip.
        """


class AddGrain(Grainer):
    """Built-in grain.Add plugin"""

    def grain(self, clip: vs.VideoNode, /, strength: Tuple[float, float]) -> vs.VideoNode:
        return clip.grain.Add(var=strength[0], uvar=strength[1], **self.kwargs)


class F3kdbGrain(Grainer):
    """Built-in f3kdb.Deband plugin"""

    def grain(self, clip: vs.VideoNode, /, strength: Tuple[float, float]) -> vs.VideoNode:
        return core.neo_f3kdb.Deband(clip, None, 1, 1, 1, int(strength[0]), int(strength[1]), **self.kwargs)


class Graigasm:
    """Custom graining interface based on luma values"""

    thrs: List[float]
    strengths: List[Tuple[float, float]]
    sizes: List[float]
    sharps: List[float]
    overflows: List[float]
    grainers: List[Grainer]

    def __init__(
        self,
        thrs: Sequence[float],
        strengths: Sequence[Tuple[float, float]],
        sizes: Sequence[float],
        sharps: Sequence[float],
        *,
        overflows: Union[float, Sequence[float], None] = None,
        grainers: Union[Grainer, Sequence[Grainer]] = AddGrain(seed=-1, constant=False),
    ) -> None:
        """Constructor checks and initializes the values.
           Length of thrs must be equal to strengths, sizes and sharps.
           thrs, strengths, sizes and sharps match the same area.

        Args:
            thrs (Sequence[float]):
                Sequence of thresholds defining the grain boundary.
                Below the threshold, it's grained, above the threshold, it's not grained.

            strengths (Sequence[Tuple[float, float]]):
                Sequence of tuple representing the grain strengh of the luma and the chroma, respectively.

            sizes (Sequence[float]):
                Sequence of size of grain.

            sharps (Sequence[float]):
                Sequence of sharpened grain values. 50 is neutral Catmull-Rom (b=0, c=0.5).

            overflows (Union[float, Sequence[float]], optional):
                Percentage value determining by how much the hard limit of threshold will be extended.
                Range 0.0 - 1.0. Defaults to 1 divided by thrs's length for each thr.

            grainers (Union[Grainer, Sequence[Grainer]], optional):
                Grainer used for each combo of thrs, strengths, sizes and sharps.
                Defaults to AddGrain(seed=-1, constant=False).
        """
        self.thrs = list(thrs)
        self.strengths = list(strengths)
        self.sizes = list(sizes)
        self.sharps = list(sharps)

        length = len(self.thrs)
        datas: List[Any] = [self.strengths, self.sizes, self.sharps]
        if all(len(lst) != length for lst in datas):
            raise ValueError('Graigasm: "thrs", "strengths", "sizes" and "sharps" must have the same length!')

        if overflows is None:
            overflows = [1 / length]
        if isinstance(overflows, (float, int)):
            overflows = [float(overflows)] * length
        else:
            overflows = list(overflows)
            overflows += [overflows[-1]] * (length - len(overflows))
        self.overflows = overflows

        if isinstance(grainers, Grainer):
            grainers = [grainers] * length
        else:
            grainers = list(grainers)
            grainers += [grainers[-1]] * (length - len(grainers))
        self.grainers = grainers

    def graining(
        self, clip: vs.VideoNode, /, *, prefilter: Optional[vs.VideoNode] = None, show_masks: bool = False
    ) -> vs.VideoNode:
        """Do grain stuff using settings from constructor.

        Args:
            clip (vs.VideoNode): Source clip.

            prefilter (clip, optional):
                Prefilter clip used to compute masks.
                Defaults to None.

            show_masks (bool, optional):
                Returns interleaved masks. Defaults to False.

        Returns:
            vs.VideoNode: Grained clip.
        """
        assert clip.format
        if clip.format.color_family not in (vs.YUV, vs.GRAY):
            raise ValueError("graining: Only YUV and GRAY format are supported!")

        bits = get_depth(clip)
        is_float = clip.format.sample_type == vs.FLOAT
        peak = 1.0 if is_float else (1 << bits) - 1
        num_planes = clip.format.num_planes
        neutral = [0.5] + [0.0] * (num_planes - 1) if is_float else [float(1 << (bits - 1))] * num_planes

        pref = prefilter if prefilter is not None else get_y(clip)

        mod = self._get_mod(clip)

        masks = [
            self._make_mask(pref, thr, ovf, peak, is_float=is_float) for thr, ovf in zip(self.thrs, self.overflows)
        ]
        masks = [pref.std.BlankClip(color=0)] + masks
        masks = [core.std.Expr([masks[i], masks[i - 1]], "x y -") for i in range(1, len(masks))]

        if num_planes == 3:
            if is_float:
                masks_chroma = [mask.resize.Bilinear(*get_plane_sizes(clip, 1)) for mask in masks]
                masks = [join([mask, mask_chroma, mask_chroma]) for mask, mask_chroma in zip(masks, masks_chroma)]
            else:
                masks = [join([mask] * 3).resize.Bilinear(format=clip.format.id) for mask in masks]

        if show_masks:
            return core.std.Interleave(
                [mask.text.Text(f"Threshold: {thr}", 7).text.FrameNum(9) for thr, mask in zip(self.thrs, masks)]
            )

        graineds = [
            self._make_grained(clip, strength, size, sharp, grainer, neutral, mod)
            for strength, size, sharp, grainer in zip(self.strengths, self.sizes, self.sharps, self.grainers)
        ]

        clips_adg = [
            core.std.Expr([grained, clip, mask], f"x z {peak} / * y 1 z {peak} / - * +")
            for grained, mask in zip(graineds, masks)
        ]

        out = clip
        for clip_adg in clips_adg:
            out = core.std.MergeDiff(clip_adg, core.std.MakeDiff(clip, out))  # type: ignore

        return out

    def _make_grained(
        self,
        clip: vs.VideoNode,
        strength: Tuple[float, float],
        size: float,
        sharp: float,
        grainer: Grainer,
        neutral: List[float],
        mod: int,
    ) -> vs.VideoNode:
        ss_w = self._m__(round(clip.width / size), mod)
        ss_h = self._m__(round(clip.height / size), mod)
        b = sharp / -50 + 1
        c = (1 - b) / 2

        blank = core.std.BlankClip(clip, ss_w, ss_h, color=neutral)
        grained = grainer.grain(blank, strength=strength).resize.Bicubic(
            clip.width, clip.height, filter_param_a=b, filter_param_b=c
        )

        return clip.std.MakeDiff(grained)

    @staticmethod
    def _get_mod(clip: vs.VideoNode) -> int:
        ss_mod: Dict[Tuple[int, int], int] = {(0, 0): 1, (1, 1): 2, (1, 0): 2, (0, 1): 2, (2, 2): 4, (2, 0): 4}
        assert clip.format is not None
        try:
            return ss_mod[(clip.format.subsampling_w, clip.format.subsampling_h)]
        except KeyError as kerr:
            raise ValueError("Graigasm: Format unknown!") from kerr

    @staticmethod
    def _make_mask(clip: vs.VideoNode, thr: float, overflow: float, peak: float, *, is_float: bool) -> vs.VideoNode:
        def _func(x: float) -> int:
            min_thr = thr - (overflow * peak) / 2
            max_thr = thr + (overflow * peak) / 2
            if min_thr <= x <= max_thr:
                x = abs(((x - min_thr) / (max_thr - min_thr)) * peak - peak)
            elif x < min_thr:
                x = peak
            elif x > max_thr:
                x = 0.0
            return round(x)

        min_thr = f"{thr} {overflow} {peak} * 2 / -"
        max_thr = f"{thr} {overflow} {peak} * 2 / +"
        # if x >= min_thr and x <= max_thr -> gradient else ...
        expr = f"x {min_thr} >= x {max_thr} <= and x {min_thr} - {max_thr} {min_thr} - / {peak} * {peak} - abs _ ?"
        # ... if x < min_thr -> peak else ...
        expr = expr.replace("_", f"x {min_thr} < {peak} _ ?")
        # ... if x > max_thr -> 0 else x
        expr = expr.replace("_", f"x {max_thr} > 0 x ?")

        return pick_px_op(is_float, (expr, _func))(clip)

    @staticmethod
    def _m__(x: int, mod: int, /) -> int:
        return x - x % mod


class BilateralMethod(Enum):
    BILATERAL = 0
    BILATERAL_GPU = 1
    BILATERAL_GPU_RTC = 2

    @property
    def func(self) -> Callable[..., vs.VideoNode]:
        return [  # type: ignore
            lambda: core.bilateral.Bilateral,  # type: ignore
            lambda: core.bilateralgpu.Bilateral,  # type: ignore
            lambda: core.bilateralgpu_rtc.Bilateral,  # type: ignore
        ][self.value]()  # type: ignore


def decsiz(
    clip: vs.VideoNode,
    sigmaS: float = 10.0,
    sigmaR: float = 0.009,
    min_in: Optional[float] = None,
    max_in: Optional[float] = None,
    gamma: float = 1.0,
    blur_method: BilateralMethod = BilateralMethod.BILATERAL,
    protect_mask: Optional[vs.VideoNode] = None,
    prefilter: bool = True,
    planes: Optional[List[int]] = None,
    show_mask: bool = False,
) -> vs.VideoNode:
    """Denoising function using Bilateral intended to decrease the filesize
       by just blurring the invisible grain above max_in and keeping all of it
       below min_in. The range in between is progressive.

    Args:
        clip (vs.VideoNode): Source clip.

        sigmaS (float, optional): Bilateral parameter.
            Sigma of Gaussian function to calculate spatial weight. Defaults to 10.0.

        sigmaR (float, optional): Bilateral parameter.
            Sigma of Gaussian function to calculate range weight. Defaults to 0.009.

        min_in (Union[int, float], optional):
            Minimum pixel value below which the grain is kept. Defaults to None.

        max_in (Union[int, float], optional):
            Maximum pixel value above which the grain is blurred. Defaults to None.

        gamma (float, optional):
            Controls the degree of non-linearity of the conversion. Defaults to 1.0.

        protect_mask (vs.VideoNode, optional):
            Mask that includes all the details that should not be blurred.
            If None, it uses the default one.

        prefilter (bool, optional):
            Blurs the luma as reference or not. Defaults to True.

        planes (List[int], optional): Defaults to all planes.

        show_mask (bool, optional): Returns the mask.

    Returns:
        vs.VideoNode: Denoised clip.

    Example:
        import vardefunc as vdf

        clip = depth(clip, 16)
        clip = vdf.decsiz(clip, min_in=128<<8, max_in=200<<8)
    """
    if clip.format is None:
        raise ValueError("decsiz: Variable format not allowed!")

    bits = clip.format.bits_per_sample
    is_float = clip.format.sample_type == vs.FLOAT
    peak = (1 << bits) - 1 if not is_float else 1.0
    gamma = 1 / gamma
    if clip.format.color_family == vs.GRAY:
        planes = [0]
    else:
        planes = [0, 1, 2] if not planes else planes

    if not protect_mask:
        clip16 = depth(clip, 16)
        masks = split(range_mask(clip16, rad=3, radc=2).resize.Bilinear(format=vs.YUV444P16)) + [
            FDoGTCanny().edgemask(get_y(clip16)).std.Maximum().std.Minimum()
        ]
        protect_mask = core.std.Expr(masks, "x y max z max 3250 < 0 65535 ? a max 8192 < 0 65535 ?").std.BoxBlur(
            hradius=1, vradius=1, hpasses=2, vpasses=2
        )

    clip_y = get_y(clip)
    if prefilter:
        pre = clip_y.std.BoxBlur(hradius=2, vradius=2, hpasses=4, vpasses=4)
    else:
        pre = clip_y

    denoise_mask = pick_px_op(
        is_float,
        (
            f"x {min_in} max {max_in} min {min_in} - {max_in} {min_in} - / {gamma} pow 0 max 1 min {peak} *",
            lambda x: round(
                min(1, max(0, pow((min(max_in, max(min_in, x)) - min_in) / (max_in - min_in), gamma))) * peak
            ),
        ),  # type: ignore
    )(pre)

    mask = core.std.Expr(
        [
            depth(protect_mask, bits, range_out=ColorRange.FULL, range_in=ColorRange.FULL, dither_type=DitherType.NONE),
            denoise_mask,
        ],
        "y x -",
    )

    if show_mask:
        return mask

    if blur_method == BilateralMethod.BILATERAL:
        denoise = core.bilateral.Bilateral(clip, sigmaS=sigmaS, sigmaR=sigmaR, planes=planes, algorithm=0)
    else:
        denoise = blur_method.func(clip, sigmaS, sigmaR)

    return core.std.MaskedMerge(clip, denoise, mask, planes)


def adaptative_regrain(
    denoised: vs.VideoNode,
    new_grained: vs.VideoNode,
    original_grained: vs.VideoNode,
    range_avg: Tuple[float, float] = (0.5, 0.4),
    luma_scaling: int = 28,
) -> vs.VideoNode:
    """Merge back the original grain below the lower range_avg value,
       apply the new grain clip above the higher range_avg value
       and weight both of them between the range_avg values for a smooth merge.
       Intended for use in applying a static grain in higher PlaneStatsAverage values
       to decrease the file size since we can't see a dynamic grain on that level.
       However, in dark scenes, it's more noticeable so we apply the original grain.

    Args:
        denoised (vs.VideoNode): The denoised clip.
        new_grained (vs.VideoNode): The new regrained clip.
        original_grained (vs.VideoNode): The original regrained clip.
        range_avg (Tuple[float, float], optional): Range used in PlaneStatsAverage. Defaults to (0.5, 0.4).
        luma_scaling (int, optional): Parameter in adg.Mask. Defaults to 28.

    Returns:
        vs.VideoNode: The new adaptative grained clip.

    Example:
        import vardefunc as vdf

        denoise = denoise_filter(src, ...)
        diff = core.std.MakeDiff(src, denoise)
        ...
        some filters
        ...
        new_grained = core.neo_f3kdb.Deband(last, preset='depth', grainy=32, grainc=32)
        original_grained = core.std.MergeDiff(last, diff)
        adapt_regrain = vdf.adaptative_regrain(last, new_grained, original_grained, range_avg=(0.5, 0.4), luma_scaling=28)
    """

    avg = core.std.PlaneStats(denoised)
    adapt_mask = adg_mask(get_y(avg), luma_scaling)
    adapt_grained = core.std.MaskedMerge(new_grained, original_grained, adapt_mask)

    avg_max = max(range_avg)
    avg_min = min(range_avg)

    def _diff(
        n: int,
        f: vs.VideoFrame,
        avg_max: float,
        avg_min: float,  # noqa: PLW0613
        new: vs.VideoNode,
        adapt: vs.VideoNode,
    ) -> vs.VideoNode:
        psa = cast(float, f.props["PlaneStatsAverage"])
        if psa > avg_max:
            clip = new
        elif psa < avg_min:
            clip = adapt
        else:
            weight = (psa - avg_min) / (avg_max - avg_min)
            clip = core.std.Merge(adapt, new, [weight])
        return clip

    diff_function = partial(_diff, avg_max=avg_max, avg_min=avg_min, new=new_grained, adapt=adapt_grained)

    return core.std.FrameEval(denoised, diff_function, [avg])
