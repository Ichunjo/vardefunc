"""Noising/denoising functions"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Self, SupportsFloat, cast

from jetpytools import KwargsNotNone, Singleton, cachedproperty
from vsdenoise import (
    DFTTest,
    MVTools,
    SLocationLike,
    bm3d,
    ccd,
    dpir,
    mc_degrain,
    nl_means,
    prefilter_to_full_range,
    wnnm,
)
from vsdenoise import mc_clamp as vsdenoise_mc_clamp
from vsmasktools import adg_mask
from vstools import InvalidColorFamilyError, Planes, VSFunctionKwArgs, VSObject, core, get_y, join, normalize_planes, vs

__all__ = ["BasedDenoise", "adaptative_regrain", "based_denoise"]


class FilterBase[**P, R](Singleton):
    setting_name: ClassVar[str]
    __is_selected__: bool = True

    _filter_func: Callable[..., Any]

    def __init__(self, obj: BasedDenoise[P, R], filter_func: VSFunctionKwArgs) -> None:
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


class FilterChroma[**P, R](FilterBase[P, R]):
    __is_selected_chroma__: bool = False


class Filter:
    class FullRange[**P, R](FilterBase[P, R]):
        setting_name = "frange"

        if TYPE_CHECKING:

            def __call__(
                self, *, slope: float | None = None, smooth: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class DFTTest[**P, R](FilterBase[P, R]):
        setting_name = "dfttest"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, sloc: SLocationLike | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class MC[**P, R](FilterBase[P, R]):
        setting_name = "mc"

        mv: MVTools

        def apply_filter(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
            clip, mv = self._filter_func(
                clip, export_globals=True, **self._bd.settings[self.setting_name] | KwargsNotNone(kwargs)
            )
            self.mv = mv
            return clip

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, thsad: int | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class BM3D[**P, R](FilterBase[P, R]):
        setting_name = "bm3d"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, sigma: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class NLMeans[**P, R](FilterChroma[P, R]):
        setting_name = "nl_means"
        __is_selected_chroma__ = True

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, h: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class WNNM[**P, R](FilterChroma[P, R]):
        setting_name = "wnnm"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, sigma: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

    class DPIR[**P, R](FilterChroma[P, R]):
        setting_name = "dpir"

        if TYPE_CHECKING:

            def DEBLOCK(  # noqa: N802
                self, *, strength: SupportsFloat | vs.VideoNode | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...

            def DENOISE(  # noqa: N802
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

    class CCD[**P, R](FilterChroma[P, R]):
        setting_name = "ccd"

        if TYPE_CHECKING:

            def __call__(
                self, *, tr: int | None = None, thr: float | None = None, **kwargs: Any
            ) -> BasedDenoise[P, R]: ...


class BasedDenoise[**P, R](VSObject):
    def __init__(self, func: Callable[P, R]) -> None:
        self.func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        res = self.func(*args, **kwargs)

        # Reset settings
        for v in self.settings.values():
            v.clear()
        del self.settings

        # Reset selected stages
        self.frange.select()
        self.dfttest.select()
        self.mc.select()
        self.bm3d.select()
        self.nl_means

        return res

    denoise = __call__

    @cachedproperty
    def settings(self) -> Mapping[str, dict[str, Any]]:
        return {
            "frange": {},
            "dfttest": {"tr": 1, "sloc": [(0.0, 12.0), (0.35, 8.0), (1.0, 4.0)]},
            "mc": {"tr": 1, "thsad": 80},
            "bm3d": {"tr": 1, "sigma": 0.75},
            "nl_means": {"tr": 1, "h": 0.25, "planes": [1, 2]},
            "wnnm": {"tr": 1, "planes": [1, 2]},
            "dpir": {"planes": [1, 2]},
            "ccd": {"tr": 1, "planes": [1, 2]},
        }

    @settings.deleter  # type: ignore[no-redef]
    def settings(self) -> None:
        cachedproperty.clear_cache(self, "settings")

    @property
    def frange(self) -> Filter.FullRange[P, R]:
        return Filter.FullRange(self, lambda clip, **kwargs: prefilter_to_full_range(clip, **kwargs))

    @property
    def dfttest(self) -> Filter.DFTTest[P, R]:
        return Filter.DFTTest(self, lambda clip, **kwargs: DFTTest(**kwargs).denoise(clip))

    @property
    def mc(self) -> Filter.MC[P, R]:
        return Filter.MC(self, lambda clip, **kwargs: mc_degrain(clip, **kwargs))

    @property
    def bm3d(self) -> Filter.BM3D[P, R]:
        return Filter.BM3D(self, lambda clip, **kwargs: bm3d(clip, **kwargs))

    @cachedproperty
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
    def _select_chroma_denoiser[T](func: Callable[[BasedDenoise[P, R]], T]) -> Callable[[BasedDenoise[P, R]], T]:
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
    def nl_means(self) -> Filter.NLMeans[P, R]:
        return self.chroma_denoisers["nl_means"]  # type: ignore[return-value]

    @property
    @_select_chroma_denoiser
    def wnnm(self) -> Filter.WNNM[P, R]:
        return self.chroma_denoisers["wnnm"]  # type: ignore[return-value]

    @property
    @_select_chroma_denoiser
    def dpir(self) -> Filter.DPIR[P, R]:
        return self.chroma_denoisers["dpir"]  # type: ignore[return-value]

    @property
    @_select_chroma_denoiser
    def ccd(self) -> Filter.CCD[P, R]:
        return self.chroma_denoisers["ccd"]  # type: ignore[return-value]


@BasedDenoise
def based_denoise(
    clip: vs.VideoNode,
    tr: int | None = None,
    sigma: float | None = None,
    h: float | None = None,
    sloc: SLocationLike | None = None,
    thsad: float | None = None,
    mc_clamp: bool | None = None,
    planes: Planes = None,
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
    denoised = based_denoise.wnnm()(clip, 1, 1.5, wnnm_sigma=0.5)
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
    bd = based_denoise

    InvalidColorFamilyError.check(clip, (vs.YUV, vs.GRAY), bd)

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
        wclip = (
            clip
            if bd.chroma_denoiser.is_selected()
            and bd.chroma_denoiser.setting_name in ("nl_means", "wnnm")
            and any(p in planes for p in [1, 2])
            else get_y(clip)
        )
        frange = bd.frange.apply_filter(wclip, **full_range_args) if bd.frange.is_selected() else wclip
        dft = bd.dfttest.apply_filter(frange, tr=tr, sloc=sloc, **dfttest_args) if bd.dfttest.is_selected() else frange
        mc = bd.mc.apply_filter(wclip, prefilter=dft, tr=tr, thsad=thsad, **mc_degrain_args)
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


def adaptative_regrain(
    denoised: vs.VideoNode,
    new_grained: vs.VideoNode,
    original_grained: vs.VideoNode,
    range_avg: tuple[float, float] = (0.5, 0.4),
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
        range_avg (tuple[float, float], optional): Range used in PlaneStatsAverage. Defaults to (0.5, 0.4).
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
        adapt_regrain = vdf.adaptative_regrain(
            last, new_grained, original_grained, range_avg=(0.5, 0.4), luma_scaling=28
        )
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
        avg_min: float,
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

    return core.std.FrameEval(denoised, lambda n, f: _diff(n, f, avg_max, avg_min, new_grained, adapt_grained), avg)  # pyright: ignore[reportArgumentType]
