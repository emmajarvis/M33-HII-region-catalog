from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .peak_detection import PeakDetectionParams, repo_root
from .custom_profiles import DEFAULT_PROFILE_PATH, load_custom_profile


LINE_RATIO_COLUMNS = [
    "L_Ha_sum",
    "L_Ha_sum_dered",
    "log_L_Ha_sum",
    "log_L_Ha_sum_dered",
    "log_NII_Ha_sum_dered",
    "log_OIII_Hb_sum_dered",
    "log_SII_Ha_sum_dered",
    "BPT_class_sum_dered",
    "Z_N2_Brazzini2024",
    "Z_O3N2_Brazzini2024",
    "Z_N2S2Halpha_Brazzini2024",
    "Z_R23_Maiolino2008",
    "logU_KK04",
    "ne_SII_cm3",
    "R_gal_kpc",
]


@dataclass(frozen=True)
class InjectionConfig:
    n_sources: int = 100
    min_distance_px: float = 25.0
    placement_margin_px: int = 50
    sigma_min_px: float = 1.0
    sigma_max_px: float = 25.0
    sampling_mode: str = "catalog"
    radius_min_px: float | None = None
    radius_max_px: float | None = None
    radius_min_pc: float | None = None
    radius_max_pc: float | None = None
    pc_per_px: float | None = None
    log_lha_min: float | None = None
    log_lha_max: float | None = None
    oiii_ha_mode: str = "sample"
    log_oiii_ha_min: float = -1.5
    log_oiii_ha_max: float = 1.0
    morphologies: tuple[str, ...] = ("gaussian",)
    ring_radius_min_sigma: float = 1.0
    ring_radius_max_sigma: float = 2.2
    ring_width_min_sigma: float = 0.25
    ring_width_max_sigma: float = 0.6
    clump_min_count: int = 3
    clump_max_count: int = 6
    clump_spread_sigma: float = 1.5
    clump_sigma_min_fraction: float = 0.35
    clump_sigma_max_fraction: float = 0.75
    overlap_sep_min_sigma: float = 0.7
    overlap_sep_max_sigma: float = 1.8
    custom_profile_path: str | Path | None = DEFAULT_PROFILE_PATH
    pseudo_voigt_sigma_over_r50: float = 0.96852
    pseudo_voigt_gamma_over_r50: float = 0.14786
    pseudo_voigt_eta: float = 0.10307
    pseudo_voigt_rmax_over_r50: float = 4.0


GAUSSIAN_3SIGMA_FLUX_FRACTION = 1.0 - np.exp(-0.5 * 3.0**2)
DEFAULT_PC_PER_PX = 1.2890891928799646
_CUSTOM_PROFILE_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}


def load_template_catalog(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else repo_root() / "CATALOGS" / "flux_catalogs" / "total_flux_catalog_with_derived_and_metallicities.csv"
    catalog = pd.read_csv(path)
    required = {"F_Halpha_sum", "F_[OIII]5007_sum"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Template catalog is missing required columns: {sorted(missing)}")

    valid = catalog["F_Halpha_sum"].to_numpy(dtype=float) > 0
    valid &= np.isfinite(catalog["F_Halpha_sum"].to_numpy(dtype=float))
    if "radius_p50_px" in catalog:
        valid &= np.isfinite(catalog["radius_p50_px"].to_numpy(dtype=float))
        valid &= catalog["radius_p50_px"].to_numpy(dtype=float) > 0
    return catalog.loc[valid].reset_index(drop=True)


def _sample_sigma_px(row: pd.Series, rng: np.random.Generator, config: InjectionConfig) -> float:
    for col, factor in (("radius_p50_px", np.sqrt(2.0 * np.log(2.0))), ("radius_areaeq_px_after_carve", 2.0), ("radius_areaeq_px", 2.0)):
        if col in row and np.isfinite(row[col]) and row[col] > 0:
            return float(np.clip(row[col] / factor, config.sigma_min_px, config.sigma_max_px))
    return float(rng.uniform(config.sigma_min_px, config.sigma_max_px))


def _catalog_radius_values(catalog: pd.DataFrame, config: InjectionConfig) -> np.ndarray:
    if "radius_p50_px" in catalog:
        values = pd.to_numeric(catalog["radius_p50_px"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            return values
    return np.array([config.sigma_min_px, config.sigma_max_px], dtype=float) * np.sqrt(2.0 * np.log(2.0))


def _catalog_log_lha_values(catalog: pd.DataFrame) -> np.ndarray:
    if "log_L_Ha_sum" in catalog:
        values = pd.to_numeric(catalog["log_L_Ha_sum"], errors="coerce").to_numpy(dtype=float)
    elif "L_Ha_sum" in catalog:
        lum = pd.to_numeric(catalog["L_Ha_sum"], errors="coerce").to_numpy(dtype=float)
        values = np.log10(lum, where=(lum > 0), out=np.full_like(lum, np.nan))
    else:
        values = np.array([], dtype=float)
    return values[np.isfinite(values)]


def _flux_per_luminosity(catalog: pd.DataFrame) -> float:
    if "L_Ha_sum" not in catalog:
        return 1.0
    flux = pd.to_numeric(catalog["F_Halpha_sum"], errors="coerce").to_numpy(dtype=float)
    lum = pd.to_numeric(catalog["L_Ha_sum"], errors="coerce").to_numpy(dtype=float)
    ratio = flux / lum
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    return float(np.nanmedian(ratio)) if ratio.size else 1.0


def _pc_per_px(catalog: pd.DataFrame, config: InjectionConfig) -> float:
    if config.pc_per_px is not None and np.isfinite(config.pc_per_px) and config.pc_per_px > 0:
        return float(config.pc_per_px)

    ratios = []
    for px_col, pc_col in (
        ("radius_p50_px", "radius_p50_pc"),
        ("radius_areaeq_px_after_carve", "radius_areaeq_pc_after_carve"),
        ("radius_areaeq_px", "radius_areaeq_pc"),
    ):
        if px_col not in catalog or pc_col not in catalog:
            continue
        px = pd.to_numeric(catalog[px_col], errors="coerce").to_numpy(dtype=float)
        pc = pd.to_numeric(catalog[pc_col], errors="coerce").to_numpy(dtype=float)
        ratio = pc / px
        ratios.extend(ratio[np.isfinite(ratio) & (ratio > 0)].tolist())
    if ratios:
        return float(np.nanmedian(ratios))
    return DEFAULT_PC_PER_PX


def _default_bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    return float(np.nanpercentile(values, 5)), float(np.nanpercentile(values, 95))


def _sample_independent_properties(
    template_catalog: pd.DataFrame,
    rng: np.random.Generator,
    config: InjectionConfig,
) -> dict[str, float | str]:
    pc_per_px = _pc_per_px(template_catalog, config)
    if config.radius_min_pc is not None or config.radius_max_pc is not None:
        radius_min_pc = 0.0 if config.radius_min_pc is None else float(config.radius_min_pc)
        radius_max_pc = 50.0 if config.radius_max_pc is None else float(config.radius_max_pc)
        if not np.isfinite(radius_min_pc) or not np.isfinite(radius_max_pc) or radius_max_pc <= radius_min_pc:
            raise ValueError("Physical radius bounds must be finite and radius_max_pc > radius_min_pc.")
        radius_pc = float(rng.uniform(radius_min_pc, radius_max_pc))
        radius_px = radius_pc / pc_per_px
        sigma_px = max(radius_px / 3.0, np.finfo(float).eps)
    else:
        radius_values = _catalog_radius_values(template_catalog, config)
        default_radius_min, default_radius_max = _default_bounds(radius_values)
        radius_min = config.radius_min_px if config.radius_min_px is not None else default_radius_min
        radius_max = config.radius_max_px if config.radius_max_px is not None else default_radius_max
        if not np.isfinite(radius_min) or not np.isfinite(radius_max) or radius_max <= radius_min:
            radius_min, radius_max = config.sigma_min_px, config.sigma_max_px
        radius_px = float(rng.uniform(radius_min, radius_max))
        sigma_px = float(np.clip(radius_px / np.sqrt(2.0 * np.log(2.0)), config.sigma_min_px, config.sigma_max_px))
        radius_pc = radius_px * pc_per_px

    log_lha_values = _catalog_log_lha_values(template_catalog)
    default_log_lha_min, default_log_lha_max = _default_bounds(log_lha_values)
    log_lha_min = config.log_lha_min if config.log_lha_min is not None else default_log_lha_min
    log_lha_max = config.log_lha_max if config.log_lha_max is not None else default_log_lha_max
    if not np.isfinite(log_lha_min) or not np.isfinite(log_lha_max) or log_lha_max <= log_lha_min:
        flux_values = pd.to_numeric(template_catalog["F_Halpha_sum"], errors="coerce").to_numpy(dtype=float)
        flux_values = flux_values[np.isfinite(flux_values) & (flux_values > 0)]
        log_lha_min, log_lha_max = _default_bounds(np.log10(flux_values))
    log_lha = float(rng.uniform(log_lha_min, log_lha_max))
    lha = float(10.0**log_lha)
    ha_flux = lha * _flux_per_luminosity(template_catalog)

    log_oiii_ha = float(rng.uniform(config.log_oiii_ha_min, config.log_oiii_ha_max))
    return {
        "sampling_mode": "independent",
        "sigma_px": sigma_px,
        "injected_radius_px": radius_px,
        "injected_radius_pc": radius_pc,
        "pc_per_px": pc_per_px,
        "F_Halpha_sum_template": ha_flux,
        "F_OIII5007_sum_template": ha_flux * 10.0**log_oiii_ha,
        "L_Ha_sum": lha,
        "log_L_Ha_sum": log_lha,
        "log_oiii_ha": log_oiii_ha,
        "template_region_id": "",
    }


def _sample_log_oiii_ha(row: pd.Series, rng: np.random.Generator, config: InjectionConfig) -> float:
    if config.oiii_ha_mode == "uniform":
        return float(rng.uniform(config.log_oiii_ha_min, config.log_oiii_ha_max))

    ha = float(row.get("F_Halpha_sum", np.nan))
    oiii = float(row.get("F_[OIII]5007_sum", np.nan))
    if np.isfinite(ha) and np.isfinite(oiii) and ha > 0 and oiii > 0:
        return float(np.clip(np.log10(oiii / ha), config.log_oiii_ha_min, config.log_oiii_ha_max))
    return float(rng.uniform(config.log_oiii_ha_min, config.log_oiii_ha_max))


def _is_clear(x: float, y: float, taken: list[tuple[float, float]], min_distance_px: float) -> bool:
    if not taken:
        return True
    dx = np.array([x - tx for tx, _ in taken])
    dy = np.array([y - ty for _, ty in taken])
    return bool(np.all(dx * dx + dy * dy >= min_distance_px * min_distance_px))


def _sample_morphology(rng: np.random.Generator, config: InjectionConfig) -> str:
    aliases = {"pseudo_voight": "pseudo_voigt", "pseudo-voigt": "pseudo_voigt", "pseudo-voight": "pseudo_voigt"}
    allowed = {"gaussian", "ring", "clumpy", "overlap_pair", "custom", "pseudo_voigt"}
    morphologies = tuple(aliases.get(m.strip(), m.strip()) for m in config.morphologies if m.strip())
    unknown = sorted(set(morphologies) - allowed)
    if unknown:
        raise ValueError(f"Unknown injection morphology: {unknown}. Allowed: {sorted(allowed)}")
    if not morphologies:
        return "gaussian"
    return str(rng.choice(morphologies))


def _morphology_metadata(
    morphology: str,
    sigma_px: float,
    rng: np.random.Generator,
    config: InjectionConfig,
) -> dict[str, float | int | str]:
    if morphology == "gaussian":
        radius = 3.0 * sigma_px
        return {
            "morphology": morphology,
            "n_components": 1,
            "morphology_radius_px": radius,
            "ring_radius_px": np.nan,
            "ring_width_px": np.nan,
            "component_separation_px": 0.0,
        }
    if morphology == "ring":
        ring_radius = sigma_px * rng.uniform(config.ring_radius_min_sigma, config.ring_radius_max_sigma)
        ring_width = sigma_px * rng.uniform(config.ring_width_min_sigma, config.ring_width_max_sigma)
        return {
            "morphology": morphology,
            "n_components": 1,
            "morphology_radius_px": ring_radius + 3.0 * ring_width,
            "ring_radius_px": ring_radius,
            "ring_width_px": ring_width,
            "component_separation_px": 0.0,
        }
    if morphology == "clumpy":
        n_components = int(rng.integers(config.clump_min_count, config.clump_max_count + 1))
        return {
            "morphology": morphology,
            "n_components": n_components,
            "morphology_radius_px": max(3.0 * sigma_px, config.clump_spread_sigma * sigma_px + 3.0 * config.clump_sigma_max_fraction * sigma_px),
            "ring_radius_px": np.nan,
            "ring_width_px": np.nan,
            "component_separation_px": np.nan,
        }
    if morphology == "overlap_pair":
        separation = sigma_px * rng.uniform(config.overlap_sep_min_sigma, config.overlap_sep_max_sigma)
        return {
            "morphology": morphology,
            "n_components": 1,
            "morphology_radius_px": 3.0 * sigma_px,
            "ring_radius_px": np.nan,
            "ring_width_px": np.nan,
            "component_separation_px": separation,
        }
    if morphology == "custom":
        r_profile, _ = _get_custom_profile(config)
        r50_px = sigma_px * np.sqrt(2.0 * np.log(2.0))
        radius = float(np.nanmax(r_profile) * r50_px)
        return {
            "morphology": morphology,
            "n_components": 1,
            "morphology_radius_px": radius,
            "ring_radius_px": np.nan,
            "ring_width_px": np.nan,
            "component_separation_px": 0.0,
            "custom_profile_rmax_over_r50": float(np.nanmax(r_profile)),
        }
    if morphology == "pseudo_voigt":
        r50_px = sigma_px * np.sqrt(2.0 * np.log(2.0))
        radius = float(config.pseudo_voigt_rmax_over_r50 * r50_px)
        return {
            "morphology": morphology,
            "n_components": 1,
            "morphology_radius_px": radius,
            "ring_radius_px": np.nan,
            "ring_width_px": np.nan,
            "component_separation_px": 0.0,
            "pseudo_voigt_sigma_over_r50": float(config.pseudo_voigt_sigma_over_r50),
            "pseudo_voigt_gamma_over_r50": float(config.pseudo_voigt_gamma_over_r50),
            "pseudo_voigt_eta": float(config.pseudo_voigt_eta),
            "pseudo_voigt_rmax_over_r50": float(config.pseudo_voigt_rmax_over_r50),
        }
    raise ValueError(f"Unknown morphology: {morphology}")


def _get_custom_profile(config: InjectionConfig) -> tuple[np.ndarray, np.ndarray]:
    path = Path(config.custom_profile_path or DEFAULT_PROFILE_PATH)
    if not path.is_absolute():
        path = repo_root() / path
    if path not in _CUSTOM_PROFILE_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"Custom injection profile not found: {path}. "
                "Run peak_completeness/build_average_region_profile.ipynb first, or pass --custom-profile-path."
            )
        _CUSTOM_PROFILE_CACHE[path] = load_custom_profile(path)
    return _CUSTOM_PROFILE_CACHE[path]


def sample_injections(
    field: str,
    amp_map: np.ndarray,
    params: PeakDetectionParams,
    template_catalog: pd.DataFrame,
    rng: np.random.Generator,
    config: InjectionConfig,
    existing_peaks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    finite = np.isfinite(amp_map)
    y_valid, x_valid = np.where(finite)
    in_roi = (
        (x_valid >= params.xi + config.placement_margin_px)
        & (x_valid < params.xf - config.placement_margin_px)
        & (y_valid >= params.yi + config.placement_margin_px)
        & (y_valid < params.yf - config.placement_margin_px)
    )
    x_valid = x_valid[in_roi]
    y_valid = y_valid[in_roi]
    if len(x_valid) == 0:
        raise ValueError(f"No finite placement pixels found for field {field}.")

    taken: list[tuple[float, float]] = []
    if existing_peaks is not None and {"x", "y"} <= set(existing_peaks.columns):
        taken.extend(list(zip(existing_peaks["x"].astype(float), existing_peaks["y"].astype(float))))

    rows = []
    attempts = 0
    max_attempts = config.n_sources * 1000
    while len(rows) < config.n_sources and attempts < max_attempts:
        attempts += 1
        tpl = template_catalog.iloc[int(rng.integers(0, len(template_catalog)))]
        if config.sampling_mode == "independent":
            source_props = _sample_independent_properties(template_catalog, rng, config)
        elif config.sampling_mode == "catalog":
            sigma_px_catalog = _sample_sigma_px(tpl, rng, config)
            log_oiii_ha_catalog = _sample_log_oiii_ha(tpl, rng, config)
            radius_px_catalog = sigma_px_catalog * np.sqrt(2.0 * np.log(2.0))
            pc_per_px_catalog = _pc_per_px(template_catalog, config)
            source_props = {
                "sampling_mode": "catalog",
                "sigma_px": sigma_px_catalog,
                "injected_radius_px": radius_px_catalog,
                "injected_radius_pc": radius_px_catalog * pc_per_px_catalog,
                "pc_per_px": pc_per_px_catalog,
                "template_region_id": tpl.get("region_id", ""),
                "F_Halpha_sum_template": float(tpl["F_Halpha_sum"]),
                "F_OIII5007_sum_template": float(tpl.get("F_[OIII]5007_sum", np.nan)),
                "L_Ha_sum": float(tpl.get("L_Ha_sum", np.nan)),
                "log_L_Ha_sum": float(tpl.get("log_L_Ha_sum", np.nan)),
                "log_oiii_ha": log_oiii_ha_catalog,
            }
        else:
            raise ValueError("sampling_mode must be 'catalog' or 'independent'")

        sigma_px = float(source_props["sigma_px"])
        morphology = _sample_morphology(rng, config)
        morphology_meta = _morphology_metadata(morphology, sigma_px, rng, config)
        margin = max(config.placement_margin_px, int(np.ceil(float(morphology_meta["morphology_radius_px"]) + 2 * sigma_px)) + 1)

        idx = int(rng.integers(0, len(x_valid)))
        x = float(x_valid[idx])
        y = float(y_valid[idx])
        if not (params.xi + margin <= x < params.xf - margin and params.yi + margin <= y < params.yf - margin):
            continue
        clear_distance = config.min_distance_px
        if morphology == "overlap_pair" and np.isfinite(float(morphology_meta["component_separation_px"])):
            clear_distance += 0.5 * float(morphology_meta["component_separation_px"])
        if not _is_clear(x, y, taken, clear_distance):
            continue

        ha_flux = float(source_props["F_Halpha_sum_template"])
        log_oiii_ha = float(source_props["log_oiii_ha"])
        oiii_ha = 10.0**log_oiii_ha
        if morphology == "gaussian":
            normalization_sum = _gaussian_unit_sum_inside_radius(x, y, sigma_px, float(morphology_meta["morphology_radius_px"]))
            aperture_fraction = GAUSSIAN_3SIGMA_FLUX_FRACTION
        elif morphology == "custom":
            r50_px = float(source_props["injected_radius_px"])
            normalization_sum = _custom_profile_unit_sum_inside_radius(x, y, r50_px, config)
            aperture_fraction = 1.0
        elif morphology == "pseudo_voigt":
            r50_px = float(source_props["injected_radius_px"])
            normalization_sum = _pseudo_voigt_unit_sum_inside_radius(x, y, r50_px, config)
            aperture_fraction = 1.0
        else:
            normalization_sum = 2.0 * np.pi * sigma_px * sigma_px
            aperture_fraction = 1.0
        ha_peak_amp = ha_flux / normalization_sum
        oiii_peak_amp = ha_peak_amp * oiii_ha
        total_peak_amp = ha_peak_amp + oiii_peak_amp

        def build_row(row_index: int, x_row: float, y_row: float, peak_scale: float = 1.0, pair_member: int = 0, overlap_group_id: str = "") -> dict:
            return {
            "field": field,
            "injection_id": f"{field}_inj_{row_index:04d}",
            "x_inj": x_row,
            "y_inj": y_row,
            "sampling_mode": source_props["sampling_mode"],
            "sigma_px": sigma_px,
            "injected_radius_px": source_props["injected_radius_px"],
            "injected_radius_pc": source_props["injected_radius_pc"],
            "pc_per_px": source_props["pc_per_px"],
            "gaussian_3sigma_flux_fraction": aperture_fraction,
            "gaussian_3sigma_unit_sum": normalization_sum if morphology == "gaussian" else np.nan,
            "fwhm_px": 2.354820045 * sigma_px,
            "template_region_id": source_props["template_region_id"],
            "F_Halpha_sum_template": ha_flux,
            "F_OIII5007_sum_template": source_props["F_OIII5007_sum_template"],
            "L_Ha_sum": source_props["L_Ha_sum"],
            "log_L_Ha_sum": source_props["log_L_Ha_sum"],
            "log_oiii_ha": log_oiii_ha,
            "oiii_ha": oiii_ha,
            "log_ha_oiii": -log_oiii_ha,
            "ha_oiii": 1.0 / oiii_ha if oiii_ha != 0 else np.nan,
            "ha_peak_amp": ha_peak_amp * peak_scale,
            "oiii_peak_amp": oiii_peak_amp * peak_scale,
            "total_peak_amp": total_peak_amp * peak_scale,
            "surface_brightness_ha": ha_peak_amp * peak_scale,
            "overlap_group_id": overlap_group_id,
            "overlap_pair_member": pair_member,
            **morphology_meta,
            }

        if morphology == "overlap_pair" and len(rows) <= config.n_sources - 2:
            separation = float(morphology_meta["component_separation_px"])
            theta = rng.uniform(0.0, 2.0 * np.pi)
            dx = 0.5 * separation * np.cos(theta)
            dy = 0.5 * separation * np.sin(theta)
            overlap_group_id = f"{field}_overlap_{len(rows) + 1:04d}"
            peak_scales = rng.uniform(0.7, 1.3, size=2)
            new_rows = [
                build_row(len(rows) + 1, x - dx, y - dy, float(peak_scales[0]), 1, overlap_group_id),
                build_row(len(rows) + 2, x + dx, y + dy, float(peak_scales[1]), 2, overlap_group_id),
            ]
        else:
            new_rows = [build_row(len(rows) + 1, x, y)]

        for row in new_rows:
            for col in LINE_RATIO_COLUMNS:
                if col in {"L_Ha_sum", "log_L_Ha_sum"}:
                    continue
                if config.sampling_mode == "catalog" and col in tpl:
                    row[col] = tpl[col]
            rows.append(row)
            taken.append((float(row["x_inj"]), float(row["y_inj"])))
            if len(rows) >= config.n_sources:
                break

    if len(rows) != config.n_sources:
        raise RuntimeError(f"Placed {len(rows)} of {config.n_sources} requested sources for {field}.")
    return pd.DataFrame(rows)


def _add_patch(out: np.ndarray, x0: float, y0: float, radius: int, values_func) -> None:
    ny, nx = out.shape
    x_min = max(0, int(np.floor(x0)) - radius)
    x_max = min(nx, int(np.floor(x0)) + radius + 1)
    y_min = max(0, int(np.floor(y0)) - radius)
    y_max = min(ny, int(np.floor(y0)) + radius + 1)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    values = values_func(xx, yy)
    finite = np.isfinite(out[y_min:y_max, x_min:x_max])
    patch = out[y_min:y_max, x_min:x_max]
    patch[finite] += values[finite]
    out[y_min:y_max, x_min:x_max] = patch


def _gaussian_unit_sum_inside_radius(x0: float, y0: float, sigma: float, radius: float) -> float:
    patch_radius = max(1, int(np.ceil(radius)))
    x_min = int(np.floor(x0)) - patch_radius
    x_max = int(np.floor(x0)) + patch_radius + 1
    y_min = int(np.floor(y0)) - patch_radius
    y_max = int(np.floor(y0)) + patch_radius + 1
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    distance = np.hypot(xx - x0, yy - y0)
    unit_values = np.exp(-0.5 * (((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2))
    unit_sum = float(unit_values[distance <= radius].sum())
    if np.isfinite(unit_sum) and unit_sum > 0:
        return unit_sum
    return float(2.0 * np.pi * sigma * sigma * GAUSSIAN_3SIGMA_FLUX_FRACTION)


def _inject_gaussian(out: np.ndarray, x0: float, y0: float, sigma: float, peak_amp: float, nsigma: float = 5.0) -> None:
    radius = int(np.ceil(nsigma * sigma))
    _add_patch(
        out,
        x0,
        y0,
        radius,
        lambda xx, yy: peak_amp * np.exp(-0.5 * (((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2)),
    )


def _inject_ring(out: np.ndarray, x0: float, y0: float, ring_radius: float, ring_width: float, peak_amp: float) -> None:
    radius = int(np.ceil(ring_radius + 5.0 * ring_width))
    _add_patch(
        out,
        x0,
        y0,
        radius,
        lambda xx, yy: peak_amp * np.exp(-0.5 * ((np.hypot(xx - x0, yy - y0) - ring_radius) / ring_width) ** 2),
    )


def _inject_clumpy(out: np.ndarray, row, rng: np.random.Generator, config: InjectionConfig) -> None:
    sigma = float(row.sigma_px)
    n_components = int(row.n_components)
    fractions = rng.dirichlet(np.ones(n_components))
    for frac in fractions:
        radius = rng.uniform(0.0, config.clump_spread_sigma * sigma)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        x0 = float(row.x_inj) + radius * np.cos(theta)
        y0 = float(row.y_inj) + radius * np.sin(theta)
        clump_sigma = sigma * rng.uniform(config.clump_sigma_min_fraction, config.clump_sigma_max_fraction)
        _inject_gaussian(out, x0, y0, clump_sigma, float(row.total_peak_amp) * float(frac), nsigma=5.0)


def _inject_overlap_pair(out: np.ndarray, row, rng: np.random.Generator) -> None:
    sigma = float(row.sigma_px)
    separation = float(row.component_separation_px)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    dx = 0.5 * separation * np.cos(theta)
    dy = 0.5 * separation * np.sin(theta)
    frac = rng.uniform(0.35, 0.65)
    _inject_gaussian(out, float(row.x_inj) - dx, float(row.y_inj) - dy, sigma, float(row.total_peak_amp) * frac, nsigma=5.0)
    _inject_gaussian(out, float(row.x_inj) + dx, float(row.y_inj) + dy, sigma, float(row.total_peak_amp) * (1.0 - frac), nsigma=5.0)


def _custom_profile_unit_sum_inside_radius(x0: float, y0: float, r50_px: float, config: InjectionConfig) -> float:
    r_profile, y_profile = _get_custom_profile(config)
    rmax = float(np.nanmax(r_profile) * r50_px)
    patch_radius = max(1, int(np.ceil(rmax)))
    x_min = int(np.floor(x0)) - patch_radius
    x_max = int(np.floor(x0)) + patch_radius + 1
    y_min = int(np.floor(y0)) - patch_radius
    y_max = int(np.floor(y0)) + patch_radius + 1
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    rr = np.hypot(xx - x0, yy - y0) / max(r50_px, np.finfo(float).eps)
    unit_values = np.interp(rr, r_profile, y_profile, left=float(y_profile[0]), right=0.0)
    unit_sum = float(np.nansum(unit_values))
    if np.isfinite(unit_sum) and unit_sum > 0:
        return unit_sum
    return float(2.0 * np.pi * r50_px * r50_px)


def _inject_custom_profile(out: np.ndarray, x0: float, y0: float, r50_px: float, peak_amp: float, config: InjectionConfig) -> None:
    r_profile, y_profile = _get_custom_profile(config)
    radius = int(np.ceil(float(np.nanmax(r_profile) * r50_px)))
    _add_patch(
        out,
        x0,
        y0,
        radius,
        lambda xx, yy: peak_amp
        * np.interp(
            np.hypot(xx - x0, yy - y0) / max(r50_px, np.finfo(float).eps),
            r_profile,
            y_profile,
            left=float(y_profile[0]),
            right=0.0,
        ),
    )


def _pseudo_voigt_unit_values(r_over_r50: np.ndarray, config: InjectionConfig) -> np.ndarray:
    sigma = max(float(config.pseudo_voigt_sigma_over_r50), np.finfo(float).eps)
    gamma = max(float(config.pseudo_voigt_gamma_over_r50), np.finfo(float).eps)
    eta = float(np.clip(config.pseudo_voigt_eta, 0.0, 1.0))
    gaussian = np.exp(-0.5 * (r_over_r50 / sigma) ** 2)
    lorentzian = 1.0 / (1.0 + (r_over_r50 / gamma) ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def _pseudo_voigt_unit_sum_inside_radius(x0: float, y0: float, r50_px: float, config: InjectionConfig) -> float:
    rmax = float(config.pseudo_voigt_rmax_over_r50 * r50_px)
    patch_radius = max(1, int(np.ceil(rmax)))
    x_min = int(np.floor(x0)) - patch_radius
    x_max = int(np.floor(x0)) + patch_radius + 1
    y_min = int(np.floor(y0)) - patch_radius
    y_max = int(np.floor(y0)) + patch_radius + 1
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    rr = np.hypot(xx - x0, yy - y0) / max(r50_px, np.finfo(float).eps)
    unit_values = _pseudo_voigt_unit_values(rr, config)
    unit_values = np.where(rr <= float(config.pseudo_voigt_rmax_over_r50), unit_values, 0.0)
    unit_sum = float(np.nansum(unit_values))
    if np.isfinite(unit_sum) and unit_sum > 0:
        return unit_sum
    return float(2.0 * np.pi * r50_px * r50_px)


def _inject_pseudo_voigt(out: np.ndarray, x0: float, y0: float, r50_px: float, peak_amp: float, config: InjectionConfig) -> None:
    radius = int(np.ceil(float(config.pseudo_voigt_rmax_over_r50 * r50_px)))
    _add_patch(
        out,
        x0,
        y0,
        radius,
        lambda xx, yy: peak_amp
        * _pseudo_voigt_unit_values(
            np.hypot(xx - x0, yy - y0) / max(r50_px, np.finfo(float).eps),
            config,
        )
        * (np.hypot(xx - x0, yy - y0) <= float(config.pseudo_voigt_rmax_over_r50 * r50_px)),
    )


def inject_gaussians(amp_map: np.ndarray, injections: pd.DataFrame, nsigma: float = 5.0, rng: np.random.Generator | None = None, config: InjectionConfig | None = None) -> np.ndarray:
    out = np.array(amp_map, dtype=float, copy=True)
    rng = rng or np.random.default_rng()
    config = config or InjectionConfig()
    for row in injections.itertuples(index=False):
        sigma = float(row.sigma_px)
        morphology = str(getattr(row, "morphology", "gaussian"))
        if morphology == "gaussian":
            _inject_gaussian(out, float(row.x_inj), float(row.y_inj), sigma, float(row.total_peak_amp), nsigma=nsigma)
        elif morphology == "ring":
            _inject_ring(out, float(row.x_inj), float(row.y_inj), float(row.ring_radius_px), float(row.ring_width_px), float(row.total_peak_amp))
        elif morphology == "clumpy":
            _inject_clumpy(out, row, rng, config)
        elif morphology == "overlap_pair":
            _inject_gaussian(out, float(row.x_inj), float(row.y_inj), sigma, float(row.total_peak_amp), nsigma=nsigma)
        elif morphology == "custom":
            _inject_custom_profile(out, float(row.x_inj), float(row.y_inj), float(row.injected_radius_px), float(row.total_peak_amp), config)
        elif morphology == "pseudo_voigt":
            _inject_pseudo_voigt(out, float(row.x_inj), float(row.y_inj), float(row.injected_radius_px), float(row.total_peak_amp), config)
        else:
            raise ValueError(f"Unknown injection morphology: {morphology}")
    return out
