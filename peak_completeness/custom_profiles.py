from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import curve_fit

from .peak_detection import amplitude_map_path, repo_root


DEFAULT_PROFILE_DIR = repo_root() / "peak_completeness" / "average_region_profile"
DEFAULT_PROFILE_PATH = DEFAULT_PROFILE_DIR / "average_region_profile.csv"
DEFAULT_REGION_PROFILE_PATH = DEFAULT_PROFILE_DIR / "region_profile_metrics.csv"
DEFAULT_ALL_PROFILES_PATH = DEFAULT_PROFILE_DIR / "all_region_profiles.csv"
DEFAULT_FIT_PATH = DEFAULT_PROFILE_DIR / "average_profile_model_fits.csv"


@dataclass(frozen=True)
class ProfileExtractionConfig:
    fields: tuple[str, ...] = ("NW", "NE", "SE", "SW", "F5", "F6", "F7", "F8", "F9")
    n_theta: int = 72
    dr_px: float = 0.5
    r_grid_max: float = 4.0
    n_r_grid: int = 161
    min_boundary_pixels: int = 20
    smooth_sigma_bins: float = 1.0
    background_inner_fraction: float = 0.8
    background_outer_fraction: float = 1.0
    min_peak_contrast_sigma: float = 3.0


def final_peak_catalog_path(field: str) -> Path:
    return repo_root() / "CATALOGS" / f"final_peaks_{field}.csv"


def boundary_map_path(field: str) -> Path:
    return repo_root() / "Boundary_maps" / "Boundary_map_100pc" / f"Boundary_map_{field}.fits"


def _label_for_region(row: pd.Series, fallback_index: int) -> int:
    for column in ("zoi_center_label", "region_number"):
        if column in row and pd.notna(row[column]):
            return int(row[column])
    return int(fallback_index + 1)


def _robust_background(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    bg = float(np.nanmedian(values))
    sigma = float(mad_std(values - bg, ignore_nan=True)) if values.size > 1 else 0.0
    return bg, sigma


def _profile_r50(radius: np.ndarray, intensity: np.ndarray) -> float:
    y = np.clip(np.nan_to_num(intensity, nan=0.0), 0.0, None)
    if len(radius) < 2 or np.nanmax(y) <= 0:
        return np.nan
    shell = 2.0 * np.pi * radius * y
    cumulative = np.cumsum(shell)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(np.interp(0.5 * total, cumulative, radius))


def _edge_radius(mask: np.ndarray, cx: float, cy: float, theta: float, max_radius: float, dr_px: float) -> float:
    r = np.arange(0.0, max_radius + dr_px, dr_px)
    y = cy + r * np.sin(theta)
    x = cx + r * np.cos(theta)
    sampled = map_coordinates(mask.astype(float), [y, x], order=0, mode="constant", cval=0.0)
    inside = sampled > 0.5
    if not inside.any():
        return np.nan
    return float(r[np.where(inside)[0].max()])


def extract_field_region_profiles(field: str, config: ProfileExtractionConfig = ProfileExtractionConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    amp = fits.getdata(amplitude_map_path(field)).astype(float)
    labels = fits.getdata(boundary_map_path(field)).astype(float)
    peaks = pd.read_csv(final_peak_catalog_path(field), comment="#")
    r_grid = np.linspace(0.0, config.r_grid_max, config.n_r_grid)
    thetas = np.linspace(0.0, 2.0 * np.pi, config.n_theta, endpoint=False)

    profile_rows: list[dict[str, float | int | str]] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for idx, row in peaks.iterrows():
        label = _label_for_region(row, idx)
        mask = labels == label
        if np.count_nonzero(mask & np.isfinite(amp)) < config.min_boundary_pixels:
            continue

        cx = float(row["x"])
        cy = float(row["y"])
        yy, xx = np.where(mask)
        max_radius = float(np.nanmax(np.hypot(xx - cx, yy - cy)))
        if not np.isfinite(max_radius) or max_radius <= 0:
            continue

        y_grid, x_grid = np.indices(mask.shape)
        radius_grid = np.hypot(x_grid - cx, y_grid - cy)
        bg_mask = mask & (radius_grid >= config.background_inner_fraction * max_radius) & (radius_grid <= config.background_outer_fraction * max_radius)
        bg, bg_sigma = _robust_background(amp[bg_mask])
        peak = float(amp[int(round(cy)), int(round(cx))])
        contrast = peak - bg
        if not np.isfinite(contrast) or contrast <= max(config.min_peak_contrast_sigma * bg_sigma, 0.0):
            continue

        ray_profiles = []
        edge_radii = []
        for theta in thetas:
            edge_radius = _edge_radius(mask, cx, cy, theta, max_radius, config.dr_px)
            if not np.isfinite(edge_radius) or edge_radius <= config.dr_px:
                continue
            r = np.arange(0.0, edge_radius + config.dr_px, config.dr_px)
            y = cy + r * np.sin(theta)
            x = cx + r * np.cos(theta)
            prof = map_coordinates(amp, [y, x], order=1, mode="nearest")
            prof = gaussian_filter1d(prof, sigma=max(config.smooth_sigma_bins, 1e-6), mode="nearest", truncate=3.0)
            prof_norm = np.clip((prof - bg) / contrast, 0.0, None)
            r50 = _profile_r50(r, prof_norm)
            if not np.isfinite(r50) or r50 <= 0:
                continue
            ray_profiles.append(np.interp(r_grid, r / r50, prof_norm, left=prof_norm[0], right=np.nan))
            edge_radii.append(edge_radius / r50)

        if not ray_profiles:
            continue

        ray_arr = np.asarray(ray_profiles, dtype=float)
        finite_counts = np.isfinite(ray_arr).sum(axis=0)
        median_profile = np.full(ray_arr.shape[1], np.nan, dtype=float)
        for col_idx in np.where(finite_counts > 0)[0]:
            median_profile[col_idx] = float(np.nanmedian(ray_arr[:, col_idx]))
        usable = np.isfinite(median_profile)
        if usable.sum() < 5:
            continue

        metric_rows.append(
            {
                "field": field,
                "region_id": row.get("region_id", f"{field}_{idx + 1:04d}"),
                "region_number": int(row.get("region_number", idx + 1)),
                "boundary_label": label,
                "x": cx,
                "y": cy,
                "n_rays_used": int(ray_arr.shape[0]),
                "edge_radius_over_r50_median": float(np.nanmedian(edge_radii)),
                "edge_radius_over_r50_p16": float(np.nanpercentile(edge_radii, 16)),
                "edge_radius_over_r50_p84": float(np.nanpercentile(edge_radii, 84)),
                "outer_slope": float(np.nanmedian(np.gradient(median_profile, r_grid)[(r_grid >= 1.0) & (r_grid <= 2.0)])),
            }
        )
        for rr, val in zip(r_grid[usable], median_profile[usable]):
            profile_rows.append(
                {
                    "field": field,
                    "region_id": row.get("region_id", f"{field}_{idx + 1:04d}"),
                    "region_number": int(row.get("region_number", idx + 1)),
                    "r_over_r50": float(rr),
                    "normalized_intensity": float(val),
                }
            )

    return pd.DataFrame(profile_rows), pd.DataFrame(metric_rows)


def build_region_profile_catalog(config: ProfileExtractionConfig = ProfileExtractionConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_parts = []
    metric_parts = []
    for field in config.fields:
        profiles, metrics = extract_field_region_profiles(field, config)
        if not profiles.empty:
            profile_parts.append(profiles)
        if not metrics.empty:
            metric_parts.append(metrics)

    if not profile_parts:
        raise RuntimeError("No usable region profiles were extracted.")

    all_profiles = pd.concat(profile_parts, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    return all_profiles, metrics


def build_average_region_profile(config: ProfileExtractionConfig = ProfileExtractionConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_profiles, metrics = build_region_profile_catalog(config)
    average = (
        all_profiles.groupby("r_over_r50", as_index=False)
        .agg(
            median_normalized_intensity=("normalized_intensity", "median"),
            mean_normalized_intensity=("normalized_intensity", "mean"),
            p16_normalized_intensity=("normalized_intensity", lambda x: np.nanpercentile(x, 16)),
            p84_normalized_intensity=("normalized_intensity", lambda x: np.nanpercentile(x, 84)),
            n_regions=("region_id", "nunique"),
        )
        .sort_values("r_over_r50")
    )
    average["median_normalized_intensity"] = np.minimum.accumulate(average["median_normalized_intensity"].to_numpy(dtype=float))
    average["median_normalized_intensity"] = np.clip(average["median_normalized_intensity"], 0.0, 1.0)
    return average, metrics, all_profiles


def save_average_region_profile(
    average: pd.DataFrame,
    metrics: pd.DataFrame,
    all_profiles: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = DEFAULT_PROFILE_DIR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    average.to_csv(output_dir / "average_region_profile.csv", index=False)
    metrics.to_csv(output_dir / "region_profile_metrics.csv", index=False)
    if all_profiles is not None:
        all_profiles.to_csv(output_dir / "all_region_profiles.csv", index=False)


def gaussian_model(r, amp, sigma, floor):
    return floor + amp * np.exp(-0.5 * (r / sigma) ** 2)


def exponential_model(r, amp, scale, floor):
    return floor + amp * np.exp(-r / scale)


def pseudo_voigt_model(r, amp, sigma, gamma, eta, floor):
    gaussian = np.exp(-0.5 * (r / sigma) ** 2)
    lorentzian = 1.0 / (1.0 + (r / gamma) ** 2)
    return floor + amp * (eta * lorentzian + (1.0 - eta) * gaussian)


def double_gaussian_model(r, amp_narrow, sigma_narrow, amp_broad, sigma_broad, floor):
    return floor + amp_narrow * np.exp(-0.5 * (r / sigma_narrow) ** 2) + amp_broad * np.exp(-0.5 * (r / sigma_broad) ** 2)


def core_power_exp_model(r, amp, core_radius, exp_index, wing_index, floor):
    return floor + amp * np.exp(-((r / core_radius) ** exp_index)) / (1.0 + (r / core_radius) ** wing_index)


def fit_average_profile_models(average: pd.DataFrame) -> pd.DataFrame:
    r = average["r_over_r50"].to_numpy(dtype=float)
    y = average["median_normalized_intensity"].to_numpy(dtype=float)
    valid = np.isfinite(r) & np.isfinite(y) & (average["n_regions"].to_numpy(dtype=float) >= 5)
    r = r[valid]
    y = y[valid]
    models = {
        "gaussian": (gaussian_model, [1.0, 1.0, 0.0], ([0.0, 0.05, 0.0], [2.0, 10.0, 0.5])),
        "exponential": (exponential_model, [1.0, 1.0, 0.0], ([0.0, 0.05, 0.0], [2.0, 10.0, 0.5])),
        "pseudo_voigt": (pseudo_voigt_model, [1.0, 1.0, 1.0, 0.5, 0.0], ([0.0, 0.05, 0.05, 0.0, 0.0], [2.0, 10.0, 10.0, 1.0, 0.5])),
        "narrow_broad_double_gaussian": (
            double_gaussian_model,
            [0.75, 0.55, 0.15, 2.0, 0.0],
            ([0.0, 0.05, 0.15, 1.5, 0.0], [2.0, 1.0, 2.0, 15.0, 0.5]),
        ),
        "core_power_exp": (
            core_power_exp_model,
            [1.0, 1.7, 3.0, 1.0, 0.0],
            ([0.0, 0.05, 0.1, 0.0, 0.0], [2.0, 10.0, 10.0, 10.0, 0.5]),
        ),
    }
    rows = []
    for name, (func, p0, bounds) in models.items():
        try:
            popt, _ = curve_fit(func, r, y, p0=p0, bounds=bounds, maxfev=20000)
            residual = y - func(r, *popt)
            rss = float(np.sum(residual * residual))
            n = len(y)
            k = len(popt)
            aic = float(n * np.log(rss / n) + 2 * k) if rss > 0 and n > k else np.nan
            row = {"model": name, "rss": rss, "aic": aic, **{f"p{i}": float(v) for i, v in enumerate(popt)}}
            if name == "narrow_broad_double_gaussian":
                row.update(
                    {
                        "amp_narrow": float(popt[0]),
                        "sigma_narrow": float(popt[1]),
                        "amp_broad": float(popt[2]),
                        "sigma_broad": float(popt[3]),
                        "floor": float(popt[4]),
                    }
                )
            if name == "core_power_exp":
                row.update(
                    {
                        "amp": float(popt[0]),
                        "core_radius_r50": float(popt[1]),
                        "exp_index": float(popt[2]),
                        "wing_index": float(popt[3]),
                        "floor": float(popt[4]),
                    }
                )
            rows.append(row)
        except Exception as exc:
            rows.append({"model": name, "rss": np.nan, "aic": np.nan, "error": str(exc)})
    return pd.DataFrame(rows).sort_values("aic", na_position="last")


def load_custom_profile(path: str | Path = DEFAULT_PROFILE_PATH) -> tuple[np.ndarray, np.ndarray]:
    profile = pd.read_csv(path)
    if "r_over_r50" not in profile or "median_normalized_intensity" not in profile:
        raise ValueError(f"Custom profile must contain r_over_r50 and median_normalized_intensity columns: {path}")
    r = profile["r_over_r50"].to_numpy(dtype=float)
    y = profile["median_normalized_intensity"].to_numpy(dtype=float)
    valid = np.isfinite(r) & np.isfinite(y)
    r = r[valid]
    y = np.clip(y[valid], 0.0, None)
    order = np.argsort(r)
    return r[order], y[order]
