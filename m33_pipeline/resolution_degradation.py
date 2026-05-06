from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter
from tqdm.auto import tqdm

from . import paths
from .derived import (
    add_electron_density,
    add_logU_KK04,
    add_metallicity_columns,
    deproject_pixels_to_disk_pc,
)
from .io import read_catalog, read_fits_data, read_fits_data_header, safe_mkdir, write_catalog, write_fits
from .photometry import LINE_MAP_BASENAMES, region_edge_ring
from .resolution_regions import (
    build_boundary_map,
    build_peak_table,
    degrade_map_to_distance,
    distance_tag,
    flux_to_luminosity,
    makezoi_legacy,
    pixel_size_pc,
    plot_segmentation_example,
    scaled_roi_from_param_section,
)
from .validate import write_qc_report


FIELDS = ("NW", "NE", "SW", "SE", "F5", "F6", "F7", "F8", "F9")
DEFAULT_DISTANCES_MPC = (0.84, 5.0, 10.0, 20.0, 35.0, 50.0, 100.0)
PIXEL_SCALE_ARCSEC = 0.32
M33_DISTANCE_PC = 840000.0
M33_CENTER_SKY = SkyCoord("01h33m50.9s", "+30d39m36.8s")
M33_PA_DEG = 23.0
M33_INCLINATION_DEG = 56.0
CASE_B_HA_HB = 2.86
RV = 3.1
K_HBETA = 3.609
K_HALPHA = 2.535
BASE_ZOI_MAX_RADIUS_PC = 100.0
MIN_ZOI_RADIUS_PX = 4
ORIGINAL_MIN_PEAK_SEPARATION_PX = 6
MAIN_CATALOG_SNR_THRESHOLD = 6.0


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def progress_bar(iterable, *, total: int | None = None, desc: str = ""):
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


@dataclass(frozen=True)
class ResolutionPaths:
    root: Path
    degraded_dir: Path
    peaks_dir: Path
    regions_dir: Path
    catalogs_dir: Path
    plots_dir: Path
    qc_dir: Path


@dataclass(frozen=True)
class MainCatalogPeakParams:
    field: str
    snrlim: float
    bgbox: int
    stdbox: int
    signoi: float
    n_th_peaks: int
    parameter_file: Path


def resolution_stage_root() -> Path:
    return paths.repo_root() / "04_resolution_degradation"


def build_resolution_paths() -> ResolutionPaths:
    root = resolution_stage_root()
    return ResolutionPaths(
        root=root,
        degraded_dir=root / "degraded_fits",
        peaks_dir=root / "degraded_peaks",
        regions_dir=root / "region_products",
        catalogs_dir=root / "catalog_products",
        plots_dir=root / "plots",
        qc_dir=root / "qc",
    )


def ensure_resolution_dirs(stage_paths: ResolutionPaths | None = None) -> ResolutionPaths:
    stage_paths = stage_paths or build_resolution_paths()
    for path in (
        stage_paths.degraded_dir,
        stage_paths.peaks_dir,
        stage_paths.regions_dir,
        stage_paths.catalogs_dir,
        stage_paths.plots_dir,
        stage_paths.qc_dir,
    ):
        safe_mkdir(path)
    return stage_paths


def zoi_max_radius_pc_for_distance(distance_mpc: float) -> float:
    return BASE_ZOI_MAX_RADIUS_PC


def m33_pc_per_arcsec() -> float:
    return M33_DISTANCE_PC / 206265.0


def m33_pixel_scale_pc(pixscale_arcsec_per_pix: float) -> float:
    return float(pixscale_arcsec_per_pix) * m33_pc_per_arcsec()


def original_pixel_scale_pc() -> float:
    return m33_pixel_scale_pc(PIXEL_SCALE_ARCSEC)


def degraded_scale_factor(distance_mpc: float, d_orig_mpc: float = 0.84) -> float:
    return d_orig_mpc / np.asarray(distance_mpc, dtype=float)


def effective_peak_separation_px(distance_mpc: float) -> int:
    sep_pc = ORIGINAL_MIN_PEAK_SEPARATION_PX * original_pixel_scale_pc()
    pix_pc = m33_pixel_scale_pc(PIXEL_SCALE_ARCSEC / degraded_scale_factor(distance_mpc))
    return max(1, int(np.ceil(sep_pc / pix_pc)))


def effective_zoi_radius_for_map(distance_mpc: float, pix_pc: float) -> tuple[float, int]:
    zoi_pc = max(zoi_max_radius_pc_for_distance(distance_mpc), MIN_ZOI_RADIUS_PX * pix_pc)
    zoi_px = max(MIN_ZOI_RADIUS_PX, int(np.ceil(zoi_pc / pix_pc)))
    return float(zoi_pc), int(zoi_px)


@lru_cache(maxsize=None)
def field_m33_center_original_pixels(field: str) -> tuple[float, float]:
    _, header = read_fits_data_header(paths.calibrated_field_map_dir(field) / f"M33{field}-{LINE_MAP_BASENAMES['Halpha']}.fits")
    x_center, y_center = WCS(header).world_to_pixel(M33_CENTER_SKY)
    return float(x_center), float(y_center)


@lru_cache(maxsize=None)
def final_catalog_peak_count(field: str) -> int:
    return int(len(read_catalog(paths.final_peaks_csv(field), comment="#")))


def _parse_main_catalog_param_file(path: Path, field: str) -> MainCatalogPeakParams:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return MainCatalogPeakParams(
        field=field,
        snrlim=float(values["snrlim"]),
        bgbox=int(float(values["bgbox"])),
        stdbox=int(float(values["stdbox"])),
        signoi=float(values["signoi"]),
        n_th_peaks=int(float(values["n_TH_peaks"])),
        parameter_file=path,
    )


@lru_cache(maxsize=None)
def main_catalog_peak_params(field: str) -> MainCatalogPeakParams:
    param_dir = paths.repo_root() / "final_peak_info"
    candidates = sorted(param_dir.glob(f"parameters_field{field}_*.txt"))
    if not candidates:
        return MainCatalogPeakParams(
            field=field,
            snrlim=MAIN_CATALOG_SNR_THRESHOLD,
            bgbox=25,
            stdbox=3,
            signoi=3.0,
            n_th_peaks=final_catalog_peak_count(field),
            parameter_file=Path(""),
        )
    target = final_catalog_peak_count(field)
    parsed = [_parse_main_catalog_param_file(path, field) for path in candidates]
    return min(parsed, key=lambda item: abs(item.n_th_peaks - target))


def detect_peaks_in_map_with_threshold(
    flux_map: np.ndarray,
    err_map: np.ndarray,
    *,
    field: str,
    distance_mpc: float,
    min_separation_px: int,
    snr_threshold: float,
    flux_threshold: float,
    max_peaks: int | None = None,
) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err_map > 0, flux_map / err_map, np.nan)
    local_max = maximum_filter(np.nan_to_num(flux_map, nan=-np.inf), size=2 * min_separation_px + 1, mode="nearest")
    candidate_mask = (
        np.isfinite(flux_map)
        & np.isfinite(snr)
        & (flux_map == local_max)
        & (snr >= snr_threshold)
        & (flux_map >= flux_threshold)
    )
    y_idx, x_idx = np.where(candidate_mask)
    if len(x_idx) == 0:
        return pd.DataFrame(columns=["field", "distance_mpc", "x", "y", "peak_flux", "peak_snr"])

    order = np.argsort(np.nan_to_num(flux_map[y_idx, x_idx], nan=-np.inf))[::-1]
    selected: list[tuple[int, int]] = []
    for idx in order:
        x = int(x_idx[idx])
        y = int(y_idx[idx])
        if any((x - sx) ** 2 + (y - sy) ** 2 < min_separation_px**2 for sx, sy in selected):
            continue
        selected.append((x, y))
        if max_peaks is not None and len(selected) >= max_peaks:
            break

    return pd.DataFrame(
        [
            {
                "field": field,
                "distance_mpc": distance_mpc,
                "x": float(x),
                "y": float(y),
                "peak_flux": float(flux_map[y, x]),
                "peak_snr": float(snr[y, x]),
                "peak_rank": rank,
            }
            for rank, (x, y) in enumerate(selected, start=1)
        ]
    )


@lru_cache(maxsize=None)
def calibrated_flux_percentile_threshold(field: str) -> float:
    flux_map = read_fits_data(degraded_flux_path(field, "Halpha", 0.84))
    err_map = read_fits_data(degraded_err_path(field, "Halpha", 0.84))
    params = main_catalog_peak_params(field)
    target = final_catalog_peak_count(field)
    min_sep = effective_peak_separation_px(0.84)
    low, high = 50.0, 99.95
    best_percentile = 90.0
    best_diff = float("inf")
    for percentile in np.linspace(low, high, 40):
        finite_flux = flux_map[np.isfinite(flux_map)]
        flux_cut = np.nanpercentile(finite_flux, percentile)
        peaks = detect_peaks_in_map_with_threshold(
            flux_map,
            err_map,
            field=field,
            distance_mpc=0.84,
            min_separation_px=min_sep,
            snr_threshold=params.snrlim,
            flux_threshold=flux_cut,
        )
        diff = abs(len(peaks) - target)
        if diff < best_diff:
            best_diff = diff
            best_percentile = float(percentile)
    return best_percentile


def line_key_to_column_name(line_name: str) -> str:
    if line_name == "Halpha":
        return "Halpha"
    if line_name == "Hbeta":
        return "Hbeta"
    if line_name == "[OIII]5007":
        return "[OIII]5007"
    if line_name == "[SII]6716":
        return "[SII]6716"
    if line_name == "[SII]6731":
        return "[SII]6731"
    if line_name == "[NII]6583":
        return "[NII]6583"
    if line_name == "[OII]3727":
        return "[OII]3727"
    raise KeyError(f"Unsupported line name: {line_name}")


def field_distance_tag(field: str, distance_mpc: float) -> str:
    return f"{field}_{distance_tag(distance_mpc)}Mpc"


def degraded_flux_path(field: str, line_name: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.degraded_dir / field / f"{field_distance_tag(field, distance_mpc)}_{LINE_MAP_BASENAMES[line_name]}_flux.fits"


def degraded_err_path(field: str, line_name: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.degraded_dir / field / f"{field_distance_tag(field, distance_mpc)}_{LINE_MAP_BASENAMES[line_name]}_err.fits"


def peaks_catalog_path(field: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.peaks_dir / field / f"peaks_{field_distance_tag(field, distance_mpc)}.csv"


def zoi_map_path(field: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.regions_dir / field / f"zoi_map_{field_distance_tag(field, distance_mpc)}.fits"


def boundary_map_path(field: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.regions_dir / field / f"boundary_map_{field_distance_tag(field, distance_mpc)}.fits"


def boundary_metrics_path(field: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.regions_dir / field / f"boundary_metrics_{field_distance_tag(field, distance_mpc)}.csv"


def field_catalog_path(field: str, distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.catalogs_dir / field / f"resolution_catalog_{field_distance_tag(field, distance_mpc)}.csv"


def total_catalog_path(distance_mpc: float, stage_paths: ResolutionPaths | None = None) -> Path:
    stage_paths = stage_paths or build_resolution_paths()
    return stage_paths.catalogs_dir / f"resolution_catalog_all_fields_{distance_tag(distance_mpc)}Mpc.csv"


def read_calibrated_line_map(field: str, line_name: str) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    field_dir = paths.calibrated_field_map_dir(field)
    base = LINE_MAP_BASENAMES[line_name]
    flux, header = read_fits_data_header(field_dir / f"M33{field}-{base}.fits")
    err = read_fits_data(field_dir / f"M33{field}-{base}-err.fits")
    return flux, err, header


def iter_degraded_map_records(stage_paths: ResolutionPaths | None = None) -> list[dict[str, object]]:
    stage_paths = stage_paths or build_resolution_paths()
    records: list[dict[str, object]] = []
    for field_dir in sorted(stage_paths.degraded_dir.glob("*")):
        if not field_dir.is_dir():
            continue
        for flux_path in sorted(field_dir.glob("*_flux.fits")):
            records.append({"field": field_dir.name, "path": flux_path})
    return records


def degrade_all_fields(
    fields: tuple[str, ...] = FIELDS,
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC,
    fwhm_psf_orig: float = 0.8,
    stage_paths: ResolutionPaths | None = None,
) -> pd.DataFrame:
    stage_paths = ensure_resolution_dirs(stage_paths)
    records: list[dict[str, object]] = []
    stage_start = time.perf_counter()
    total_steps = len(fields) * len(distances_mpc)

    with tqdm(total=total_steps, desc="Degrade", dynamic_ncols=True) as pbar:
        for field in fields:
            field_start = time.perf_counter()
            safe_mkdir(stage_paths.degraded_dir / field)
            for distance_mpc in distances_mpc:
                distance_start = time.perf_counter()
                for line_name in LINE_MAP_BASENAMES:
                    flux, err, header = read_calibrated_line_map(field, line_name)
                    degraded_flux, pixscale_new, spatial_resolution_pc = degrade_map_to_distance(
                        image=flux,
                        d_orig=0.84,
                        d_target=distance_mpc,
                        pixscale_orig=pixel_scale_arcsec,
                        fwhm_psf_orig=fwhm_psf_orig,
                        plot=False,
                        overlay_peaks=False,
                    )
                    degraded_err, _, _ = degrade_map_to_distance(
                        image=err,
                        d_orig=0.84,
                        d_target=distance_mpc,
                        pixscale_orig=pixel_scale_arcsec,
                        fwhm_psf_orig=fwhm_psf_orig,
                        plot=False,
                        overlay_peaks=False,
                    )
                    write_fits(degraded_flux_path(field, line_name, distance_mpc, stage_paths), degraded_flux, header=header)
                    write_fits(degraded_err_path(field, line_name, distance_mpc, stage_paths), degraded_err, header=header)
                    records.append(
                        {
                            "field": field,
                            "distance_mpc": distance_mpc,
                            "line_name": line_name,
                            "pixscale_arcsec_per_pix": pixscale_new,
                            "spatial_resolution_pc": spatial_resolution_pc,
                            "flux_path": str(degraded_flux_path(field, line_name, distance_mpc, stage_paths)),
                            "err_path": str(degraded_err_path(field, line_name, distance_mpc, stage_paths)),
                        }
                    )
                pbar.update(1)
                tqdm.write(
                    f"[degrade] {field} {distance_mpc:g} Mpc done in "
                    f"{format_elapsed(time.perf_counter() - distance_start)}"
                )
            tqdm.write(f"[degrade] field {field} done in {format_elapsed(time.perf_counter() - field_start)}")
    tqdm.write(f"[degrade] all fields done in {format_elapsed(time.perf_counter() - stage_start)}")

    summary = pd.DataFrame(records)
    write_catalog(summary, stage_paths.catalogs_dir / "degraded_map_inventory.csv")
    write_qc_report(
        stage_paths.qc_dir / "01_degrade_maps.json",
        stage="degrade_maps",
        field=None,
        warnings=[],
        summary={"n_products": int(len(summary)), "fields": list(fields), "distances_mpc": list(distances_mpc)},
    )
    return summary


def detect_peaks_in_map(
    flux_map: np.ndarray,
    err_map: np.ndarray,
    *,
    field: str,
    distance_mpc: float,
    min_separation_px: int = 6,
    snr_threshold: float = 5.0,
    flux_percentile_threshold: float = 90.0,
    max_peaks: int | None = None,
) -> pd.DataFrame:
    finite_flux = flux_map[np.isfinite(flux_map)]
    if finite_flux.size == 0:
        return pd.DataFrame(columns=["field", "distance_mpc", "x", "y", "peak_flux", "peak_snr"])
    flux_cut = np.nanpercentile(finite_flux, flux_percentile_threshold)
    return detect_peaks_in_map_with_threshold(
        flux_map,
        err_map,
        field=field,
        distance_mpc=distance_mpc,
        min_separation_px=min_separation_px,
        snr_threshold=snr_threshold,
        flux_threshold=flux_cut,
        max_peaks=max_peaks,
    )


def identify_degraded_peaks(
    fields: tuple[str, ...] = FIELDS,
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    stage_paths: ResolutionPaths | None = None,
    min_separation_px: int = 6,
    snr_threshold: float = MAIN_CATALOG_SNR_THRESHOLD,
    flux_percentile_threshold: float = 90.0,
) -> pd.DataFrame:
    stage_paths = ensure_resolution_dirs(stage_paths)
    all_rows: list[pd.DataFrame] = []
    stage_start = time.perf_counter()
    total_steps = len(fields) * len(distances_mpc)

    with tqdm(total=total_steps, desc="Peaks", dynamic_ncols=True) as pbar:
        for field in fields:
            field_start = time.perf_counter()
            safe_mkdir(stage_paths.peaks_dir / field)
            main_params = main_catalog_peak_params(field)
            calibrated_percentile = calibrated_flux_percentile_threshold(field)
            for distance_mpc in distances_mpc:
                distance_start = time.perf_counter()
                if np.isclose(distance_mpc, 0.84):
                    peaks_df = read_catalog(paths.final_peaks_csv(field), comment="#").copy()
                    peaks_df = peaks_df.rename(columns={"x": "x", "y": "y"})
                    peaks_df["distance_mpc"] = distance_mpc
                    write_catalog(peaks_df, peaks_catalog_path(field, distance_mpc, stage_paths))
                    all_rows.append(peaks_df)
                    pbar.update(1)
                    tqdm.write(
                        f"[peaks] {field} {distance_mpc:g} Mpc done in "
                        f"{format_elapsed(time.perf_counter() - distance_start)} "
                        f"({len(peaks_df)} peaks)"
                    )
                    continue
                flux_map = read_fits_data(degraded_flux_path(field, "Halpha", distance_mpc, stage_paths))
                err_map = read_fits_data(degraded_err_path(field, "Halpha", distance_mpc, stage_paths))
                distance_min_separation_px = max(1, min(min_separation_px, effective_peak_separation_px(distance_mpc)))
                peaks_df = detect_peaks_in_map(
                    flux_map,
                    err_map,
                    field=field,
                    distance_mpc=distance_mpc,
                    min_separation_px=distance_min_separation_px,
                    snr_threshold=main_params.snrlim if snr_threshold == MAIN_CATALOG_SNR_THRESHOLD else snr_threshold,
                    flux_percentile_threshold=calibrated_percentile if flux_percentile_threshold == 90.0 else flux_percentile_threshold,
                )
                write_catalog(peaks_df, peaks_catalog_path(field, distance_mpc, stage_paths))
                all_rows.append(peaks_df)
                pbar.update(1)
                tqdm.write(
                    f"[peaks] {field} {distance_mpc:g} Mpc done in "
                    f"{format_elapsed(time.perf_counter() - distance_start)} "
                    f"({len(peaks_df)} peaks)"
                )
            tqdm.write(f"[peaks] field {field} done in {format_elapsed(time.perf_counter() - field_start)}")
    tqdm.write(f"[peaks] all fields done in {format_elapsed(time.perf_counter() - stage_start)}")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    write_catalog(combined, stage_paths.catalogs_dir / "degraded_peak_inventory.csv")
    write_qc_report(
        stage_paths.qc_dir / "02_identify_peaks.json",
        stage="identify_peaks",
        field=None,
        warnings=[],
        summary={"n_peaks": int(len(combined)), "fields": list(fields), "distances_mpc": list(distances_mpc)},
    )
    return combined


def build_regions_for_field_distance(
    field: str,
    distance_mpc: float,
    *,
    stage_paths: ResolutionPaths,
    influence_exp: float = 2.0,
    sizebox: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    flux_map = read_fits_data(degraded_flux_path(field, "Halpha", distance_mpc, stage_paths))
    peaks_raw = read_catalog(peaks_catalog_path(field, distance_mpc, stage_paths))
    if peaks_raw.empty:
        empty_metrics = pd.DataFrame(columns=["region_id", "field", "distance_mpc"])
        zoi_map = np.full_like(flux_map, np.nan, dtype=float)
        boundary_map = np.zeros_like(flux_map, dtype=float)
        return empty_metrics, zoi_map, boundary_map

    peaks_df = build_peak_table(peaks_raw, "x", "y", flux_map.shape, field)
    peaks_df["distance_mpc"] = distance_mpc
    peaks_df["pixscale_arcsec_per_pix"] = PIXEL_SCALE_ARCSEC / (0.84 / distance_mpc)
    peaks_df["pixel_scale_pc"] = m33_pixel_scale_pc(peaks_df["pixscale_arcsec_per_pix"].iloc[0])
    peaks_df["spatial_resolution_pc"] = peaks_df["pixel_scale_pc"]
    scale_factor = 0.84 / distance_mpc
    roi = scaled_roi_from_param_section(field, flux_map.shape, sizebox=sizebox, scale_factor=scale_factor)
    pix_pc = float(peaks_df["pixel_scale_pc"].iloc[0])
    zoi_rmax_pc, zoi_rmax_px = effective_zoi_radius_for_map(distance_mpc, pix_pc)
    peaks_df["zoi_rmax_px"] = float(zoi_rmax_px)

    zoi_map = makezoi_legacy(
        flux_map,
        peaks_df,
        roi=roi,
        exponent=influence_exp,
        rmax_px=zoi_rmax_px,
    )
    boundary_map, metrics_df = build_boundary_map(
        flux_map,
        zoi_map,
        peaks_df,
        pixel_scale_pc=pix_pc,
        n_theta=36,
        r_bin=1.0,
        edge_frac=0.5,
    )
    metrics_df["distance_mpc"] = distance_mpc
    metrics_df["pixscale_arcsec_per_pix"] = float(peaks_df["pixscale_arcsec_per_pix"].iloc[0])
    metrics_df["zoi_max_radius_pc"] = zoi_rmax_pc
    return metrics_df, zoi_map, boundary_map


def build_regions_for_all_fields(
    fields: tuple[str, ...] = FIELDS,
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    stage_paths: ResolutionPaths | None = None,
) -> pd.DataFrame:
    stage_paths = ensure_resolution_dirs(stage_paths)
    all_metrics: list[pd.DataFrame] = []
    stage_start = time.perf_counter()
    total_steps = len(fields) * len(distances_mpc)

    with tqdm(total=total_steps, desc="Regions", dynamic_ncols=True) as pbar:
        for field in fields:
            field_start = time.perf_counter()
            safe_mkdir(stage_paths.regions_dir / field)
            for distance_mpc in distances_mpc:
                distance_start = time.perf_counter()
                metrics_df, zoi_map, boundary_map = build_regions_for_field_distance(field, distance_mpc, stage_paths=stage_paths)
                write_catalog(metrics_df, boundary_metrics_path(field, distance_mpc, stage_paths))
                write_fits(zoi_map_path(field, distance_mpc, stage_paths), np.nan_to_num(zoi_map, nan=0.0).astype(np.float32))
                write_fits(boundary_map_path(field, distance_mpc, stage_paths), boundary_map.astype(np.float32))
                all_metrics.append(metrics_df)
                pbar.update(1)
                tqdm.write(
                    f"[regions] {field} {distance_mpc:g} Mpc done in "
                    f"{format_elapsed(time.perf_counter() - distance_start)} "
                    f"({len(metrics_df)} regions)"
                )
            tqdm.write(f"[regions] field {field} done in {format_elapsed(time.perf_counter() - field_start)}")
    tqdm.write(f"[regions] all fields done in {format_elapsed(time.perf_counter() - stage_start)}")

    combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    write_catalog(combined, stage_paths.catalogs_dir / "region_metrics_inventory.csv")
    write_qc_report(
        stage_paths.qc_dir / "03_build_regions.json",
        stage="build_regions",
        field=None,
        warnings=[],
        summary={"n_regions": int(len(combined)), "fields": list(fields), "distances_mpc": list(distances_mpc)},
    )
    return combined


def extinction_multiplier(ebv: np.ndarray, k_lambda: float) -> np.ndarray:
    return np.power(10.0, 0.4 * k_lambda * ebv)


def add_extinction_and_deredden_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ha = out["F_Halpha_sum"].to_numpy(dtype=float)
    hb = out["F_Hbeta_sum"].to_numpy(dtype=float)
    ha_err = out["F_Halpha_e_sum"].to_numpy(dtype=float)
    hb_err = out["F_Hbeta_e_sum"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_ratio = ha / hb
        ebv = (2.5 / (K_HBETA - K_HALPHA)) * np.log10(observed_ratio / CASE_B_HA_HB)
        ebv = np.where(np.isfinite(ebv), np.maximum(ebv, 0.0), np.nan)
        observed_ratio_err = np.abs(observed_ratio) * np.sqrt((ha_err / ha) ** 2 + (hb_err / hb) ** 2)
        ebv_err = np.abs(2.5 / ((K_HBETA - K_HALPHA) * np.log(10.0))) * np.abs(observed_ratio_err / observed_ratio)

    out["sum_E_BV"] = ebv
    out["sum_E_BV_err"] = ebv_err
    out["sum_A_V"] = RV * ebv
    out["sum_A_V_err"] = RV * ebv_err

    line_k = {
        "Halpha": K_HALPHA,
        "Hbeta": K_HBETA,
        "[OIII]5007": 3.47,
        "[NII]6583": 2.52,
        "[SII]6716": 2.45,
        "[SII]6731": 2.44,
        "[OII]3727": 4.74,
    }
    for line_name, k_lambda in line_k.items():
        col = f"F_{line_key_to_column_name(line_name)}_sum"
        err_col = f"F_{line_key_to_column_name(line_name)}_e_sum"
        factor = extinction_multiplier(ebv, k_lambda)
        out[f"F_{line_key_to_column_name(line_name)}_sum_dered"] = out[col].to_numpy(dtype=float) * factor
        out[f"F_{line_key_to_column_name(line_name)}_e_sum_dered"] = out[err_col].to_numpy(dtype=float) * factor

    out["L_Ha_sum"] = flux_to_luminosity(out["F_Halpha_sum"].to_numpy(dtype=float), out["distance_mpc"].to_numpy(dtype=float))
    out["L_Ha_sum_dered"] = flux_to_luminosity(out["F_Halpha_sum_dered"].to_numpy(dtype=float), out["distance_mpc"].to_numpy(dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_L_Ha_sum_dered"] = np.log10(out["L_Ha_sum_dered"])
    return out


def add_galactocentric_radius(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_pix_orig"] = np.nan
    out["y_pix_orig"] = np.nan
    out["x_gal_pc"] = np.nan
    out["y_gal_pc"] = np.nan
    out["r_gal_pc"] = np.nan

    pc_per_pixel_orig = original_pixel_scale_pc()
    for field, field_rows in out.groupby("field").groups.items():
        center_x, center_y = field_m33_center_original_pixels(field)
        idx = list(field_rows)
        scale_factors = degraded_scale_factor(out.loc[idx, "distance_mpc"].to_numpy(dtype=float))
        x_orig = out.loc[idx, "x_pix"].to_numpy(dtype=float) / scale_factors
        y_orig = out.loc[idx, "y_pix"].to_numpy(dtype=float) / scale_factors
        x_major, y_minor = deproject_pixels_to_disk_pc(
            x_orig,
            y_orig,
            x0=center_x,
            y0=center_y,
            pa_deg=M33_PA_DEG,
            inc_deg=M33_INCLINATION_DEG,
            pc_per_pixel=pc_per_pixel_orig,
        )
        out.loc[idx, "x_pix_orig"] = x_orig
        out.loc[idx, "y_pix_orig"] = y_orig
        out.loc[idx, "x_gal_pc"] = x_major
        out.loc[idx, "y_gal_pc"] = y_minor
        out.loc[idx, "r_gal_pc"] = np.sqrt(np.asarray(x_major) ** 2 + np.asarray(y_minor) ** 2)
    out["r_gal_kpc"] = out["r_gal_pc"] / 1000.0
    return out


def sum_region_fluxes(boundary_map: np.ndarray, flux_map: np.ndarray, err_map: np.ndarray) -> dict[int, dict[str, float]]:
    label_values = np.unique(boundary_map[np.isfinite(boundary_map)])
    label_values = label_values[label_values > 0].astype(int)
    output: dict[int, dict[str, float]] = {}
    for label_value in label_values:
        region_mask = boundary_map == label_value
        ring_mask = region_edge_ring(region_mask, iterations=1)
        flux_vals = flux_map[region_mask]
        err_vals = err_map[region_mask]
        flux_vals = flux_vals[np.isfinite(flux_vals)]
        err_vals = err_vals[np.isfinite(err_vals)]
        flux_sum = float(np.nansum(flux_vals)) if flux_vals.size else np.nan
        err_sum = float(np.sqrt(np.nansum(err_vals**2))) if err_vals.size else np.nan
        background = float(np.nanmean(flux_map[ring_mask])) if np.any(np.isfinite(flux_map[ring_mask])) else np.nan
        output[label_value] = {
            "F_sum": flux_sum,
            "F_e_sum": err_sum,
            "background_edge": background,
            "npix_region": int(np.count_nonzero(region_mask)),
            "npix_edge_ring": int(np.count_nonzero(ring_mask)),
        }
    return output


def build_field_catalog_for_distance(field: str, distance_mpc: float, stage_paths: ResolutionPaths) -> pd.DataFrame:
    boundary_map = read_fits_data(boundary_map_path(field, distance_mpc, stage_paths))
    metrics_df = read_catalog(boundary_metrics_path(field, distance_mpc, stage_paths))
    if metrics_df.empty:
        return metrics_df

    merged = metrics_df.copy()
    for line_name in LINE_MAP_BASENAMES:
        flux_map = read_fits_data(degraded_flux_path(field, line_name, distance_mpc, stage_paths))
        err_map = read_fits_data(degraded_err_path(field, line_name, distance_mpc, stage_paths))
        stats = sum_region_fluxes(boundary_map, flux_map, err_map)
        prefix = line_key_to_column_name(line_name)
        merged[f"F_{prefix}_sum"] = merged["zoi_center_label"].map(lambda x: stats.get(int(x), {}).get("F_sum", np.nan))
        merged[f"F_{prefix}_e_sum"] = merged["zoi_center_label"].map(lambda x: stats.get(int(x), {}).get("F_e_sum", np.nan))
        merged[f"{prefix}_b_edge"] = merged["zoi_center_label"].map(lambda x: stats.get(int(x), {}).get("background_edge", np.nan))

    merged["field_distance"] = field_distance_tag(field, distance_mpc)
    merged = add_extinction_and_deredden_columns(merged)
    merged = add_galactocentric_radius(merged)
    merged = add_logU_KK04(merged, n_mc=0)
    merged = add_electron_density(merged, n_mc=0)
    merged = add_metallicity_columns(merged, use_odr=False)
    merged["metallicity_indicator"] = merged["Z_O3N2_M2013"]
    return merged


def build_resolution_catalogs(
    fields: tuple[str, ...] = FIELDS,
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    stage_paths: ResolutionPaths | None = None,
) -> dict[float, pd.DataFrame]:
    stage_paths = ensure_resolution_dirs(stage_paths)
    combined_by_distance: dict[float, pd.DataFrame] = {}
    write_global_combined = set(fields) == set(FIELDS)
    stage_start = time.perf_counter()
    total_steps = len(fields) * len(distances_mpc) + len(distances_mpc)

    with tqdm(total=total_steps, desc="Catalogs", dynamic_ncols=True) as pbar:
        for field in fields:
            field_start = time.perf_counter()
            safe_mkdir(stage_paths.catalogs_dir / field)
            for distance_mpc in distances_mpc:
                distance_start = time.perf_counter()
                field_catalog = build_field_catalog_for_distance(field, distance_mpc, stage_paths)
                write_catalog(field_catalog, field_catalog_path(field, distance_mpc, stage_paths))
                pbar.update(1)
                tqdm.write(
                    f"[catalogs] {field} {distance_mpc:g} Mpc done in "
                    f"{format_elapsed(time.perf_counter() - distance_start)} "
                    f"({len(field_catalog)} rows)"
                )
            tqdm.write(f"[catalogs] field {field} done in {format_elapsed(time.perf_counter() - field_start)}")

        for distance_mpc in distances_mpc:
            distance_start = time.perf_counter()
            field_catalogs = []
            for field in fields:
                path = field_catalog_path(field, distance_mpc, stage_paths)
                if path.exists():
                    df = read_catalog(path)
                    if not df.empty:
                        field_catalogs.append(df)
            combined = pd.concat(field_catalogs, ignore_index=True) if field_catalogs else pd.DataFrame()
            combined_by_distance[distance_mpc] = combined
            if write_global_combined:
                write_catalog(combined, total_catalog_path(distance_mpc, stage_paths))
            pbar.update(1)
            tqdm.write(
                f"[catalogs] combined {distance_mpc:g} Mpc done in "
                f"{format_elapsed(time.perf_counter() - distance_start)} "
                f"({len(combined)} rows)"
            )
    tqdm.write(f"[catalogs] all fields done in {format_elapsed(time.perf_counter() - stage_start)}")

    write_qc_report(
        stage_paths.qc_dir / "04_build_catalogs.json",
        stage="build_catalogs",
        field=None,
        warnings=[],
        summary={
            "fields": list(fields),
            "distances_mpc": list(distances_mpc),
            "n_catalogs": int(sum(len(df) for df in combined_by_distance.values())),
        },
    )
    return combined_by_distance


def fit_loglog_relation(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    coeff = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
    return float(coeff[0]), float(coeff[1])


def fit_linear_relation(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    coeff = np.polyfit(x[mask], y[mask], 1)
    return float(coeff[0]), float(coeff[1])


def filter_catalog_by_fields(df: pd.DataFrame, include_fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    if include_fields is None or df.empty or "field" not in df.columns:
        return df
    return df[df["field"].isin(include_fields)].copy()


def field_suffix(include_fields: tuple[str, ...] | None = None) -> str:
    if not include_fields:
        return ""
    clean = "_".join(include_fields)
    clean = re.sub(r"[^A-Za-z0-9_]+", "", clean)
    return f"__fields_{clean}"


def plot_peak_overlay(
    field: str,
    distance_mpc: float,
    stage_paths: ResolutionPaths | None = None,
    output_name: str | None = None,
) -> Path:
    stage_paths = ensure_resolution_dirs(stage_paths)
    flux_map = read_fits_data(degraded_flux_path(field, "Halpha", distance_mpc, stage_paths))
    peaks_df = read_catalog(peaks_catalog_path(field, distance_mpc, stage_paths))
    pix_pc = m33_pixel_scale_pc(PIXEL_SCALE_ARCSEC / degraded_scale_factor(distance_mpc))
    fig, ax = plt.subplots(figsize=(8, 8))
    finite = flux_map[np.isfinite(flux_map) & (flux_map > 0)]
    norm = None
    if finite.size:
        vmin = np.nanpercentile(finite, 5)
        vmax = np.nanpercentile(finite, 99.5)
        norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, np.nextafter(0, 1)), vmax=max(vmax, vmin * 1.01))
    ax.imshow(flux_map, origin="lower", cmap="rainbow", norm=norm)
    if not peaks_df.empty:
        ax.scatter(peaks_df["x"], peaks_df["y"], s=20, facecolors="none", edgecolors="cyan", linewidths=0.8)
    ax.text(
        0.03,
        0.97,
        f"{distance_tag(distance_mpc)} Mpc\n{pix_pc:.1f} pc/pix\nN={len(peaks_df)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )
    ax.set_title(f"{field} degraded Halpha peaks at {distance_tag(distance_mpc)} Mpc")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    output_path = stage_paths.plots_dir / (output_name or f"{field_distance_tag(field, distance_mpc)}_peak_overlay.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_peak_subplots_for_field(
    field: str,
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    stage_paths: ResolutionPaths | None = None,
) -> Path:
    stage_paths = ensure_resolution_dirs(stage_paths)
    ncols = 3
    nrows = int(np.ceil(len(distances_mpc) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, distance_mpc in zip(axes, distances_mpc):
        flux_map = read_fits_data(degraded_flux_path(field, "Halpha", distance_mpc, stage_paths))
        peaks_df = read_catalog(peaks_catalog_path(field, distance_mpc, stage_paths))
        finite = flux_map[np.isfinite(flux_map) & (flux_map > 0)]
        norm = None
        if finite.size:
            vmin = np.nanpercentile(finite, 5)
            vmax = np.nanpercentile(finite, 99.5)
            norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, np.nextafter(0, 1)), vmax=max(vmax, vmin * 1.01))
        ax.imshow(flux_map, origin="lower", cmap="rainbow", norm=norm)
        if not peaks_df.empty:
            ax.scatter(peaks_df["x"], peaks_df["y"], s=16, facecolors="none", edgecolors="cyan", linewidths=0.7)
        pix_pc = m33_pixel_scale_pc(PIXEL_SCALE_ARCSEC / degraded_scale_factor(distance_mpc))
        ax.text(
            0.03,
            0.97,
            f"{distance_tag(distance_mpc)} Mpc\n{pix_pc:.1f} pc/pix\nN={len(peaks_df)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )
        ax.set_title(f"{field}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(distances_mpc):]:
        ax.axis("off")

    plt.tight_layout()
    output_path = stage_paths.plots_dir / f"{field}_peak_subplots.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_luminosity_vs_radius_with_fits(
    catalogs_by_distance: dict[float, pd.DataFrame],
    stage_paths: ResolutionPaths | None = None,
    include_fields: tuple[str, ...] | None = None,
) -> Path:
    stage_paths = ensure_resolution_dirs(stage_paths)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(catalogs_by_distance)))

    for color, (distance_mpc, df) in zip(colors, sorted(catalogs_by_distance.items())):
        df = filter_catalog_by_fields(df, include_fields)
        if df.empty:
            continue
        radius_col = "radius_pc_eq_effective" if "radius_pc_eq_effective" in df.columns else "radius_pc_eq"
        x = df[radius_col].to_numpy(dtype=float)
        y = df["L_Ha_sum_dered"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if mask.sum() == 0:
            continue
        ax.scatter(x[mask], y[mask], s=14, alpha=0.35, color=color, label=f"{distance_tag(distance_mpc)} Mpc")
        fit = fit_loglog_relation(x, y)
        if fit is not None:
            slope, intercept = fit
            xfit = np.geomspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
            yfit = np.power(10.0, intercept) * np.power(xfit, slope)
            ax.plot(xfit, yfit, color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Region radius $R_{\rm eq}$ (pc)")
    ax.set_ylabel(r"Dereddened H$\alpha$ luminosity")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    output_path = stage_paths.plots_dir / f"luminosity_vs_radius_all_distances{field_suffix(include_fields)}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_derived_properties_vs_galactocentric_radius(
    catalogs_by_distance: dict[float, pd.DataFrame],
    stage_paths: ResolutionPaths | None = None,
    include_fields: tuple[str, ...] | None = None,
) -> list[Path]:
    stage_paths = ensure_resolution_dirs(stage_paths)
    outputs: list[Path] = []
    properties = [
        ("logU_KK04", r"$\log U$", "logU_vs_rgal_all_distances.png"),
        ("ne_SII_cm3", r"$n_e$ [cm$^{-3}$]", "density_vs_rgal_all_distances.png"),
        ("metallicity_indicator", r"$12+\log({\rm O/H})$", "metallicity_vs_rgal_all_distances.png"),
    ]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(catalogs_by_distance)))

    for column, ylabel, filename in properties:
        fig, ax = plt.subplots(figsize=(8, 6))
        for color, (distance_mpc, df) in zip(colors, sorted(catalogs_by_distance.items())):
            df = filter_catalog_by_fields(df, include_fields)
            if df.empty or column not in df.columns:
                continue
            mask = np.isfinite(df["r_gal_kpc"]) & np.isfinite(df[column])
            if mask.sum() == 0:
                continue
            x = df.loc[mask, "r_gal_kpc"].to_numpy(dtype=float)
            y = df.loc[mask, column].to_numpy(dtype=float)
            ax.scatter(x, y, s=12, alpha=0.35, color=color, label=f"{distance_tag(distance_mpc)} Mpc")
            fit = fit_linear_relation(x, y)
            if fit is not None:
                slope, intercept = fit
                xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                yfit = slope * xfit + intercept
                ax.plot(xfit, yfit, color=color, linewidth=2)
        ax.set_xlabel("Galactocentric radius (kpc)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        stem, ext = filename.rsplit(".", 1)
        output_path = stage_paths.plots_dir / f"{stem}{field_suffix(include_fields)}.{ext}"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def plot_boundary_example(
    field: str,
    distance_mpc: float,
    stage_paths: ResolutionPaths | None = None,
    output_name: str | None = None,
) -> Path:
    stage_paths = ensure_resolution_dirs(stage_paths)
    image = read_fits_data(degraded_flux_path(field, "Halpha", distance_mpc, stage_paths))
    label_map = read_fits_data(boundary_map_path(field, distance_mpc, stage_paths))
    catalog = read_catalog(boundary_metrics_path(field, distance_mpc, stage_paths))
    fig, _ = plot_segmentation_example(image, label_map, catalog, title=f"{field} boundaries at {distance_tag(distance_mpc)} Mpc")
    output_path = stage_paths.plots_dir / (output_name or f"{field_distance_tag(field, distance_mpc)}_boundaries.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_boundary_subplots_for_field(
    field: str,
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    stage_paths: ResolutionPaths | None = None,
) -> Path:
    stage_paths = ensure_resolution_dirs(stage_paths)
    ncols = 3
    nrows = int(np.ceil(len(distances_mpc) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, distance_mpc in zip(axes, distances_mpc):
        image = read_fits_data(degraded_flux_path(field, "Halpha", distance_mpc, stage_paths))
        boundary = read_fits_data(boundary_map_path(field, distance_mpc, stage_paths))
        finite = image[np.isfinite(image) & (image > 0)]
        norm = None
        if finite.size:
            vmin = np.nanpercentile(finite, 5)
            vmax = np.nanpercentile(finite, 99.5)
            norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, np.nextafter(0, 1)), vmax=max(vmax, vmin * 1.01))
        ax.imshow(image, origin="lower", cmap="rainbow", norm=norm)
        ax.contour(boundary > 0, levels=[0.5], colors=[plt.cm.rainbow(0.95)], linewidths=0.6)
        pix_pc = m33_pixel_scale_pc(PIXEL_SCALE_ARCSEC / degraded_scale_factor(distance_mpc))
        n_regions = int(np.unique(boundary[boundary > 0]).size)
        ax.text(
            0.03,
            0.97,
            f"{distance_tag(distance_mpc)} Mpc\n{pix_pc:.1f} pc/pix\nN={n_regions}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )
        ax.set_title(f"{field}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(distances_mpc):]:
        ax.axis("off")

    plt.tight_layout()
    output_path = stage_paths.plots_dir / f"{field}_boundary_subplots.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def make_resolution_plots(
    distances_mpc: tuple[float, ...] = DEFAULT_DISTANCES_MPC,
    *,
    example_field: str = "NW",
    example_distance_mpc: float = 20.0,
    stage_paths: ResolutionPaths | None = None,
    include_fields: tuple[str, ...] | None = None,
) -> list[Path]:
    stage_paths = ensure_resolution_dirs(stage_paths)
    catalogs_by_distance = {
        distance_mpc: read_catalog(total_catalog_path(distance_mpc, stage_paths))
        for distance_mpc in distances_mpc
        if total_catalog_path(distance_mpc, stage_paths).exists()
    }
    outputs = [
        plot_peak_overlay(example_field, example_distance_mpc, stage_paths=stage_paths),
        plot_boundary_example(example_field, example_distance_mpc, stage_paths=stage_paths),
        plot_luminosity_vs_radius_with_fits(catalogs_by_distance, stage_paths=stage_paths, include_fields=include_fields),
    ]
    outputs.extend(plot_derived_properties_vs_galactocentric_radius(catalogs_by_distance, stage_paths=stage_paths, include_fields=include_fields))
    write_qc_report(
        stage_paths.qc_dir / "05_make_plots.json",
        stage="make_plots",
        field=example_field,
        warnings=[],
        summary={"n_outputs": len(outputs), "example_distance_mpc": example_distance_mpc, "include_fields": list(include_fields) if include_fields else None},
    )
    return outputs
