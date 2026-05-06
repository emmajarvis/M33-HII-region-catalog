from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import binary_dilation

from .config import PhotometryConfig
from .io import read_catalog, read_fits_data, write_catalog
from . import paths


LINE_MAP_BASENAMES = {
    "Halpha": "Haflux",
    "Hbeta": "Hbflux",
    "[OIII]5007": "OIII5007flux",
    "[SII]6716": "SII6716flux",
    "[SII]6731": "SII6731flux",
    "[NII]6583": "NII6584flux",
    "[OII]3727": "OII3727flux",
}


def region_edge_ring(region_mask, iterations: int = 1):
    region_mask = np.asarray(region_mask, dtype=bool)
    dilated = binary_dilation(region_mask, iterations=iterations)
    return dilated & (~region_mask)


def zoi_annulus_mask(zoi_map: np.ndarray, boundary_map: np.ndarray, region_id: int) -> np.ndarray:
    zoi_mask = np.asarray(zoi_map == region_id, dtype=bool)
    region_mask = np.asarray(boundary_map == region_id, dtype=bool)
    return zoi_mask & (~region_mask)


def robust_dig_background(
    flux: np.ndarray,
    annulus_mask: np.ndarray,
    clip_sigma: float = 3.0,
    clip_iterations: int = 2,
):
    vals = np.asarray(flux, dtype=float)[annulus_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "dig_median": np.nan,
            "dig_mad": np.nan,
            "n_annulus": 0,
            "n_annulus_used": 0,
            "dig_clip_upper": np.nan,
        }

    working = vals.copy()
    clip_upper = np.nan
    for _ in range(max(int(clip_iterations), 1)):
        med = float(np.nanmedian(working))
        mad = float(1.4826 * np.nanmedian(np.abs(working - med)))
        if not np.isfinite(mad) or mad <= 0 or not np.isfinite(clip_sigma):
            break
        clip_upper = med + float(clip_sigma) * mad
        clipped = working[working <= clip_upper]
        if clipped.size == 0 or clipped.size == working.size:
            break
        working = clipped

    med = float(np.nanmedian(working))
    mad = float(1.4826 * np.nanmedian(np.abs(working - med))) if working.size else np.nan
    if not np.isfinite(clip_upper) and np.isfinite(mad) and mad > 0:
        clip_upper = med + float(clip_sigma) * mad
    return {
        "dig_median": med,
        "dig_mad": mad,
        "n_annulus": int(vals.size),
        "n_annulus_used": int(working.size),
        "dig_clip_upper": float(clip_upper) if np.isfinite(clip_upper) else np.nan,
    }


def integrated_flux_and_snr(flux, err, region_mask, background_per_pixel=np.nan, clip_negative_after_bg: bool = False):
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)

    f_vals = flux[region_mask]
    e_vals = err[region_mask]
    good = np.isfinite(f_vals) & np.isfinite(e_vals)
    f_vals = f_vals[good]
    e_vals = e_vals[good]

    npix = f_vals.size
    if npix == 0:
        return {
            "npix": 0,
            "F_raw": np.nan,
            "sigma_F": np.nan,
            "SNR_raw": np.nan,
            "background_per_pixel": np.nan,
            "F_bgsub": np.nan,
            "SNR_bgsub": np.nan,
        }

    flux_raw = np.sum(f_vals)
    sigma_flux = np.sqrt(np.sum(e_vals**2))
    snr_raw = flux_raw / sigma_flux if sigma_flux > 0 else np.nan

    background = background_per_pixel
    flux_bgsub = np.nan
    snr_bgsub = np.nan

    if np.isfinite(background):
        corrected = f_vals - background
        if clip_negative_after_bg:
            corrected = np.clip(corrected, 0, None)
        flux_bgsub = np.sum(corrected)
        snr_bgsub = flux_bgsub / sigma_flux if sigma_flux > 0 else np.nan

    return {
        "npix": npix,
        "F_raw": flux_raw,
        "sigma_F": sigma_flux,
        "SNR_raw": snr_raw,
        "background_per_pixel": background,
        "F_bgsub": flux_bgsub,
        "SNR_bgsub": snr_bgsub,
    }


def merge_check_duplicates(left_df, right_df, on, how: str = "left", suffixes=("_x", "_y"), rtol=0, atol=0):
    if isinstance(on, str):
        on = [on]

    overlap = (set(left_df.columns) & set(right_df.columns)) - set(on)
    merged = left_df.merge(right_df, on=on, how=how, suffixes=suffixes)

    for col in overlap:
        cx = f"{col}{suffixes[0]}"
        cy = f"{col}{suffixes[1]}"
        if cx not in merged.columns or cy not in merged.columns:
            continue
        a = merged[cx].to_numpy()
        b = merged[cy].to_numpy()
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            equal = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        else:
            equal = np.all((a == b) | (pd.isna(a) & pd.isna(b)))
        if equal:
            merged[col] = merged[cx]
            merged.drop(columns=[cx, cy], inplace=True)

    return merged


def _read_fits(path: Path):
    with fits.open(path) as hdul:
        return hdul[0].data, hdul[0].header


def load_flux_maps(field: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    folder = paths.calibrated_field_map_dir(field)
    maps = {}
    for line_name, base in LINE_MAP_BASENAMES.items():
        flux_data, _ = _read_fits(folder / f"M33{field}-{base}.fits")
        err_data, _ = _read_fits(folder / f"M33{field}-{base}-err.fits")
        maps[line_name] = (flux_data, err_data)
    return maps


def load_region_inputs(field: str, max_zoi_pc: int) -> dict[str, object]:
    peaks_df = read_catalog(paths.final_peaks_csv(field), comment="#")
    zoi_map = read_fits_data(paths.zoi_fits(field, max_zoi_pc))
    boundary_map = read_fits_data(paths.boundary_fits(field, max_zoi_pc))
    boundary_metrics_df = read_catalog(paths.boundary_metrics_csv(field, max_zoi_pc))
    return {
        "peaks_df": peaks_df,
        "zoi_map": zoi_map,
        "boundary_map": boundary_map,
        "boundary_metrics_df": boundary_metrics_df,
    }


def load_integrated_flux_table(field: str, expected_len: int) -> pd.DataFrame:
    ascii_path = paths.fit_int_spec_ascii(field)
    if ascii_path.exists():
        int_flux_df = pd.read_csv(ascii_path, sep=r"\s+", comment="#")
        int_flux_df = int_flux_df.sort_values(by="id").reset_index(drop=True)
        missing_ids = sorted(set(range(expected_len)) - set(int_flux_df["id"]))
        for missing_id in missing_ids:
            filler = {col: np.nan for col in int_flux_df.columns}
            filler["id"] = missing_id
            int_flux_df = pd.concat([int_flux_df, pd.DataFrame([filler])], ignore_index=True)
        int_flux_df = int_flux_df.sort_values(by="id").reset_index(drop=True)
    else:
        int_flux_df = pd.DataFrame({"id": np.arange(expected_len, dtype=int)})

    rename_map = {col: f"{col}_int" for col in int_flux_df.columns}
    return int_flux_df.rename(columns=rename_map)


def load_integrated_flux_table_canonical(field: str, expected_len: int) -> pd.DataFrame:
    ascii_path = paths.fit_int_spec_ascii(field)
    if ascii_path.exists():
        int_flux_df = pd.read_csv(ascii_path, sep=r"\s+", comment="#")
        int_flux_df = int_flux_df.sort_values(by="id").reset_index(drop=True)
    else:
        int_flux_df = pd.DataFrame({"id": np.arange(expected_len, dtype=int)})

    for missing_id in sorted(set(range(expected_len)) - set(int_flux_df["id"])):
        filler = {col: np.nan for col in int_flux_df.columns}
        filler["id"] = missing_id
        int_flux_df = pd.concat([int_flux_df, pd.DataFrame([filler])], ignore_index=True)
    int_flux_df = int_flux_df.sort_values(by="id").reset_index(drop=True)

    rename_map = {}
    for col in int_flux_df.columns:
        if col == "id":
            rename_map[col] = "id_int"
        elif col in {"x", "y", "FIELD"}:
            rename_map[col] = f"{col}_int"
        elif col.startswith("F_"):
            rename_map[col] = f"{col}_sum"
        elif col.startswith("SNR_"):
            rename_map[col] = f"{col}_sum"
        elif col.startswith("F_") and col.endswith("_e"):
            rename_map[col] = f"{col[:-2]}_e_sum"
        elif col.endswith("_e") and col.startswith("F_"):
            rename_map[col] = f"{col[:-2]}_e_sum"
        else:
            rename_map[col] = col

    canonical_df = int_flux_df.rename(columns=rename_map)
    if "FIELD_int" in canonical_df.columns:
        canonical_df = canonical_df.drop(columns=["FIELD_int"])
    return canonical_df


def _apply_active_flux_columns(df: pd.DataFrame, line_names: list[str], dig_mode: str) -> pd.DataFrame:
    out = df.copy()
    if dig_mode not in {"no_dig", "dig_subtracted"}:
        raise ValueError(f"Unknown DIG mode: {dig_mode}")
    for line_name in line_names:
        base_flux = f"F_{line_name}_sum_nodig"
        base_err = f"F_{line_name}_e_sum_nodig"
        base_snr = f"SNR_{line_name}_sum_nodig"
        dig_flux = f"F_{line_name}_sum_digsub"
        dig_err = f"F_{line_name}_e_sum_digsub"
        dig_snr = f"SNR_{line_name}_sum_digsub"
        if dig_mode == "dig_subtracted":
            out[f"F_{line_name}_sum"] = out[dig_flux]
            out[f"F_{line_name}_e_sum"] = out[dig_err]
            out[f"SNR_{line_name}_sum"] = out[dig_snr]
        else:
            out[f"F_{line_name}_sum"] = out[base_flux]
            out[f"F_{line_name}_e_sum"] = out[base_err]
            out[f"SNR_{line_name}_sum"] = out[base_snr]
    out["dig_mode"] = dig_mode
    return out


def _compute_dig_background_catalog(
    boundary_map: np.ndarray,
    zoi_map: np.ndarray,
    maps: dict[str, tuple[np.ndarray, np.ndarray]],
    config: PhotometryConfig,
) -> pd.DataFrame:
    labels = np.unique(boundary_map[np.isfinite(boundary_map)])
    labels = labels[labels > 0].astype(int)
    rows = []
    for region_id in labels:
        region_mask = boundary_map == region_id
        annulus_mask = zoi_annulus_mask(zoi_map, boundary_map, region_id)
        row = {
            "region_id": region_id,
            "npix_region": int(np.sum(region_mask)),
            "npix_dig_annulus": int(np.sum(annulus_mask)),
        }
        for line_name, (flux, err) in maps.items():
            dig = robust_dig_background(
                flux,
                annulus_mask,
                clip_sigma=config.dig_clip_sigma,
                clip_iterations=config.dig_clip_iterations,
            )
            stats = integrated_flux_and_snr(
                flux=flux,
                err=err,
                region_mask=region_mask,
                background_per_pixel=dig["dig_median"],
                clip_negative_after_bg=config.clip_negative_after_bg,
            )
            row[f"F_{line_name}_sum_nodig"] = stats["F_raw"]
            row[f"F_{line_name}_e_sum_nodig"] = stats["sigma_F"]
            row[f"SNR_{line_name}_sum_nodig"] = stats["SNR_raw"]
            row[f"{line_name}_dig_median"] = dig["dig_median"]
            row[f"{line_name}_dig_mad"] = dig["dig_mad"]
            row[f"{line_name}_dig_clip_upper"] = dig["dig_clip_upper"]
            row[f"{line_name}_dig_npix"] = dig["n_annulus"]
            row[f"{line_name}_dig_npix_used"] = dig["n_annulus_used"]
            row[f"F_{line_name}_sum_digsub"] = stats["F_bgsub"]
            row[f"F_{line_name}_e_sum_digsub"] = stats["sigma_F"]
            row[f"SNR_{line_name}_sum_digsub"] = stats["SNR_bgsub"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_summed_map_flux_catalog(field: str, max_zoi_pc: int, config: PhotometryConfig, dig_mode: str = "no_dig") -> pd.DataFrame:
    region_inputs = load_region_inputs(field, max_zoi_pc)
    peaks_df = region_inputs["peaks_df"]
    zoi_map = region_inputs["zoi_map"]
    boundary_map = region_inputs["boundary_map"]
    boundary_metrics_df = region_inputs["boundary_metrics_df"]
    maps = load_flux_maps(field)
    flux_catalog_df = _compute_dig_background_catalog(boundary_map, zoi_map, maps, config)
    flux_catalog_df = _apply_active_flux_columns(flux_catalog_df, list(maps), dig_mode=dig_mode)
    if {"F_Halpha_sum_nodig", "F_Halpha_sum_digsub"}.issubset(flux_catalog_df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_catalog_df["DIG_fraction"] = 1.0 - (
                flux_catalog_df["F_Halpha_sum_digsub"].to_numpy(dtype=float)
                / flux_catalog_df["F_Halpha_sum_nodig"].to_numpy(dtype=float)
            )
    merged_df = pd.concat(
        [
            flux_catalog_df.reset_index(drop=True),
            peaks_df.reset_index(drop=True),
            boundary_metrics_df.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df["field"] = field
    merged_df["flux_method"] = "summed_map"
    merged_df["dig_mode"] = dig_mode
    return merged_df


def build_integrated_spectrum_flux_catalog(field: str, max_zoi_pc: int, config: PhotometryConfig, dig_mode: str = "no_dig") -> pd.DataFrame:
    region_inputs = load_region_inputs(field, max_zoi_pc)
    peaks_df = region_inputs["peaks_df"]
    zoi_map = region_inputs["zoi_map"]
    boundary_map = region_inputs["boundary_map"]
    boundary_metrics_df = region_inputs["boundary_metrics_df"]
    int_flux_df = load_integrated_flux_table_canonical(field, len(peaks_df))
    maps = load_flux_maps(field)

    labels = np.unique(boundary_map[np.isfinite(boundary_map)])
    labels = labels[labels > 0].astype(int)
    region_rows = []
    for region_id in labels:
        region_mask = boundary_map == region_id
        region_rows.append(
            {
                "region_id": region_id,
                "npix_region": int(np.sum(region_mask)),
                "npix_edge_ring": np.nan,
            }
        )
    base_df = pd.DataFrame(region_rows).sort_values("region_id").reset_index(drop=True)
    if base_df.empty:
        base_df = pd.DataFrame(
            {
                "region_id": np.arange(len(peaks_df), dtype=int) + 1,
                "npix_region": np.nan,
                "npix_edge_ring": np.nan,
            }
        )

    dig_df = _compute_dig_background_catalog(boundary_map, zoi_map, maps, config)
    dig_df = dig_df.drop(columns=["npix_region"], errors="ignore")

    merged_flux_df = pd.concat([base_df.reset_index(drop=True), int_flux_df.reset_index(drop=True)], axis=1)
    merged_flux_df = merged_flux_df.loc[:, ~merged_flux_df.columns.duplicated()]
    merged_flux_df = merge_check_duplicates(merged_flux_df, dig_df, on="region_id", how="left")

    for line_name in maps:
        raw_flux_col = f"F_{line_name}_sum"
        raw_err_col = f"F_{line_name}_e_sum"
        raw_snr_col = f"SNR_{line_name}_sum"
        if raw_flux_col not in merged_flux_df.columns:
            merged_flux_df[raw_flux_col] = np.nan
        if raw_err_col not in merged_flux_df.columns:
            merged_flux_df[raw_err_col] = np.nan
        if raw_snr_col not in merged_flux_df.columns:
            merged_flux_df[raw_snr_col] = np.nan
        merged_flux_df[f"F_{line_name}_sum_nodig"] = merged_flux_df[raw_flux_col]
        merged_flux_df[f"F_{line_name}_e_sum_nodig"] = merged_flux_df[raw_err_col]
        merged_flux_df[f"SNR_{line_name}_sum_nodig"] = merged_flux_df[raw_snr_col]
        background = merged_flux_df[f"{line_name}_dig_median"].to_numpy(dtype=float)
        npix_region = merged_flux_df["npix_region"].to_numpy(dtype=float)
        raw_flux = merged_flux_df[raw_flux_col].to_numpy(dtype=float)
        raw_err = merged_flux_df[raw_err_col].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            dig_flux = raw_flux - background * npix_region
        merged_flux_df[f"F_{line_name}_sum_digsub"] = dig_flux
        merged_flux_df[f"F_{line_name}_e_sum_digsub"] = raw_err
        with np.errstate(divide="ignore", invalid="ignore"):
            merged_flux_df[f"SNR_{line_name}_sum_digsub"] = dig_flux / raw_err
    merged_flux_df = _apply_active_flux_columns(merged_flux_df, list(maps), dig_mode=dig_mode)
    if {"F_Halpha_sum_nodig", "F_Halpha_sum_digsub"}.issubset(merged_flux_df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            merged_flux_df["DIG_fraction"] = 1.0 - (
                merged_flux_df["F_Halpha_sum_digsub"].to_numpy(dtype=float)
                / merged_flux_df["F_Halpha_sum_nodig"].to_numpy(dtype=float)
            )

    merged_df = pd.concat(
        [
            merged_flux_df.reset_index(drop=True),
            peaks_df.reset_index(drop=True),
            boundary_metrics_df.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df["field"] = field
    merged_df["flux_method"] = "integrated_spectrum"
    merged_df["dig_mode"] = dig_mode
    return merged_df


def build_field_flux_catalog(
    field: str,
    max_zoi_pc: int,
    config: PhotometryConfig,
    method: str = "summed_map",
    dig_mode: str = "no_dig",
) -> pd.DataFrame:
    if method == "summed_map":
        return build_summed_map_flux_catalog(field, max_zoi_pc, config, dig_mode=dig_mode)
    if method == "integrated_spectrum":
        return build_integrated_spectrum_flux_catalog(field, max_zoi_pc, config, dig_mode=dig_mode)
    raise ValueError(f"Unknown flux catalog method: {method}")


def write_field_flux_catalog(field: str, df: pd.DataFrame, method: str = "summed_map", dig_mode: str = "no_dig") -> Path:
    out_path = paths.flux_catalog_csv(field, method=method, dig_mode=dig_mode)
    write_catalog(df, out_path)
    return out_path
