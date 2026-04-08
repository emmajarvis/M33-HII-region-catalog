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


def integrated_flux_and_snr(flux, err, region_mask, ring_mask=None, clip_negative_after_bg: bool = False):
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
            "b_edge": np.nan,
            "F_bgsub": np.nan,
            "SNR_bgsub": np.nan,
        }

    flux_raw = np.sum(f_vals)
    sigma_flux = np.sqrt(np.sum(e_vals**2))
    snr_raw = flux_raw / sigma_flux if sigma_flux > 0 else np.nan

    background = np.nan
    flux_bgsub = np.nan
    snr_bgsub = np.nan

    if ring_mask is not None:
        ring_vals = flux[ring_mask]
        ring_vals = ring_vals[np.isfinite(ring_vals)]
        background = np.nanmean(ring_vals) if ring_vals.size else np.nan
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
        "b_edge": background,
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
        int_flux_df = pd.read_csv(ascii_path, delim_whitespace=True, comment="#")
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


def build_field_flux_catalog(field: str, max_zoi_pc: int, config: PhotometryConfig) -> pd.DataFrame:
    region_inputs = load_region_inputs(field, max_zoi_pc)
    peaks_df = region_inputs["peaks_df"]
    boundary_map = region_inputs["boundary_map"]
    boundary_metrics_df = region_inputs["boundary_metrics_df"]
    int_flux_df = load_integrated_flux_table(field, len(peaks_df))
    maps = load_flux_maps(field)

    labels = np.unique(boundary_map[np.isfinite(boundary_map)])
    labels = labels[labels > 0].astype(int)

    rows = []
    for region_id in labels:
        region_mask = boundary_map == region_id
        ring_mask = region_edge_ring(region_mask, iterations=config.edge_ring_iterations)
        row = {
            "region_id": region_id,
            "npix_region": int(np.sum(region_mask)),
            "npix_edge_ring": int(np.sum(ring_mask)),
        }
        for line_name, (flux, err) in maps.items():
            stats = integrated_flux_and_snr(
                flux=flux,
                err=err,
                region_mask=region_mask,
                ring_mask=ring_mask,
                clip_negative_after_bg=config.clip_negative_after_bg,
            )
            row[f"F_{line_name}_sum"] = stats["F_raw"]
            row[f"F_{line_name}_e_sum"] = stats["sigma_F"]
            row[f"SNR_{line_name}_sum"] = stats["SNR_raw"]
            row[f"{line_name}_b_edge"] = stats["b_edge"]
            row[f"F_{line_name}_bgsub"] = stats["F_bgsub"]
            row[f"SNR_{line_name}_bgsub"] = stats["SNR_bgsub"]
        rows.append(row)

    flux_catalog_df = pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)
    merged_df = pd.concat(
        [
            flux_catalog_df.reset_index(drop=True),
            peaks_df.reset_index(drop=True),
            boundary_metrics_df.reset_index(drop=True),
            int_flux_df.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df["field"] = field
    return merged_df


def write_field_flux_catalog(field: str, df: pd.DataFrame) -> Path:
    out_path = paths.flux_catalog_csv(field)
    write_catalog(df, out_path)
    return out_path
