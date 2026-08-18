from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import binary_dilation, gaussian_filter

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

INTEGRATED_SPECTRA_CALIBRATIONS = {
    "NE": (0.5037, 1.1034, 1.0652),
    "NW": (0.6845, 1.0958, 1.0966),
    "SE": (0.8600, 1.1489, 1.3555),
    "SW": (0.9279, 1.1128, 0.8805),
    "F5": (0.6194, 1.5244, 1.5353),
    "F6": (0.7183, 1.1947, 1.5717),
    "F7": (0.8184, 0.9557, 1.1887),
    "F8": (0.9931, 0.8587, 0.9162),
    "F9": (0.8288, 0.8131, 0.8942),
}

INTEGRATED_SPECTRA_SN_COLUMNS = {
    0: ("F_[OII]3727", "F_[OII]3727_e"),
    1: (
        "F_Hbeta",
        "F_Hbeta_e",
        "F_[OIII]4959",
        "F_[OIII]4959_e",
        "F_[OIII]5007",
        "F_[OIII]5007_e",
    ),
    2: (
        "F_[NII]6548",
        "F_[NII]6548_e",
        "F_Halpha",
        "F_Halpha_e",
        "F_[NII]6583",
        "F_[NII]6583_e",
        "F_[SII]6716",
        "F_[SII]6716_e",
        "F_[SII]6731",
        "F_[SII]6731_e",
    ),
}


def region_edge_ring(region_mask, iterations: int = 1):
    region_mask = np.asarray(region_mask, dtype=bool)
    dilated = binary_dilation(region_mask, iterations=iterations)
    return dilated & (~region_mask)


def zoi_annulus_mask(zoi_map: np.ndarray, boundary_map: np.ndarray, region_id: int) -> np.ndarray:
    zoi_mask = np.asarray(zoi_map == region_id, dtype=bool)
    region_mask = np.asarray(boundary_map == region_id, dtype=bool)
    return zoi_mask & (~region_mask)


def _scaled_local_background_mask(
    zoi_map: np.ndarray,
    boundary_map: np.ndarray,
    region_id: int,
    config: PhotometryConfig,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Build a local background shell inside the region ZoI, excluding all H II boundaries."""
    region_mask = np.asarray(boundary_map == region_id, dtype=bool)
    zoi_mask = np.asarray(zoi_map == region_id, dtype=bool)
    all_region_mask = np.asarray(np.isfinite(boundary_map) & (boundary_map > 0), dtype=bool)
    background_map = zoi_mask & (~all_region_mask)
    npix_region = int(np.sum(region_mask))

    if npix_region <= 0:
        return background_map & False, {
            "dig_annulus_inner_px": int(config.dig_annulus_inner_px),
            "dig_annulus_width_px": 0,
            "dig_annulus_outer_px": int(config.dig_annulus_inner_px),
            "dig_annulus_source": "empty_region",
        }

    equivalent_radius_px = float(np.sqrt(npix_region / np.pi))
    width_px = int(np.ceil(equivalent_radius_px * float(config.dig_annulus_width_fraction)))
    width_px = max(width_px, int(config.dig_annulus_min_width_px))
    width_px = min(width_px, int(config.dig_annulus_max_width_px))
    inner_px = max(int(config.dig_annulus_inner_px), 0)
    outer_px = inner_px + width_px

    outer_mask = binary_dilation(region_mask, iterations=max(outer_px, 1))
    if inner_px > 0:
        inner_mask = binary_dilation(region_mask, iterations=inner_px)
    else:
        inner_mask = region_mask
    local_mask = outer_mask & (~inner_mask) & background_map
    if np.any(local_mask):
        source = "local_zoi_annulus"
        background_mask = local_mask
    else:
        source = "full_zoi_background"
        background_mask = background_map

    return background_mask, {
        "dig_annulus_inner_px": inner_px,
        "dig_annulus_width_px": width_px,
        "dig_annulus_outer_px": outer_px,
        "dig_annulus_source": source,
    }


def robust_dig_background(
    flux: np.ndarray,
    annulus_mask: np.ndarray,
    clip_sigma: float = 3.0,
    clip_iterations: int = 2,
    background_percentile: float = 25.0,
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

    # Use the requested percentile of positive, sigma-clipped background values
    # so the background subtraction can only subtract flux, never add it back.
    positive_working = working[working > 0]
    if positive_working.size == 0:
        bg_level = 0.0
    else:
        bg_level = float(np.nanpercentile(positive_working, background_percentile))
    bg_level = max(bg_level, 0.0)
    mad = float(1.4826 * np.nanmedian(np.abs(working - bg_level))) if working.size else np.nan
    if not np.isfinite(clip_upper) and np.isfinite(mad) and mad > 0:
        clip_upper = bg_level + float(clip_sigma) * mad
    return {
        "dig_median": bg_level,
        "dig_mad": mad,
        "n_annulus": int(vals.size),
        "n_annulus_used": int(working.size),
        "dig_clip_upper": float(clip_upper) if np.isfinite(clip_upper) else np.nan,
    }


def _smoothed_masked_background_map(
    flux: np.ndarray,
    boundary_map: np.ndarray,
    config: PhotometryConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Estimate a smooth line-background map after masking all H II boundaries."""
    flux = np.asarray(flux, dtype=float)
    all_region_mask = np.asarray(np.isfinite(boundary_map) & (boundary_map > 0), dtype=bool)
    valid_background = np.isfinite(flux) & (~all_region_mask)
    positive_background = np.where(valid_background & (flux > 0), flux, 0.0)
    weights = valid_background.astype(float)

    sigma = float(config.dig_background_smooth_sigma_px)
    smoothed_flux = gaussian_filter(positive_background, sigma=sigma, mode="nearest")
    smoothed_weight = gaussian_filter(weights, sigma=sigma, mode="nearest")
    min_weight = float(config.dig_background_min_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        background = smoothed_flux / smoothed_weight
    background[(smoothed_weight <= min_weight) | ~np.isfinite(background)] = np.nan
    background = np.where(np.isfinite(background) & (background > 0), background, 0.0)
    return background, {
        "dig_background_smooth_sigma_px": sigma,
        "dig_background_min_weight": min_weight,
        "dig_background_npix": int(np.sum(valid_background)),
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


def _background_subtracted_flux(raw_flux, npix_region, background_per_pixel) -> float:
    if not np.isfinite(raw_flux) or not np.isfinite(npix_region) or not np.isfinite(background_per_pixel):
        return np.nan
    return float(raw_flux) - float(background_per_pixel) * float(npix_region)


def _select_nonnegative_dig_background(
    raw_flux,
    npix_region,
    local_background=np.nan,
) -> tuple[float, float, str]:
    """Choose a DIG background that does not drive the integrated flux negative."""
    local_background = _positive_or_zero(local_background)
    flux_local = _background_subtracted_flux(raw_flux, npix_region, local_background)
    if np.isfinite(flux_local) and flux_local >= 0:
        return local_background, flux_local, "local_line"

    return 0.0, float(raw_flux) if np.isfinite(raw_flux) else np.nan, "none_negative_guard"


def _select_capped_dig_background(
    raw_flux,
    npix_region,
    model_background_flux,
    max_subtraction_fraction,
) -> tuple[float, float, float, str]:
    """Choose a capped integrated DIG background that cannot make a flux negative."""
    if not np.isfinite(raw_flux):
        return np.nan, np.nan, np.nan, "invalid_raw_flux"
    if not np.isfinite(npix_region) or float(npix_region) <= 0:
        return 0.0, float(raw_flux), np.nan, "none_empty_region"
    if float(raw_flux) <= 0:
        return 0.0, float(raw_flux), 0.0, "none_nonpositive_raw_flux"

    model_background_flux = _positive_or_zero(model_background_flux)
    max_fraction = max(float(max_subtraction_fraction), 0.0)
    cap = max_fraction * float(raw_flux)
    selected_background_flux = min(model_background_flux, cap)
    selected_background_flux = min(selected_background_flux, float(raw_flux))
    corrected_flux = float(raw_flux) - selected_background_flux
    if corrected_flux < 0:
        selected_background_flux = 0.0
        corrected_flux = float(raw_flux)
        method = "none_negative_guard"
    elif model_background_flux > cap:
        method = "smoothed_map_capped"
    elif selected_background_flux > 0:
        method = "smoothed_map"
    else:
        method = "none_zero_background"

    background_per_pixel = selected_background_flux / float(npix_region)
    fraction = selected_background_flux / float(raw_flux) if raw_flux > 0 else np.nan
    return background_per_pixel, corrected_flux, fraction, method


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


def _calibrate_integrated_flux_df(df: pd.DataFrame, field: str) -> pd.DataFrame:
    calibration = INTEGRATED_SPECTRA_CALIBRATIONS.get(field)
    if calibration is None or df.empty:
        return df

    out = df.copy()
    for sn_index, columns in INTEGRATED_SPECTRA_SN_COLUMNS.items():
        scale = float(calibration[sn_index])
        for column in columns:
            if column not in out.columns:
                continue
            out[column] = pd.to_numeric(out[column], errors="coerce") / scale
    return out


def _read_integrated_flux_ascii(field: str) -> pd.DataFrame:
    ascii_path = paths.fit_int_spec_ascii(field)
    if not ascii_path.exists():
        return pd.DataFrame()
    int_flux_df = pd.read_csv(ascii_path, sep=r"\s+", comment="#")
    int_flux_df = int_flux_df.sort_values(by="id").reset_index(drop=True)
    return _calibrate_integrated_flux_df(int_flux_df, field)


def load_flux_maps(field: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    folder = paths.calibrated_field_map_dir(field)
    maps = {}
    for line_name, base in LINE_MAP_BASENAMES.items():
        flux_data, _ = _read_fits(folder / f"M33{field}-{base}.fits")
        err_data, _ = _read_fits(folder / f"M33{field}-{base}-err.fits")
        maps[line_name] = (flux_data, err_data)
    return maps


def load_region_inputs(
    field: str,
    max_zoi_pc: int,
    *,
    peaks_csv: str | Path | None = None,
    zoi_fits: str | Path | None = None,
    boundary_fits: str | Path | None = None,
    boundary_metrics_csv: str | Path | None = None,
) -> dict[str, object]:
    peaks_df = read_catalog(peaks_csv or paths.final_peaks_csv(field), comment="#")
    zoi_map = read_fits_data(zoi_fits or paths.zoi_fits(field, max_zoi_pc))
    boundary_map = read_fits_data(boundary_fits or paths.boundary_fits(field, max_zoi_pc))
    boundary_metrics_df = read_catalog(boundary_metrics_csv or paths.boundary_metrics_csv(field, max_zoi_pc))
    return {
        "peaks_df": peaks_df,
        "zoi_map": zoi_map,
        "boundary_map": boundary_map,
        "boundary_metrics_df": boundary_metrics_df,
    }


def load_integrated_flux_table(field: str, expected_len: int) -> pd.DataFrame:
    int_flux_df = _read_integrated_flux_ascii(field)
    if not int_flux_df.empty:
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
    int_flux_df = _read_integrated_flux_ascii(field)
    if int_flux_df.empty:
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
        if dig_mode == "dig_subtracted":
            out[f"F_{line_name}_sum"] = out[dig_flux]
            out[f"F_{line_name}_e_sum"] = out[base_err]
            # Use observed/no-DIG line detections for S/N cuts. DIG subtraction can
            # drive valid bright regions negative in weak lines.
            out[f"SNR_{line_name}_sum"] = out[base_snr]
        else:
            out[f"F_{line_name}_sum"] = out[base_flux]
            out[f"F_{line_name}_e_sum"] = out[base_err]
            out[f"SNR_{line_name}_sum"] = out[base_snr]
    out["dig_mode"] = dig_mode
    return out


def _mask_negative_flux_triplet(df: pd.DataFrame, flux_col: str, err_col: str, snr_col: str) -> pd.DataFrame:
    out = df.copy()
    if flux_col not in out.columns:
        return out
    flux = pd.to_numeric(out[flux_col], errors="coerce")
    neg_mask = flux < 0
    if neg_mask.any():
        out.loc[neg_mask, flux_col] = np.nan
        if err_col in out.columns:
            out.loc[neg_mask, err_col] = np.nan
        if snr_col in out.columns:
            out.loc[neg_mask, snr_col] = np.nan
    return out


def _mask_negative_fluxes(df: pd.DataFrame, line_names: list[str], prefixes: list[str]) -> pd.DataFrame:
    out = df.copy()
    for line_name in line_names:
        for prefix in prefixes:
            flux_col = f"F_{line_name}_{prefix}"
            err_col = f"F_{line_name}_e_{prefix}"
            snr_col = f"SNR_{line_name}_{prefix}"
            out = _mask_negative_flux_triplet(out, flux_col, err_col, snr_col)
    return out


def _positive_or_zero(value) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return max(value, 0.0)


def add_peak_pixel_fluxes(
    df: pd.DataFrame,
    maps: dict[str, tuple[np.ndarray, np.ndarray]],
    dig_mode: str,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Sample each calibrated line map at the catalogued H II-region peak."""
    if dig_mode not in {"no_dig", "dig_subtracted"}:
        raise ValueError(f"Unknown DIG mode: {dig_mode}")

    out = df.copy()
    x = pd.to_numeric(out[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=float)
    integer_xy = np.isfinite(x) & np.isfinite(y) & np.isclose(x, np.rint(x)) & np.isclose(y, np.rint(y))
    x_pix = np.where(integer_xy, np.rint(x), -1).astype(int)
    y_pix = np.where(integer_xy, np.rint(y), -1).astype(int)
    out["peak_pixel_valid"] = False

    for line_name, (flux_map, err_map) in maps.items():
        in_bounds = (
            integer_xy
            & (x_pix >= 0)
            & (x_pix < flux_map.shape[1])
            & (y_pix >= 0)
            & (y_pix < flux_map.shape[0])
        )
        raw_flux = np.full(len(out), np.nan, dtype=float)
        raw_err = np.full(len(out), np.nan, dtype=float)
        raw_flux[in_bounds] = np.asarray(flux_map, dtype=float)[y_pix[in_bounds], x_pix[in_bounds]]
        raw_err[in_bounds] = np.asarray(err_map, dtype=float)[y_pix[in_bounds], x_pix[in_bounds]]
        background_col = f"{line_name}_dig_median"
        background = (
            pd.to_numeric(out[background_col], errors="coerce").to_numpy(dtype=float)
            if background_col in out.columns
            else np.full(len(out), np.nan, dtype=float)
        )
        digsub_flux = raw_flux - background
        digsub_flux = np.where(np.isfinite(digsub_flux) & (digsub_flux >= 0), digsub_flux, raw_flux)
        active_flux = digsub_flux if dig_mode == "dig_subtracted" else raw_flux
        with np.errstate(divide="ignore", invalid="ignore"):
            observed_snr = raw_flux / raw_err

        out[f"F_{line_name}_peak_nodig"] = raw_flux
        out[f"F_{line_name}_e_peak_nodig"] = raw_err
        out[f"F_{line_name}_peak_digsub"] = digsub_flux
        out[f"F_{line_name}_peak"] = active_flux
        out[f"F_{line_name}_e_peak"] = raw_err
        out[f"SNR_{line_name}_peak"] = observed_snr
        out["peak_pixel_valid"] |= in_bounds & np.isfinite(raw_flux) & np.isfinite(raw_err)

    return out


def _compute_dig_background_catalog(
    boundary_map: np.ndarray,
    zoi_map: np.ndarray,
    maps: dict[str, tuple[np.ndarray, np.ndarray]],
    config: PhotometryConfig,
    boundary_metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    labels = np.unique(boundary_map[np.isfinite(boundary_map)])
    labels = labels[labels > 0].astype(int)
    background_maps = {
        line_name: _smoothed_masked_background_map(flux, boundary_map, config)
        for line_name, (flux, _err) in maps.items()
    }
    rows = []
    for region_id in labels:
        region_mask = boundary_map == region_id
        background_mask, annulus_info = _scaled_local_background_mask(zoi_map, boundary_map, region_id, config)
        row = {
            "region_id": region_id,
            "npix_region": int(np.sum(region_mask)),
            "npix_dig_annulus": int(np.sum(background_mask)),
            **annulus_info,
        }
        for line_name, (flux, err) in maps.items():
            background_map, background_info = background_maps[line_name]
            region_background_values = np.asarray(background_map, dtype=float)[region_mask]
            finite_background = region_background_values[np.isfinite(region_background_values)]
            model_background_flux = float(np.sum(finite_background)) if finite_background.size else 0.0
            raw_stats = integrated_flux_and_snr(
                flux=flux,
                err=err,
                region_mask=region_mask,
                background_per_pixel=np.nan,
                clip_negative_after_bg=False,
            )
            background_per_pixel, flux_bgsub, background_fraction, background_method = _select_capped_dig_background(
                raw_stats["F_raw"],
                raw_stats["npix"],
                model_background_flux,
                config.dig_max_subtraction_fraction,
            )
            row[f"F_{line_name}_sum_nodig"] = raw_stats["F_raw"]
            row[f"F_{line_name}_e_sum_nodig"] = raw_stats["sigma_F"]
            row[f"SNR_{line_name}_sum_nodig"] = raw_stats["SNR_raw"]
            row[f"{line_name}_dig_median"] = background_per_pixel
            row[f"{line_name}_dig_local_median"] = background_per_pixel
            row[f"{line_name}_dig_method"] = background_method
            row[f"{line_name}_dig_mad"] = (
                float(1.4826 * np.nanmedian(np.abs(finite_background - np.nanmedian(finite_background))))
                if finite_background.size
                else np.nan
            )
            row[f"{line_name}_dig_clip_upper"] = np.nan
            row[f"{line_name}_dig_npix"] = background_info["dig_background_npix"]
            row[f"{line_name}_dig_npix_used"] = int(finite_background.size)
            row[f"{line_name}_dig_model_sum"] = model_background_flux
            row[f"{line_name}_dig_subtracted_sum"] = background_per_pixel * raw_stats["npix"]
            row[f"{line_name}_dig_fraction_raw"] = background_fraction
            row[f"{line_name}_dig_background_smooth_sigma_px"] = background_info["dig_background_smooth_sigma_px"]
            row[f"{line_name}_dig_max_subtraction_fraction"] = config.dig_max_subtraction_fraction
            row[f"F_{line_name}_sum_digsub"] = flux_bgsub
        rows.append(row)
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_summed_map_flux_catalog(
    field: str,
    max_zoi_pc: int,
    config: PhotometryConfig,
    dig_mode: str = "no_dig",
    *,
    peaks_csv: str | Path | None = None,
    zoi_fits: str | Path | None = None,
    boundary_fits: str | Path | None = None,
    boundary_metrics_csv: str | Path | None = None,
) -> pd.DataFrame:
    region_inputs = load_region_inputs(
        field,
        max_zoi_pc,
        peaks_csv=peaks_csv,
        zoi_fits=zoi_fits,
        boundary_fits=boundary_fits,
        boundary_metrics_csv=boundary_metrics_csv,
    )
    peaks_df = region_inputs["peaks_df"]
    zoi_map = region_inputs["zoi_map"]
    boundary_map = region_inputs["boundary_map"]
    boundary_metrics_df = region_inputs["boundary_metrics_df"]
    maps = load_flux_maps(field)
    flux_catalog_df = _compute_dig_background_catalog(
        boundary_map,
        zoi_map,
        maps,
        config,
        boundary_metrics_df=boundary_metrics_df,
    )
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
    merged_df = add_peak_pixel_fluxes(merged_df, maps, dig_mode=dig_mode)
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

    dig_df = _compute_dig_background_catalog(
        boundary_map,
        zoi_map,
        maps,
        config,
        boundary_metrics_df=boundary_metrics_df,
    )
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
        npix_region = merged_flux_df["npix_region"].to_numpy(dtype=float)
        raw_flux = merged_flux_df[raw_flux_col].to_numpy(dtype=float)
        if f"{line_name}_dig_model_sum" in merged_flux_df.columns:
            model_background_flux = merged_flux_df[f"{line_name}_dig_model_sum"].to_numpy(dtype=float)
        else:
            background_per_pixel = merged_flux_df[f"{line_name}_dig_median"].to_numpy(dtype=float)
            model_background_flux = background_per_pixel * npix_region
        selected_background = np.full(len(merged_flux_df), np.nan, dtype=float)
        dig_flux = np.full(len(merged_flux_df), np.nan, dtype=float)
        dig_fraction = np.full(len(merged_flux_df), np.nan, dtype=float)
        dig_method = np.full(len(merged_flux_df), "", dtype=object)
        for i in range(len(merged_flux_df)):
            selected_background[i], dig_flux[i], dig_fraction[i], dig_method[i] = _select_capped_dig_background(
                raw_flux[i],
                npix_region[i],
                model_background_flux[i],
                config.dig_max_subtraction_fraction,
            )
        merged_flux_df[f"{line_name}_dig_median"] = selected_background
        merged_flux_df[f"{line_name}_dig_local_median"] = selected_background
        merged_flux_df[f"{line_name}_dig_method"] = dig_method
        merged_flux_df[f"{line_name}_dig_subtracted_sum"] = selected_background * npix_region
        merged_flux_df[f"{line_name}_dig_fraction_raw"] = dig_fraction
        merged_flux_df[f"F_{line_name}_sum_digsub"] = dig_flux
    merged_flux_df = _mask_negative_fluxes(
        merged_flux_df,
        list(maps),
        prefixes=["sum", "sum_nodig"],
    )
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
    merged_df = add_peak_pixel_fluxes(merged_df, maps, dig_mode=dig_mode)
    return merged_df


def build_field_flux_catalog(
    field: str,
    max_zoi_pc: int,
    config: PhotometryConfig,
    method: str = "summed_map",
    dig_mode: str = "no_dig",
    **region_input_paths,
) -> pd.DataFrame:
    if method == "summed_map":
        return build_summed_map_flux_catalog(field, max_zoi_pc, config, dig_mode=dig_mode, **region_input_paths)
    if region_input_paths:
        raise ValueError("Custom region input paths are only supported for summed_map flux catalogs.")
    if method == "integrated_spectrum":
        return build_integrated_spectrum_flux_catalog(field, max_zoi_pc, config, dig_mode=dig_mode)
    raise ValueError(f"Unknown flux catalog method: {method}")


def write_field_flux_catalog(field: str, df: pd.DataFrame, method: str = "summed_map", dig_mode: str = "no_dig") -> Path:
    out_path = paths.flux_catalog_csv(field, method=method, dig_mode=dig_mode)
    write_catalog(df, out_path)
    return out_path
