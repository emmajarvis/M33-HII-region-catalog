from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d, map_coordinates, zoom


def distance_tag(distance_mpc: float) -> str:
    if float(distance_mpc).is_integer():
        return str(int(distance_mpc))
    return str(distance_mpc)


def pixel_size_pc(pixscale_arcsec_per_pix: float, distance_mpc: float) -> float:
    radians_per_pixel = np.deg2rad(pixscale_arcsec_per_pix / 3600.0)
    return float(distance_mpc) * 1e6 * radians_per_pixel


def find_peak_columns(df: pd.DataFrame) -> tuple[str, str]:
    possible_x = ["x", "X", "x_pix", "XPIX", "xpeak", "x_peak", "xpic", "col", "i"]
    possible_y = ["y", "Y", "y_pix", "YPIX", "ypeak", "y_peak", "ypic", "row", "j"]

    xcol = next((c for c in possible_x if c in df.columns), None)
    ycol = next((c for c in possible_y if c in df.columns), None)

    if xcol is None or ycol is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            xcol, ycol = numeric_cols[0], numeric_cols[1]

    if xcol is None or ycol is None:
        raise ValueError(f"Could not identify x/y columns. Found: {list(df.columns)}")

    return xcol, ycol


def load_peak_catalog_for_distance(
    distance_mpc: float,
    *,
    d_orig: float = 0.84,
    orig_file: str | os.PathLike[str] = "CATALOGS/final_peaks_NW.csv",
    peaks_dir: str | os.PathLike[str] = "/Users/emmajarvis/Documents/SIGNALS/M33/NW-peak-identification/resolution/",
    peak_filename_template: str = "{dist}Mpc_list_pix_A_PicL16_sizd1_facbg1.20_snoi2.00_bg1_30_size1.csv",
) -> tuple[pd.DataFrame, str, str, str]:
    if np.isclose(distance_mpc, d_orig):
        peak_file = Path(orig_file)
    else:
        peak_file = Path(peaks_dir) / peak_filename_template.format(dist=distance_tag(distance_mpc))

    if not peak_file.exists():
        raise FileNotFoundError(f"Peak file not found: {peak_file}")

    df = pd.read_csv(peak_file, comment="#")
    xcol, ycol = find_peak_columns(df)
    return df, xcol, ycol, str(peak_file)


def degrade_map_to_distance(
    image: np.ndarray,
    d_orig: float,
    d_target: float,
    pixscale_orig: float,
    fwhm_psf_orig: float,
    fwhm_instr: float | None = None,
    plot: bool = True,
    ax=None,
    overlay_peaks: bool = True,
    orig_file: str | os.PathLike[str] = "CATALOGS/final_peaks_NW.csv",
    peaks_dir: str | os.PathLike[str] = "/Users/emmajarvis/Documents/SIGNALS/M33/NW-peak-identification/resolution/",
    peak_filename_template: str = "{dist}Mpc_list_pix_A_PicL16_sizd1_facbg1.20_snoi2.00_bg1_30_size1.csv",
    peak_marker: str = "o",
    peak_color: str = "white",
    peak_size: float = 40,
    peak_edgecolor: str = "black",
    peak_linewidth: float = 0.8,
):
    scale = 1e14
    image_scaled = image * scale

    mask = np.isnan(image_scaled)
    image_filled = np.nan_to_num(image_scaled, nan=0.0)

    scale_factor = d_orig / d_target
    pixscale_new = pixscale_orig / scale_factor

    resampled = zoom(image_filled, scale_factor, order=1)
    mask_resampled = zoom(mask.astype(float), scale_factor, order=0) > 0.5

    fwhm_target = fwhm_psf_orig if fwhm_instr is None else fwhm_instr

    def fwhm_to_sigma(fwhm_arcsec: float, pixscale_arcsec: float) -> float:
        return (fwhm_arcsec / pixscale_arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    sigma_curr = fwhm_to_sigma(fwhm_psf_orig, pixscale_new)
    sigma_targ = fwhm_to_sigma(fwhm_target, pixscale_new)
    sigma_kernel = np.sqrt(max(0.0, sigma_targ**2 - sigma_curr**2))
    kernel = Gaussian2DKernel(sigma_kernel if sigma_kernel > 0 else 1e-8)

    degraded = convolve_fft(
        resampled,
        kernel,
        boundary="fill",
        fill_value=0.0,
        preserve_nan=False,
        normalize_kernel=True,
        mask=mask_resampled,
    )

    degraded[mask_resampled] = np.nan
    degraded /= scale

    spatial_resolution_pc = (fwhm_target * d_target * 1e6) / 206265.0

    peaks = None
    n_peaks = 0
    if overlay_peaks:
        try:
            peaks, xcol, ycol, _ = load_peak_catalog_for_distance(
                d_target,
                d_orig=d_orig,
                orig_file=orig_file,
                peaks_dir=peaks_dir,
                peak_filename_template=peak_filename_template,
            )
            n_peaks = len(peaks)
        except Exception as exc:
            print(f"Could not read peak file for {d_target} Mpc: {exc}")

    if plot:
        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 12))
            created_fig = True

        finite = degraded[np.isfinite(degraded) & (degraded > 0)]
        if finite.size:
            vmin = np.nanpercentile(finite, 20)
            vmax = np.nanpercentile(finite, 99)
            norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, np.nextafter(0, 1)), vmax=max(vmax, vmin * 1.01))
        else:
            norm = None

        im = ax.imshow(degraded, origin="lower", cmap="rainbow", norm=norm)
        ax.text(
            0.03,
            0.97,
            f"Dist: {d_target} Mpc\nRes: {spatial_resolution_pc:.1f} pc\n# Peaks: {n_peaks}",
            color="k",
            fontsize=20,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        if peaks is not None and len(peaks) > 0:
            ax.scatter(
                peaks[xcol].values,
                peaks[ycol].values,
                s=peak_size,
                marker=peak_marker,
                facecolors=peak_color,
                edgecolors=peak_edgecolor,
                linewidths=peak_linewidth,
                zorder=5,
            )

        ax.set_xticks([])
        ax.set_yticks([])

        if created_fig:
            plt.colorbar(im, ax=ax)
            plt.tight_layout()

    return degraded, pixscale_new, spatial_resolution_pc


def scaled_roi_from_param_section(
    field: str,
    shape: tuple[int, int],
    *,
    sizebox: int = 1,
    scale_factor: float = 1.0,
    repo_root: str | os.PathLike[str] = ".",
) -> tuple[int, int, int, int]:
    roi_path = Path(repo_root) / "peak_files" / "param_sections" / f"param_section_{sizebox}_{field}.txt"
    ny, nx = shape

    if roi_path.exists():
        xi, xf, yi, yf = np.loadtxt(roi_path, unpack=True)
        xi = int(np.floor(float(xi) * scale_factor))
        xf = int(np.ceil(float(xf) * scale_factor))
        yi = int(np.floor(float(yi) * scale_factor))
        yf = int(np.ceil(float(yf) * scale_factor))
    else:
        margin = max(1, int(np.ceil(50 * scale_factor)))
        xi, xf, yi, yf = margin, nx - margin, margin, ny - margin

    xi = max(0, min(xi, nx - 1))
    xf = max(xi + 1, min(xf, nx))
    yi = max(0, min(yi, ny - 1))
    yf = max(yi + 1, min(yf, ny))
    return xi, xf, yi, yf


def build_peak_table(
    peaks_df: pd.DataFrame,
    xcol: str,
    ycol: str,
    image_shape: tuple[int, int],
    field: str,
) -> pd.DataFrame:
    ny, nx = image_shape
    work = peaks_df.copy()
    work["x"] = pd.to_numeric(work[xcol], errors="coerce")
    work["y"] = pd.to_numeric(work[ycol], errors="coerce")
    work = work.dropna(subset=["x", "y"]).reset_index(drop=True)

    inside = (
        (work["x"] >= 0)
        & (work["x"] < nx)
        & (work["y"] >= 0)
        & (work["y"] < ny)
    )
    work = work.loc[inside].reset_index(drop=True)
    work["field"] = field
    work["kind"] = work.get("kind", "peak")
    work["region_id"] = [f"{field}_{i + 1:04d}" for i in range(len(work))]
    work["zoi_center_label"] = np.arange(1, len(work) + 1)
    return work


def makezoi_legacy(
    image: np.ndarray,
    peaks_df: pd.DataFrame,
    *,
    roi: tuple[int, int, int, int],
    exponent: float,
    rmax_px: int,
) -> np.ndarray:
    xi, xf, yi, yf = roi
    zoi_all = np.full(image.shape, np.nan, dtype=float)

    xpic = peaks_df["x"].round().astype(int).to_numpy()
    ypic = peaks_df["y"].round().astype(int).to_numpy()
    labels = peaks_df["zoi_center_label"].astype(int).to_numpy()

    peak_flux = np.nan_to_num(image[ypic, xpic], nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(xi, xf):
        dx = xpic - i
        for j in range(yi, yf):
            pixel_value = image[j, i]
            if not np.isfinite(pixel_value) or pixel_value == 0:
                continue

            dy = ypic - j
            dist = np.sqrt(dx * dx + dy * dy)
            keep = np.where(dist < rmax_px)[0]
            if keep.size == 0:
                continue

            r = np.maximum(dist[keep], 1e-6)
            influence = peak_flux[keep] / np.power(r, exponent)
            if influence.size == 0 or np.all(~np.isfinite(influence)):
                continue

            winner = int(keep[int(np.nanargmax(np.nan_to_num(influence, nan=-np.inf)))])
            zoi_all[j, i] = float(labels[winner])

    zoi_clean = np.zeros_like(zoi_all, dtype=float)

    for label_value, xp, yp in zip(labels, xpic, ypic):
        region_mask = zoi_all == float(label_value)
        if not np.any(region_mask):
            continue

        comp_map, ncomp = ndimage.label(region_mask.astype(np.uint8))
        if ncomp <= 1:
            zoi_clean[region_mask] = float(label_value)
            continue

        py = int(np.clip(yp, 0, comp_map.shape[0] - 1))
        px = int(np.clip(xp, 0, comp_map.shape[1] - 1))
        keep_comp = int(comp_map[py, px])
        if keep_comp == 0:
            sizes = ndimage.sum(region_mask, comp_map, index=np.arange(1, ncomp + 1))
            keep_comp = int(np.argmax(sizes) + 1)
        zoi_clean[comp_map == keep_comp] = float(label_value)

    zoi_clean[zoi_clean == 0] = np.nan
    return zoi_clean


def robust_bg_sigma_local(img, cx, cy, rmax, frac_in=0.7, frac_out=0.95):
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ann = (rr >= frac_in * rmax) & (rr <= frac_out * rmax) & np.isfinite(img)
    vals = img[ann]
    if vals.size < 10:
        vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 0.0
    bg = float(np.nanmedian(vals))
    sigma = float(1.4826 * np.nanmedian(np.abs(vals - bg)))
    if not np.isfinite(sigma):
        sigma = float(np.nanstd(vals))
    return bg, sigma


def sample_ray_profile(sub, zsub, cx, cy, theta, rmax, dr=2.0, smooth_sigma=1.2):
    r = np.arange(0, max(dr, rmax) + dr, dr, dtype=float)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)

    inside = (
        (xs >= 0)
        & (xs <= sub.shape[1] - 1)
        & (ys >= 0)
        & (ys <= sub.shape[0] - 1)
    )
    r = r[inside]
    xs = xs[inside]
    ys = ys[inside]

    if r.size == 0:
        return np.array([0.0]), np.array([np.nan]), np.array([np.nan]), np.array([False])

    prof = map_coordinates(sub, [ys, xs], order=1, mode="nearest")
    prof_s = gaussian_filter1d(prof, sigma=smooth_sigma, mode="nearest") if smooth_sigma and len(prof) > 2 else prof

    if zsub is not None:
        zvals = map_coordinates(zsub, [ys, xs], order=0, mode="nearest")
        inmask = zvals > 0.5
        if inmask.any():
            last_idx = np.where(inmask)[0].max()
            r = r[: last_idx + 1]
            prof = prof[: last_idx + 1]
            prof_s = prof_s[: last_idx + 1]
            inmask = inmask[: last_idx + 1]
        else:
            inmask = np.zeros_like(r, dtype=bool)
    else:
        inmask = np.ones_like(r, dtype=bool)

    return r, prof, prof_s, inmask


def find_edge_on_ray(r, prof_s, bg, sigma, inmask=None, edge_frac=1.0, fallback_sig_k=1.0):
    if not np.isfinite(prof_s).any():
        return np.nan, "RAY_EMPTY"

    if inmask is not None and inmask.any():
        last_idx = np.where(inmask)[0].max()
        r = r[: last_idx + 1]
        prof_s = prof_s[: last_idx + 1]
        inmask = inmask[: last_idx + 1]

    i_peak = int(np.nanargmax(prof_s))
    height = prof_s[i_peak]
    prominence = height - bg
    lvl = bg + edge_frac * max(prominence, 0.0)
    start = min(len(r) - 2, i_peak + 1)

    for j in range(start, len(r) - 1):
        y0, y1 = prof_s[j], prof_s[j + 1]
        if np.isfinite([y0, y1]).all() and y0 >= lvl and y1 <= lvl:
            t = 0.0 if y1 == y0 else (lvl - y0) / (y1 - y0)
            return float(r[j] + t * (r[j + 1] - r[j])), "RAY_CROSS_FRAC_PROM"

    lvl2 = bg + fallback_sig_k * sigma
    for j in range(start, len(r) - 1):
        y0, y1 = prof_s[j], prof_s[j + 1]
        if np.isfinite([y0, y1]).all() and y0 >= lvl2 and y1 <= lvl2:
            t = 0.0 if y1 == y0 else (lvl2 - y0) / (y1 - y0)
            return float(r[j] + t * (r[j + 1] - r[j])), "RAY_CROSS_BG_KSIG"

    dy = np.diff(prof_s)
    for j in range(start, len(r) - 2):
        if np.isfinite([dy[j], dy[j + 1]]).all() and dy[j] < 0 and dy[j + 1] >= 0:
            return float(r[j + 1]), "RAY_LOCAL_MIN"

    if inmask is not None and inmask.any():
        return float(r[np.where(inmask)[0].max()]), "RAY_CLAMP_TO_ZOI"

    return np.nan, "RAY_FAIL"


def polygon_from_polar(cx, cy, thetas, radii):
    xs = cx + radii * np.cos(thetas)
    ys = cy + radii * np.sin(thetas)
    return np.vstack([xs, ys]).T


def rasterize_polygon_to_mask(shape, poly_xy):
    if len(poly_xy) < 3:
        return np.zeros(shape, dtype=bool)
    path = MplPath(poly_xy, closed=True)
    ys, xs = np.indices(shape)
    pts = np.vstack([xs.ravel(), ys.ravel()]).T
    return path.contains_points(pts, radius=0.0).reshape(shape)


def circular_local_stats(vals, win):
    vals = np.asarray(vals, float)
    n = len(vals)
    pad = win // 2
    extended = np.concatenate([vals[-pad:], vals, vals[:pad]])
    meds = np.empty(n)
    mads = np.empty(n)
    for i in range(n):
        w = extended[i : i + win]
        if not np.isfinite(w).any():
            meds[i] = np.nan
            mads[i] = np.nan
            continue
        med = np.nanmedian(w)
        meds[i] = med
        mads[i] = 1.4826 * np.nanmedian(np.abs(w - med))
    return meds, mads


def slope_cap(vals, limit):
    r = np.array(vals, float).copy()
    n = len(r)

    for i in range(1, n):
        if r[i] > r[i - 1] + limit:
            r[i] = r[i - 1] + limit
    if r[0] > r[-1] + limit:
        r[0] = r[-1] + limit

    for i in range(n - 2, -1, -1):
        if r[i] > r[i + 1] + limit:
            r[i] = r[i + 1] + limit

    for i in range(1, n):
        if r[i] < r[i - 1] - limit:
            r[i] = r[i - 1] - limit
    if r[0] < r[-1] - limit:
        r[0] = r[-1] - limit

    for i in range(n - 2, -1, -1):
        if r[i] < r[i + 1] - limit:
            r[i] = r[i + 1] - limit

    return r


def neighbor_spike_replace(vals, local_mad, k=2.0):
    r = np.array(vals, float).copy()
    n = len(r)
    for i in range(n):
        left = r[(i - 1) % n]
        right = r[(i + 1) % n]
        neigh_mean = 0.5 * (left + right)
        if r[i] > neigh_mean + k * local_mad[i]:
            r[i] = neigh_mean
    return r


def tiny_sg(vals, win=7, poly=2):
    if win is None or win < 3 or win % 2 == 0:
        return np.array(vals, float)
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(np.array(vals, float), window_length=win, polyorder=poly, mode="wrap")
    except Exception:
        arr = np.array(vals, float)
        pad = win // 2
        ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
        kernel = np.ones(win, float) / win
        return np.convolve(ext, kernel, mode="same")[pad:-pad]


def build_boundary_map(
    image: np.ndarray,
    zoi_label: np.ndarray,
    peaks_df: pd.DataFrame,
    *,
    pixel_scale_pc: float,
    n_theta: int = 72,
    r_bin: float = 2.0,
    smooth_sigma: float = 1.2,
    edge_frac: float = 1.0,
    fallback_sig_k: float = 1.0,
    ang_win_sectors: int | None = None,
    mad_k_clip: float = 2.2,
    slope_alpha: float = 0.15,
    slope_min_px: float = 4.0,
    spike_k_neigh: float = 2.0,
    sg_window: int = 7,
    sg_poly: int = 2,
    global_p95_cap: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    if ang_win_sectors is None:
        ang_win_sectors = max(5, (n_theta // 8) * 2 + 1)

    boundary_label = np.zeros_like(zoi_label, dtype=np.float32)
    metrics_rows = []
    region_masks = {}
    center_by_label = {}

    rmax_default = float(np.nanmax([
        1.0,
        np.nanmax(peaks_df.get("zoi_rmax_px", pd.Series([1.0], dtype=float))),
    ]))

    for _, row in peaks_df.iterrows():
        cx = float(row["x"])
        cy = float(row["y"])
        region_label = int(row["zoi_center_label"])
        region_id = str(row["region_id"])

        zoi_mask_full = zoi_label == region_label
        if not np.any(zoi_mask_full):
            metrics_rows.append(
                {
                    "region_id": region_id,
                    "field": row["field"],
                    "x_pix": cx,
                    "y_pix": cy,
                    "distance_mpc": row["distance_mpc"],
                    "zoi_center_label": region_label,
                    "bg_local": np.nan,
                    "sigma_local": np.nan,
                    "radius_p16_px": np.nan,
                    "radius_p50_px": np.nan,
                    "radius_p84_px": np.nan,
                    "radius_pc_eq": np.nan,
                    "n_pix": 0,
                    "region_flux_sum": np.nan,
                    "luminosity": np.nan,
                }
            )
            continue

        ys, xs = np.where(zoi_mask_full)
        x0 = max(0, int(xs.min()) - 2)
        x1 = min(image.shape[1], int(xs.max()) + 3)
        y0 = max(0, int(ys.min()) - 2)
        y1 = min(image.shape[0], int(ys.max()) + 3)

        sub = image[y0:y1, x0:x1]
        zsub = (zoi_label[y0:y1, x0:x1] == region_label).astype(float)
        cx_l = cx - x0
        cy_l = cy - y0

        rmax = max(rmax_default, float(np.sqrt(np.count_nonzero(zsub) / np.pi)) + 3.0)
        bg, sigma = robust_bg_sigma_local(sub, cx_l, cy_l, rmax=rmax)

        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        r_edges_raw = []
        r_zoi_rim = []

        for theta in thetas:
            r, _, prof_s, inmask = sample_ray_profile(sub, zsub, cx_l, cy_l, theta, rmax, dr=r_bin, smooth_sigma=smooth_sigma)
            rr, _ = find_edge_on_ray(r, prof_s, bg, sigma, inmask=inmask, edge_frac=edge_frac, fallback_sig_k=fallback_sig_k)
            rr = float(np.clip(rr, 0, rmax)) if np.isfinite(rr) else np.nan
            r_edges_raw.append(rr)
            if inmask is not None and inmask.any():
                r_zoi_rim.append(float(r[np.where(inmask)[0].max()]))
            else:
                r_zoi_rim.append(np.inf)

        r_edges_raw = np.asarray(r_edges_raw, float)
        r_zoi_rim = np.asarray(r_zoi_rim, float)

        med, mad = circular_local_stats(r_edges_raw, win=ang_win_sectors)
        mad_fallback = np.nanmedian(mad[np.isfinite(mad) & (mad > 0)])
        if not np.isfinite(mad_fallback) or mad_fallback <= 0:
            mad_fallback = 1.0
        mad = np.where(np.isfinite(mad) & (mad > 0), mad, mad_fallback)

        r_clip = np.minimum(np.maximum(r_edges_raw, med - mad_k_clip * mad), med + mad_k_clip * mad)
        med_r = float(np.nanmedian(r_clip[np.isfinite(r_clip)])) if np.isfinite(r_clip).any() else slope_min_px
        limit = max(slope_min_px, slope_alpha * med_r)
        r_slope = slope_cap(r_clip, limit=limit)
        r_fix = neighbor_spike_replace(r_slope, local_mad=mad, k=spike_k_neigh)
        r_sg = tiny_sg(r_fix, win=sg_window, poly=sg_poly) if sg_window and sg_window >= 3 else r_fix
        r_final = np.minimum(r_sg, r_zoi_rim)
        if global_p95_cap and np.isfinite(r_final).any():
            cap = np.nanpercentile(r_final[np.isfinite(r_final)], 95.0)
            r_final = np.minimum(r_final, cap)

        poly_local = polygon_from_polar(cx_l, cy_l, thetas, r_final)
        poly_global = poly_local.copy()
        poly_global[:, 0] += x0
        poly_global[:, 1] += y0

        boundary_mask = rasterize_polygon_to_mask(image.shape, poly_global)
        boundary_mask &= zoi_mask_full
        if not np.any(boundary_mask):
            boundary_mask = zoi_mask_full.copy()
        region_masks[region_label] = boundary_mask
        center_by_label[region_label] = (int(round(cx)), int(round(cy)))

        area_px = float(np.count_nonzero(boundary_mask))
        radius_eq_px = float(np.sqrt(area_px / np.pi)) if area_px > 0 else np.nan
        radius_eq_pc = radius_eq_px * pixel_scale_pc if np.isfinite(radius_eq_px) else np.nan

        vals = image[boundary_mask]
        vals = vals[np.isfinite(vals)]
        region_flux = float(np.nansum(vals)) if vals.size else np.nan
        luminosity = flux_to_luminosity(region_flux, row["distance_mpc"]) if np.isfinite(region_flux) else np.nan

        if np.isfinite(r_final).any():
            r16_px, r50_px, r84_px = np.nanpercentile(r_final[np.isfinite(r_final)], [16, 50, 84])
        else:
            r16_px = r50_px = r84_px = np.nan

        metrics_rows.append(
                {
                    "region_id": region_id,
                    "field": row["field"],
                    "x_pix": cx,
                    "y_pix": cy,
                    "distance_mpc": row["distance_mpc"],
                    "zoi_center_label": region_label,
                    "bg_local": bg,
                    "sigma_local": sigma,
                "radius_p16_px": float(r16_px),
                "radius_p50_px": float(r50_px),
                "radius_p84_px": float(r84_px),
                "radius_p16_pc": float(r16_px * pixel_scale_pc) if np.isfinite(r16_px) else np.nan,
                "radius_p50_pc": float(r50_px * pixel_scale_pc) if np.isfinite(r50_px) else np.nan,
                "radius_p84_pc": float(r84_px * pixel_scale_pc) if np.isfinite(r84_px) else np.nan,
                "radius_pc_eq": radius_eq_pc,
                "n_pix": int(area_px),
                "area_pc2": float(area_px * pixel_scale_pc**2),
                "region_flux_sum": region_flux,
                "luminosity": luminosity,
                "boundary_method": "ANG_MAD+SLOPE+NEIGH+SG+CLAMP",
            }
        )

    labels = sorted(region_masks.keys(), key=lambda lab: np.count_nonzero(region_masks[lab]))
    for i, lab_big in enumerate(labels):
        mask_big = region_masks[lab_big]
        for lab_small in labels[:i]:
            cx_s, cy_s = center_by_label[lab_small]
            if 0 <= cy_s < mask_big.shape[0] and 0 <= cx_s < mask_big.shape[1] and mask_big[cy_s, cx_s]:
                mask_big = mask_big & (~region_masks[lab_small])
        region_masks[lab_big] = mask_big

    for label_value, mask in region_masks.items():
        boundary_label[mask] = label_value

    metrics_df = pd.DataFrame(metrics_rows)
    area_after = {lab: float(np.count_nonzero(mask)) for lab, mask in region_masks.items()}
    metrics_df["n_pix"] = metrics_df["zoi_center_label"].map(area_after).fillna(0).astype(int)
    metrics_df["area_pc2"] = metrics_df["n_pix"].astype(float) * pixel_scale_pc**2
    metrics_df["radius_pc_eq"] = np.where(
        metrics_df["n_pix"] > 0,
        np.sqrt(metrics_df["area_pc2"] / np.pi),
        np.nan,
    )

    for idx, row in metrics_df.iterrows():
        lab = int(row["zoi_center_label"])
        mask = region_masks.get(lab)
        if mask is None or not np.any(mask):
            metrics_df.at[idx, "region_flux_sum"] = np.nan
            metrics_df.at[idx, "luminosity"] = np.nan
            continue
        vals = image[mask]
        vals = vals[np.isfinite(vals)]
        total_flux = float(np.nansum(vals)) if vals.size else np.nan
        metrics_df.at[idx, "region_flux_sum"] = total_flux
        metrics_df.at[idx, "luminosity"] = flux_to_luminosity(total_flux, row["distance_mpc"]) if np.isfinite(total_flux) else np.nan

    return boundary_label, metrics_df


def flux_to_luminosity(flux, distance_mpc):
    mpc_to_cm = 3.085677581e24
    d_cm = distance_mpc * mpc_to_cm
    return 4.0 * np.pi * d_cm**2 * flux


def build_region_catalogs_for_distances(
    image: np.ndarray,
    d_targets: list[float],
    *,
    field: str,
    d_orig: float = 0.84,
    pixscale_orig: float = 0.3,
    fwhm_psf_orig: float = 0.8,
    fwhm_instr: float | None = None,
    orig_file: str | os.PathLike[str] = "CATALOGS/final_peaks_NW.csv",
    peaks_dir: str | os.PathLike[str] = "/Users/emmajarvis/Documents/SIGNALS/M33/NW-peak-identification/resolution/",
    peak_filename_template: str = "{dist}Mpc_list_pix_A_PicL16_sizd1_facbg1.20_snoi2.00_bg1_30_size1.csv",
    sizebox: int = 1,
    zoi_maxrad_pc: float = 100.0,
    influence_exp: float = 2.0,
    save_region_catalogs: bool = False,
    output_dir: str | os.PathLike[str] = "region_catalogs",
) -> tuple[dict[float, pd.DataFrame], dict[float, np.ndarray], dict[float, np.ndarray], dict[float, np.ndarray]]:
    if save_region_catalogs:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    catalogs = {}
    boundary_maps = {}
    degraded_maps = {}
    zoi_maps = {}

    for d_target in d_targets:
        degraded, pixscale_new, spatial_resolution_pc = degrade_map_to_distance(
            image=image,
            d_orig=d_orig,
            d_target=d_target,
            pixscale_orig=pixscale_orig,
            fwhm_psf_orig=fwhm_psf_orig,
            fwhm_instr=fwhm_instr,
            plot=False,
            overlay_peaks=False,
            orig_file=orig_file,
            peaks_dir=peaks_dir,
            peak_filename_template=peak_filename_template,
        )

        peaks_raw, xcol, ycol, peak_file = load_peak_catalog_for_distance(
            d_target,
            d_orig=d_orig,
            orig_file=orig_file,
            peaks_dir=peaks_dir,
            peak_filename_template=peak_filename_template,
        )
        peaks_df = build_peak_table(peaks_raw, xcol, ycol, degraded.shape, field)
        peaks_df["distance_mpc"] = d_target
        peaks_df["pixscale_arcsec_per_pix"] = pixscale_new
        peaks_df["spatial_resolution_pc"] = spatial_resolution_pc
        peaks_df["peak_file"] = peak_file

        scale_factor = d_orig / d_target
        roi = scaled_roi_from_param_section(field, degraded.shape, sizebox=sizebox, scale_factor=scale_factor)
        pix_pc = pixel_size_pc(pixscale_new, d_target)
        zoi_rmax_px = int(zoi_maxrad_pc / pix_pc) + 1
        peaks_df["zoi_rmax_px"] = float(zoi_rmax_px)

        zoi_map = makezoi_legacy(
            degraded,
            peaks_df,
            roi=roi,
            exponent=influence_exp,
            rmax_px=zoi_rmax_px,
        )

        boundary_map, catalog = build_boundary_map(
            degraded,
            zoi_map,
            peaks_df,
            pixel_scale_pc=pix_pc,
        )

        catalog["distance_mpc"] = d_target
        catalog["pixscale_arcsec_per_pix"] = pixscale_new
        catalog["spatial_resolution_pc"] = spatial_resolution_pc
        catalog["peak_file"] = peak_file

        catalogs[d_target] = catalog
        boundary_maps[d_target] = boundary_map
        degraded_maps[d_target] = degraded
        zoi_maps[d_target] = zoi_map

        if save_region_catalogs:
            tag = distance_tag(d_target)
            catalog.to_csv(Path(output_dir) / f"region_catalog_{field}_{tag}Mpc.csv", index=False)
            fits.PrimaryHDU(boundary_map.astype(np.float32)).writeto(
                Path(output_dir) / f"boundary_map_{field}_{tag}Mpc.fits",
                overwrite=True,
            )
            fits.PrimaryHDU(np.nan_to_num(zoi_map, nan=0.0).astype(np.float32)).writeto(
                Path(output_dir) / f"zoi_map_{field}_{tag}Mpc.fits",
                overwrite=True,
            )

        print(f"{d_target} Mpc: built {len(catalog)} HII regions with notebook 4-5 pipeline")

    return catalogs, boundary_maps, degraded_maps, zoi_maps


def plot_luminosity_functions_from_catalogs(catalogs, bins=20, density=False, histtype="step", linewidth=2, logL_range=None, alpha=0.5):
    distances = list(catalogs.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(distances)))

    all_logl = []
    for distance in distances:
        lum = catalogs[distance]["luminosity"].to_numpy()
        good = np.isfinite(lum) & (lum > 0)
        if np.any(good):
            all_logl.append(np.log10(lum[good]))

    if not all_logl:
        raise ValueError("No valid luminosities found.")

    all_logl = np.concatenate(all_logl)
    if logL_range is None:
        xmin = np.floor(all_logl.min() * 2) / 2
        xmax = np.ceil(all_logl.max() * 2) / 2
    else:
        xmin, xmax = logL_range

    bin_edges = np.linspace(xmin, xmax, bins + 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    for color, distance in zip(colors, distances):
        lum = catalogs[distance]["luminosity"].to_numpy()
        good = np.isfinite(lum) & (lum > 0)
        if not np.any(good):
            continue
        ax.hist(
            np.log10(lum[good]),
            bins=bin_edges,
            histtype=histtype,
            linewidth=linewidth,
            density=density,
            color=color,
            label=f"{distance} Mpc",
            alpha=alpha,
        )

    ax.set_xlabel(r"log$_{10}$(Luminosity)")
    ax.set_ylabel("Density" if density else "Number of regions")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_region_size_histograms(catalogs, bins=20, density=False, histtype="step", linewidth=2, logR=False, alpha=0.5):
    distances = list(catalogs.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(distances)))
    fig, ax = plt.subplots(figsize=(8, 6))

    all_vals = []
    for distance in distances:
        radii = catalogs[distance]["radius_pc_eq"].to_numpy()
        good = np.isfinite(radii) & (radii > 0)
        if np.any(good):
            vals = np.log10(radii[good]) if logR else radii[good]
            all_vals.append(vals)

    if not all_vals:
        raise ValueError("No valid region radii found.")

    all_vals = np.concatenate(all_vals)
    bin_edges = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), bins + 1)

    for color, distance in zip(colors, distances):
        radii = catalogs[distance]["radius_pc_eq"].to_numpy()
        good = np.isfinite(radii) & (radii > 0)
        if not np.any(good):
            continue
        vals = np.log10(radii[good]) if logR else radii[good]
        ax.hist(vals, bins=bin_edges, histtype=histtype, linewidth=linewidth, density=density, color=color, label=f"{distance} Mpc", alpha=alpha)

    ax.set_xlabel(r"log$_{10}$(R$_{\rm eq}$/pc)" if logR else r"R$_{\rm eq}$ (pc)")
    ax.set_ylabel("Density" if density else "Number of regions")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_luminosity_vs_radius(catalogs, logx=True, logy=True, alpha=0.7, s=30):
    distances = list(catalogs.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(distances)))
    fig, ax = plt.subplots(figsize=(8, 6))

    for color, distance in zip(colors, distances):
        df = catalogs[distance]
        good = (
            np.isfinite(df["radius_pc_eq"].to_numpy())
            & (df["radius_pc_eq"].to_numpy() > 0)
            & np.isfinite(df["luminosity"].to_numpy())
            & (df["luminosity"].to_numpy() > 0)
        )
        if not np.any(good):
            continue
        ax.scatter(df.loc[good, "radius_pc_eq"], df.loc[good, "luminosity"], s=s, alpha=alpha, color=color, label=f"{distance} Mpc")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"R$_{\rm eq}$ (pc)")
    ax.set_ylabel("Luminosity")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_segmentation_example(image, label_map, catalog, title="Segmentation"):
    fig, ax = plt.subplots(figsize=(8, 8))
    finite = image[np.isfinite(image) & (image > 0)]
    if finite.size:
        vmin = np.nanpercentile(finite, 5)
        vmax = np.nanpercentile(finite, 99.5)
        norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, np.nextafter(0, 1)), vmax=max(vmax, vmin * 1.01))
    else:
        norm = None
    ax.imshow(image, origin="lower", cmap="gray", norm=norm)
    ax.contour(np.nan_to_num(label_map, nan=0.0), levels=[0.5], colors=["cyan"], linewidths=0.8, origin="lower")
    if {"x_pix", "y_pix"}.issubset(catalog.columns):
        ax.scatter(catalog["x_pix"], catalog["y_pix"], marker="+", s=18, linewidths=0.8, color="k")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig, ax
