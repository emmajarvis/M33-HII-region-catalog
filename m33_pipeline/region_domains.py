from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter as clock
from collections import Counter

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d, map_coordinates


@dataclass
class DomainParams:
    max_zoi_pc: int = 100
    assume_distance_mpc: float = 0.84
    n_theta: int = 72
    r_bin: float = 2.0
    smooth_sigma: float = 1.2
    edge_frac: float = 1.0
    fallback_sig_k: float = 2.5
    rmax_default: float = 100.0
    ang_win_sectors: int = 19
    mad_k_clip: float = 2.2
    slope_alpha: float = 0.3
    slope_min_px: float = 4.0
    spike_k_neigh: float = 2.0
    sg_window: int = 7
    sg_poly: int = 2
    global_p95_cap: bool = True
    influence_exp: float = 2.0
    sizebox: int = 1
    min_domain_valid_pixels: int = 20
    max_prune_iter: int = 10


def load_first_2d_image_from_fits(path: str | Path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 2:
                return hdu.data.astype(float), hdu.header
    raise RuntimeError(f"No 2D image found in: {path}")


def estimate_pc_per_pixel_from_wcs(header, distance_mpc: float):
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        w = WCS(header)
        scales_deg = proj_plane_pixel_scales(w)
        deg_per_px = float(np.nanmean(scales_deg))
        if not np.isfinite(deg_per_px) or deg_per_px <= 0:
            return None
        rad_per_px = deg_per_px * (np.pi / 180.0)
        return float(distance_mpc) * 1e6 * rad_per_px
    except Exception:
        return None


def label_at_xy(label_map, x, y):
    yi, xi = int(round(float(y))), int(round(float(x)))
    if 0 <= yi < label_map.shape[0] and 0 <= xi < label_map.shape[1]:
        v = label_map[yi, xi]
        if np.isfinite(v) and v > 0:
            return int(round(float(v)))
    return 0


def peak_intensity(x, y, image):
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    ix = np.clip(ix, 0, image.shape[1] - 1)
    iy = np.clip(iy, 0, image.shape[0] - 1)
    return float(image[iy, ix])


def makezoi_legacy(image, xpic, ypic, field: str, out_dir: str | Path, header_info=None, exponent=2.0, roi=None, rmax_px=50, filep="ZoI_map"):
    if roi is None:
        raise ValueError("ROI must be provided as (xi, xf, yi, yf)")
    xi, xf, yi, yf = roi
    h, w = image.shape
    npic = len(xpic)

    zoi_all = np.zeros_like(image, dtype=float) / 0.0
    t0 = clock()
    a_peaks = np.nan_to_num(image[ypic, xpic], nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(xi, xf):
        distx = xpic - i
        for j in range(yi, yf):
            if image[j, i] == 0 or not np.isfinite(image[j, i]):
                continue
            disty = ypic - j
            dist = np.sqrt(distx * distx + disty * disty)
            keep = np.where(dist < rmax_px)[0]
            if keep.size == 0:
                continue
            r = np.maximum(dist[keep], 1e-6)
            influence = a_peaks[keep] / np.power(r, exponent)
            winner = int(keep[int(np.argmax(np.nan_to_num(influence, nan=-np.inf)))])
            zoi_all[j, i] = float(winner)
    valid = np.isfinite(zoi_all)
    zoi_all[valid] = zoi_all[valid] + 1.0

    zoi_clean = np.zeros_like(zoi_all, dtype=float)
    for ipeak in range(npic):
        label = float(ipeak + 1)
        ry, rx = np.where(zoi_all == label)
        if ry.size == 0:
            continue
        y0, y1 = max(ry.min() - 1, 0), min(ry.max() + 2, h)
        x0, x1 = max(rx.min() - 1, 0), min(rx.max() + 2, w)
        reg = np.copy(zoi_all[y0:y1, x0:x1])
        reg[reg != label] = 0.0
        mask = reg == label
        lab, ncomp = ndimage.label(mask.astype(np.uint8))
        if ncomp <= 1:
            zoi_clean[y0:y1, x0:x1] = np.where(mask, label, zoi_clean[y0:y1, x0:x1])
        else:
            py = int(ypic[ipeak]) - y0
            px = int(xpic[ipeak]) - x0
            if py < 0 or px < 0 or py >= lab.shape[0] or px >= lab.shape[1]:
                comp_sizes = ndimage.sum(mask, lab, index=np.arange(1, ncomp + 1))
                keep_comp = int(np.argmax(comp_sizes) + 1)
            else:
                keep_comp = int(lab[py, px])
            keep = lab == keep_comp
            zoi_clean[y0:y1, x0:x1] = np.where(keep, label, zoi_clean[y0:y1, x0:x1])

    zoi_clean[zoi_clean == 0] = np.nan
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoi_fits_path = out_dir / f"{filep}_{field}.fits"
    if header_info is not None:
        fits.writeto(zoi_fits_path, zoi_clean, header_info, overwrite=True)
    else:
        fits.writeto(zoi_fits_path, zoi_clean, overwrite=True)
    print(f"   Saved ZOI map: {zoi_fits_path} ({clock() - t0:.2f} s)")
    return zoi_clean


def make_contours_legacy(label_map, field: str, out_dir: str | Path, header_info=None, filep="ContZoI_map"):
    h, w = label_map.shape
    outfile = np.zeros((h, w), dtype=float)
    labels = [int(v) for v in np.unique(label_map[np.isfinite(label_map)]) if v > 0]
    for label in labels:
        cy, cx = np.where(label_map == label)
        if cy.size == 0:
            continue
        y0, y1 = max(cy.min() - 1, 0), min(cy.max() + 2, h)
        x0, x1 = max(cx.min() - 1, 0), min(cx.max() + 2, w)
        reg = np.copy(label_map[y0:y1, x0:x1])
        reg[reg != label] = 0.0
        mask = reg == label
        edge = np.logical_xor(mask, ndimage.binary_erosion(mask))
        cont = np.zeros_like(reg, dtype=float)
        cont[edge] = label + 1.0
        outfile[y0:y1, x0:x1] = np.where(cont > 0, 1.0, outfile[y0:y1, x0:x1])
    outfile[outfile == 0] = np.nan
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filep}_{field}.fits"
    if header_info is not None:
        fits.writeto(out_path, outfile, header_info, overwrite=True)
    else:
        fits.writeto(out_path, outfile, overwrite=True)
    print(f"   Saved contours: {out_path}")
    return outfile


def robust_bg_sigma_local(img, cx, cy, rmax, frac_in=0.7, frac_out=0.95):
    yy, xx = np.indices(img.shape)
    rr = np.hypot(xx - cx, yy - cy)
    rin, rout = frac_in * rmax, frac_out * rmax
    mask = (rr >= rin) & (rr < rout) & np.isfinite(img)
    vals = img[mask]
    bg = np.nanmedian(vals) if vals.size else 0.0
    sig = mad_std(vals - bg, ignore_nan=True) if vals.size else 0.0
    return float(bg), float(sig)


def sample_ray_profile(img, mask_in, cx, cy, theta, rmax, dr=1.0, smooth_sigma=1.2):
    r = np.arange(0, rmax + dr, dr, dtype=float)
    ys = cy + r * np.sin(theta)
    xs = cx + r * np.cos(theta)
    prof = map_coordinates(img, [ys, xs], order=1, mode="nearest")
    prof = np.where(np.isfinite(prof), prof, np.nan)
    prof_s = gaussian_filter1d(prof, sigma=max(1e-6, smooth_sigma), mode="nearest", truncate=3.0)
    inmask = None
    if mask_in is not None:
        lab = map_coordinates(mask_in, [ys, xs], order=0, mode="nearest")
        inmask = np.isfinite(lab) & (lab > 0.5)
        if inmask.any():
            last_idx = np.where(inmask)[0].max()
            r = r[: last_idx + 1]
            prof = prof[: last_idx + 1]
            prof_s = prof_s[: last_idx + 1]
            inmask = inmask[: last_idx + 1]
        else:
            r = r[:1]
            prof = prof[:1] * np.nan
            prof_s = prof_s[:1] * np.nan
            inmask = inmask[:1]
    return r, prof, prof_s, inmask


def find_edge_on_ray(r, prof_s, bg, sigma, inmask=None, edge_frac=0.5, fallback_sig_k=1.0):
    if not np.isfinite(prof_s).any():
        return np.nan, "RAY_EMPTY"
    if inmask is not None and inmask.any():
        last_idx = np.where(inmask)[0].max()
        r = r[: last_idx + 1]
        prof_s = prof_s[: last_idx + 1]
        inmask = inmask[: last_idx + 1]
    i_peak = int(np.nanargmax(prof_s))
    height = prof_s[i_peak]
    prom = height - bg
    lvl = bg + edge_frac * max(prom, 0.0)
    start = min(len(r) - 2, i_peak + 1)
    for j in range(start, len(r) - 1):
        y0, y1 = prof_s[j], prof_s[j + 1]
        if np.isfinite([y0, y1]).all() and (y0 >= lvl) and (y1 <= lvl):
            t = 0.0 if y1 == y0 else (lvl - y0) / (y1 - y0)
            return float(r[j] + t * (r[j + 1] - r[j])), "RAY_CROSS_FRAC_PROM"
    lvl2 = bg + fallback_sig_k * sigma
    for j in range(start, len(r) - 1):
        y0, y1 = prof_s[j], prof_s[j + 1]
        if np.isfinite([y0, y1]).all() and (y0 >= lvl2) and (y1 <= lvl2):
            t = 0.0 if y1 == y0 else (lvl2 - y0) / (y1 - y0)
            return float(r[j] + t * (r[j + 1] - r[j])), "RAY_CROSS_BG_KSIG"
    dy = np.diff(prof_s)
    for j in range(start, len(r) - 2):
        if np.isfinite([dy[j], dy[j + 1]]).all() and (dy[j] < 0) and (dy[j + 1] >= 0):
            return float(r[j + 1]), "RAY_LOCAL_MIN"
    if inmask is not None and inmask.any():
        last_idx = np.where(inmask)[0].max()
        return float(r[last_idx]), "RAY_CLAMP_TO_ZOI"
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
    x = np.concatenate([vals[-pad:], vals, vals[:pad]])
    meds = np.empty(n)
    mads = np.empty(n)
    for i in range(n):
        w = x[i: i + win]
        med = np.nanmedian(w)
        meds[i] = med
        mads[i] = 1.4826 * np.nanmedian(np.abs(w - med))
    return meds, mads


def slope_cap(vals, L):
    r = np.array(vals, float).copy()
    n = len(r)
    for i in range(1, n):
        if r[i] > r[i - 1] + L:
            r[i] = r[i - 1] + L
    if r[0] > r[-1] + L:
        r[0] = r[-1] + L
    for i in range(n - 2, -1, -1):
        if r[i] > r[i + 1] + L:
            r[i] = r[i + 1] + L
    for i in range(1, n):
        if r[i] < r[i - 1] - L:
            r[i] = r[i - 1] - L
    if r[0] < r[-1] - L:
        r[0] = r[-1] - L
    for i in range(n - 2, -1, -1):
        if r[i] < r[i + 1] - L:
            r[i] = r[i + 1] - L
    return r


def neighbor_spike_replace(vals, local_mad, k=2.0):
    r = np.array(vals, float).copy()
    n = len(r)
    for i in range(n):
        im = (i - 1) % n
        ip = (i + 1) % n
        neigh_mean = 0.5 * (r[im] + r[ip])
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
        a = np.array(vals, float)
        pad = win // 2
        x = np.concatenate([a[-pad:], a, a[:pad]])
        kern = np.ones(win, float) / float(win)
        return np.convolve(x, kern, mode="valid")


def _prepare_catalog(combined: pd.DataFrame, field: str) -> pd.DataFrame:
    required = {"x", "y"}
    if not required.issubset(combined.columns):
        raise ValueError(f"Final peak catalog must contain {required}")
    out = combined.copy()
    out["kind"] = out.get("kind", "peak")
    out["field"] = field
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["x", "y"]).reset_index(drop=True)
    if "region_number" in out.columns:
        out["region_number"] = pd.to_numeric(out["region_number"], errors="coerce").astype("Int64")
    if "region_number" not in out.columns or out["region_number"].isna().any():
        out["region_number"] = np.arange(1, len(out) + 1, dtype=int)
    out = out.sort_values("region_number").reset_index(drop=True)
    out["region_number"] = np.arange(1, len(out) + 1, dtype=int)
    out["region_id"] = [f"{field}_{int(n):04d}" for n in out["region_number"]]
    return out


def _positive_labels(label_map: np.ndarray) -> list[int]:
    labels = np.unique(np.asarray(label_map)[np.isfinite(label_map)])
    return sorted(int(v) for v in labels if int(v) > 0)


def _expected_labels(n_regions: int) -> list[int]:
    return list(range(1, int(n_regions) + 1))


def _export_final_peak_catalog(combined: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "field",
        "x",
        "y",
        "region_number",
        "region_id",
        "zoi_center_label",
        "zoi_valid_halpha_pixels",
    ]
    cols = [col for col in preferred if col in combined.columns]
    out = combined.loc[:, cols].copy()
    if "field" in out.columns:
        out["field"] = out["field"].astype(str)
    if "region_number" in out.columns:
        out["region_number"] = pd.to_numeric(out["region_number"], errors="coerce").astype(int)
    if "zoi_center_label" in out.columns:
        out["zoi_center_label"] = pd.to_numeric(out["zoi_center_label"], errors="coerce").astype(int)
    return out


def _assert_region_label_consistency(combined: pd.DataFrame, zoi_label: np.ndarray, boundary_label: np.ndarray) -> None:
    expected = _expected_labels(len(combined))
    peak_labels = sorted(int(v) for v in combined["zoi_center_label"].tolist())
    zoi_labels = _positive_labels(zoi_label)
    boundary_labels = _positive_labels(boundary_label)
    if peak_labels != expected:
        raise RuntimeError(f"Peak catalog labels are not contiguous 1..N: expected {expected[:10]}..., got {peak_labels[:10]}...")
    if zoi_labels != expected:
        raise RuntimeError(f"ZoI map labels are not contiguous 1..N: expected {expected[:10]}..., got {zoi_labels[:10]}...")
    if boundary_labels != expected:
        raise RuntimeError(f"Boundary map labels are not contiguous 1..N: expected {expected[:10]}..., got {boundary_labels[:10]}...")
    center_labels = [label_at_xy(zoi_label, x, y) for x, y in zip(combined["x"], combined["y"])]
    if center_labels != expected:
        raise RuntimeError(f"Peak centers do not land on matching ZoI labels: expected {expected[:10]}..., got {center_labels[:10]}...")


def _dedupe_peaks_by_zoi_label(combined: pd.DataFrame, zoi_label: np.ndarray, amp_map: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = combined.copy().reset_index(drop=True)
    work["zoi_center_label"] = [label_at_xy(zoi_label, x, y) for x, y in zip(work["x"], work["y"])]
    dup_mask = work["zoi_center_label"].duplicated(keep=False)
    if not dup_mask.any():
        return work, pd.DataFrame(columns=list(work.columns) + ["peak_halpha_value"])

    removed_rows = []
    keep_indices = []
    for label in sorted(int(v) for v in work.loc[work["zoi_center_label"] > 0, "zoi_center_label"].unique()):
        group = work.loc[work["zoi_center_label"] == label].copy()
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        group["peak_halpha_value"] = [peak_intensity(x, y, amp_map) for x, y in zip(group["x"], group["y"])]
        keep_idx = group["peak_halpha_value"].idxmax()
        keep_indices.append(keep_idx)
        removed_rows.append(group.loc[group.index != keep_idx].copy())

    deduped = work.loc[sorted(set(keep_indices))].copy().reset_index(drop=True)
    removed = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame(columns=list(work.columns) + ["peak_halpha_value"])
    return deduped, removed


def _load_roi(field: str, sizebox: int, shape):
    param_path = Path(f"peak_files/param_sections/param_section_{sizebox}_{field}.txt")
    if param_path.exists():
        xi, xf, yi, yf = np.loadtxt(param_path, unpack=True)
        xi, xf, yi, yf = int(xi), int(xf), int(yi), int(yf)
    else:
        matches = sorted(Path("peak_files/param_sections").glob(f"param_section_*_{field}.txt"))
        if len(matches) == 1:
            xi, xf, yi, yf = np.loadtxt(matches[0], unpack=True)
            xi, xf, yi, yf = int(xi), int(xf), int(yi), int(yf)
            print(f"[domains] warning: requested ROI file {param_path} not found; using {matches[0]}")
        elif len(matches) > 1:
            raise RuntimeError(
                f"Multiple ROI files found for {field} but requested {param_path} is missing: "
                + ", ".join(str(m) for m in matches)
            )
        else:
            ny, nx = shape
            xi, xf, yi, yf = 0, nx, 0, ny
            print(f"[domains] warning: no ROI file found for {field}; using full image extent")
    ny, nx = shape
    xi = int(max(0, min(xi, nx - 1)))
    xf = int(max(xi + 1, min(xf, nx)))
    yi = int(max(0, min(yi, ny - 1)))
    yf = int(max(yi + 1, min(yf, ny)))
    return xi, xf, yi, yf


def _rebuild_zoi_from_catalog(combined, amp_map, header_info, field: str, zoi_dir: str | Path, params: DomainParams):
    pc_per_px = estimate_pc_per_pixel_from_wcs(header_info, params.assume_distance_mpc)
    if pc_per_px is None or not np.isfinite(pc_per_px) or pc_per_px <= 0:
        raise RuntimeError("Could not estimate pc/pixel for ZoI regeneration.")
    roi = _load_roi(field, params.sizebox, amp_map.shape)
    zoi_rmax_px = int(params.max_zoi_pc / pc_per_px) + 1
    xpic = combined["x"].round().astype(int).to_numpy()
    ypic = combined["y"].round().astype(int).to_numpy()
    zoi_label = makezoi_legacy(
        amp_map, xpic=xpic, ypic=ypic, field=field, out_dir=zoi_dir, header_info=header_info,
        exponent=params.influence_exp, roi=roi, rmax_px=zoi_rmax_px, filep="ZoI_map",
    )
    contzoi = make_contours_legacy(zoi_label, field=field, out_dir=zoi_dir, header_info=header_info, filep="ContZoI_map")
    combined = combined.copy()
    combined["zoi_center_label"] = [label_at_xy(zoi_label, x, y) for x, y in zip(combined["x"], combined["y"])]
    return combined, zoi_label, contzoi


def _raw_halpha_map_path(field: str) -> Path:
    return Path(f"../M33-maps/M33-{field}/M33-{field}_SN3.LineMaps.map.Ha+OIII.1x1.amplitude.fits")


def compute_boundaries(combined, halpha, valid_halpha, zoi_label, field: str, pixel_scale_pc, params: DomainParams):
    boundary_label = np.zeros_like(zoi_label, dtype=np.float32)
    metrics_rows = []
    region_diagnostics = {}
    for _, row in combined.iterrows():
        kind = str(row.get("kind", "peak"))
        cx = float(row["x"])
        cy = float(row["y"])
        rid = str(row["region_id"])
        region_label = int(row.get("zoi_center_label", 0))
        if pixel_scale_pc is not None and np.isfinite(pixel_scale_pc) and pixel_scale_pc > 0:
            rmax = float(params.rmax_default) / float(pixel_scale_pc)
        else:
            rmax = float(params.rmax_default)
        x0 = max(0, int(np.floor(cx - rmax)))
        x1 = min(halpha.shape[1], int(np.ceil(cx + rmax + 1)))
        y0 = max(0, int(np.floor(cy - rmax)))
        y1 = min(halpha.shape[0], int(np.ceil(cy + rmax + 1)))
        sub = halpha[y0:y1, x0:x1]
        cx_l = cx - x0
        cy_l = cy - y0
        bg, sigma = robust_bg_sigma_local(sub, cx_l, cy_l, rmax=rmax)
        zsub = None
        if region_label > 0:
            zsub = (zoi_label[y0:y1, x0:x1] == region_label).astype(float)
        thetas = np.linspace(0, 2 * np.pi, params.n_theta, endpoint=False)
        # Reset per-region outputs so no state leaks forward from a previous region.
        boundary_mask = np.zeros_like(zoi_label, dtype=bool)
        r_final = np.full(params.n_theta, np.nan, dtype=float)
        edge_flag_counts: Counter[str] = Counter()
        if kind == "ring" and np.isfinite(row.get("ring_radius_px", np.nan)):
            r_final = np.array([float(row["ring_radius_px"])] * params.n_theta, dtype=float)
        else:
            r_edges_raw = []
            r_zoi_rim = []
            r_profiles_for_plot = []
            edge_flags = []
            for th in thetas:
                r, _prof, prof_s, inmask = sample_ray_profile(sub, zsub, cx_l, cy_l, th, rmax, dr=params.r_bin, smooth_sigma=params.smooth_sigma)
                rr, edge_flag = find_edge_on_ray(r, prof_s, bg, sigma, inmask=inmask, edge_frac=params.edge_frac, fallback_sig_k=params.fallback_sig_k)
                rr = float(np.clip(rr, 0, rmax)) if np.isfinite(rr) else np.nan
                r_edges_raw.append(rr)
                edge_flags.append(edge_flag)
                if inmask is not None and inmask.any():
                    r_zoi_rim.append(float(r[np.where(inmask)[0].max()]))
                else:
                    r_zoi_rim.append(np.inf)
                r_profiles_for_plot.append((r, prof_s))
            r_edges_raw = np.array(r_edges_raw, float)
            r_zoi_rim = np.array(r_zoi_rim, float)
            edge_flag_counts = Counter(edge_flags)
            region_diagnostics[rid] = {
                "sub": sub,
                "cx_l": cx_l,
                "cy_l": cy_l,
                "x0": x0,
                "y0": y0,
                "thetas": thetas,
                "r_profiles": r_profiles_for_plot,
                "r_edges_raw": r_edges_raw.copy(),
                "bg": bg,
                "sigma": sigma,
                "region_label": region_label,
                "edge_flags": edge_flags,
                "edge_flag_counts": dict(edge_flag_counts),
                "rmax_px_search": float(rmax),
            }
            med, mad = circular_local_stats(r_edges_raw, win=params.ang_win_sectors)
            mad_fallback = np.nanmedian(mad[np.isfinite(mad) & (mad > 0)])
            if not np.isfinite(mad_fallback) or mad_fallback <= 0:
                mad_fallback = 1.0
            mad = np.where(np.isfinite(mad) & (mad > 0), mad, mad_fallback)
            r_clip = np.minimum(np.maximum(r_edges_raw, med - params.mad_k_clip * mad), med + params.mad_k_clip * mad)
            med_r = float(np.nanmedian(r_clip[np.isfinite(r_clip)])) if np.isfinite(r_clip).any() else params.slope_min_px
            limit = max(params.slope_min_px, params.slope_alpha * med_r)
            r_slope = slope_cap(r_clip, L=limit)
            r_fix = neighbor_spike_replace(r_slope, local_mad=mad, k=params.spike_k_neigh)
            r_sg = tiny_sg(r_fix, win=params.sg_window, poly=params.sg_poly) if (params.sg_window and params.sg_window >= 3) else r_fix
            r_final = np.minimum(r_sg, r_zoi_rim)
            if params.global_p95_cap and np.isfinite(r_final).any():
                cap = np.nanpercentile(r_final[np.isfinite(r_final)], 95.0)
                r_final = np.minimum(r_final, cap)
            poly_local = polygon_from_polar(cx_l, cy_l, thetas, r_final)
            poly_global = poly_local.copy()
            poly_global[:, 0] += x0
            poly_global[:, 1] += y0
            boundary_mask = rasterize_polygon_to_mask(halpha.shape, poly_global)
            if region_label > 0:
                boundary_mask &= (zoi_label == region_label)
                boundary_label[boundary_mask] = region_label
        if np.isfinite(r_final).any():
            r16_px, r50_px, r84_px = np.nanpercentile(r_final[np.isfinite(r_final)], [16, 50, 84])
        else:
            r16_px = r50_px = r84_px = np.nan
        if pixel_scale_pc is not None and np.isfinite(pixel_scale_pc):
            r16_pc = r16_px * pixel_scale_pc if np.isfinite(r16_px) else np.nan
            r50_pc = r50_px * pixel_scale_pc if np.isfinite(r50_px) else np.nan
            r84_pc = r84_px * pixel_scale_pc if np.isfinite(r84_px) else np.nan
        else:
            r16_pc = r50_pc = r84_pc = np.nan
        area_px = float(np.count_nonzero(boundary_mask))
        r_areaeq_px = float(np.sqrt(area_px / np.pi)) if area_px > 0 else np.nan
        r_areaeq_pc = r_areaeq_px * pixel_scale_pc if (np.isfinite(r_areaeq_px) and pixel_scale_pc is not None) else np.nan
        metrics_rows.append({
            "region_id": rid,
            "field": field,
            "kind": kind,
            "center_x_px": cx,
            "center_y_px": cy,
            "zoi_center_label": region_label,
            "bg_local": bg,
            "sigma_local": sigma,
            "rmax_search_px": float(rmax),
            "radius_p16_px": float(r16_px),
            "radius_p50_px": float(r50_px),
            "radius_p84_px": float(r84_px),
            "radius_p16_pc": float(r16_pc),
            "radius_p50_pc": float(r50_pc),
            "radius_p84_pc": float(r84_pc),
            "radius_areaeq_px": float(r_areaeq_px),
            "radius_areaeq_pc": float(r_areaeq_pc) if np.isfinite(r_areaeq_pc) else np.nan,
            "edge_flag_primary": max(edge_flag_counts, key=edge_flag_counts.get) if kind != "ring" and edge_flag_counts else "RING_FIXED",
            "edge_flag_frac_prom_n": int(edge_flag_counts.get("RAY_CROSS_FRAC_PROM", 0)) if kind != "ring" else 0,
            "edge_flag_bg_ksig_n": int(edge_flag_counts.get("RAY_CROSS_BG_KSIG", 0)) if kind != "ring" else 0,
            "edge_flag_local_min_n": int(edge_flag_counts.get("RAY_LOCAL_MIN", 0)) if kind != "ring" else 0,
            "edge_flag_zoi_clamp_n": int(edge_flag_counts.get("RAY_CLAMP_TO_ZOI", 0)) if kind != "ring" else 0,
            "edge_flag_fail_n": int(edge_flag_counts.get("RAY_FAIL", 0)) if kind != "ring" else 0,
            "edge_flag_empty_n": int(edge_flag_counts.get("RAY_EMPTY", 0)) if kind != "ring" else 0,
        })
    metrics_df = pd.DataFrame(metrics_rows)
    label_of_center = {}
    for _, row in combined.iterrows():
        lab = int(row.get("zoi_center_label", 0))
        if lab > 0:
            label_of_center[lab] = (int(round(float(row["x"]))), int(round(float(row["y"]))))
    labels = sorted(label_of_center.keys())
    region_masks = {lab: (boundary_label == lab) for lab in labels}

    def point_in_mask(mask, x, y):
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            return False
        return bool(mask[int(y), int(x)])

    areas = {lab: int(np.count_nonzero(region_masks[lab])) for lab in labels}
    priority = sorted(labels, key=lambda lab: areas[lab])
    for j, lab_big in enumerate(priority):
        mask_big = region_masks[lab_big]
        for lab_small in priority[:j]:
            cx_s, cy_s = label_of_center[lab_small]
            if point_in_mask(mask_big, cx_s, cy_s):
                mask_big = mask_big & (~region_masks[lab_small])
        region_masks[lab_big] = mask_big
    boundary_label[:] = 0.0
    for lab, mask in region_masks.items():
        boundary_label[mask] = lab
    new_area_px_by_label = {lab: float(np.count_nonzero(boundary_label == lab)) for lab in labels}
    metrics_df["area_px_after_carve"] = metrics_df["zoi_center_label"].map(new_area_px_by_label).astype(float)
    metrics_df["radius_areaeq_px_after_carve"] = metrics_df["area_px_after_carve"].map(lambda a: float(np.sqrt(a / np.pi)) if a > 0 else np.nan)
    if pixel_scale_pc is not None and np.isfinite(pixel_scale_pc):
        metrics_df["radius_areaeq_pc_after_carve"] = metrics_df["radius_areaeq_px_after_carve"] * pixel_scale_pc
    else:
        metrics_df["radius_areaeq_pc_after_carve"] = np.nan
    finite_halpha = np.isfinite(valid_halpha)
    valid_pixels = []
    for _, row in metrics_df.iterrows():
        lab = int(row["zoi_center_label"])
        valid_pixels.append(int(np.count_nonzero((boundary_label == lab) & finite_halpha)))
    metrics_df["valid_halpha_pixels_in_boundary"] = valid_pixels
    return boundary_label, metrics_df, region_diagnostics


def run_domains_with_final_prune(field: str, peak_csv: str | Path, halpha_fits: str | Path, zoi_fits: str | Path, contzoi_fits: str | Path, boundary_fits: str | Path, metrics_csv: str | Path, params: DomainParams):
    peak_csv = Path(peak_csv)
    zoi_fits = Path(zoi_fits)
    contzoi_fits = Path(contzoi_fits)
    boundary_fits = Path(boundary_fits)
    metrics_csv = Path(metrics_csv)
    zoi_dir = zoi_fits.parent
    halpha, halpha_hdr = load_first_2d_image_from_fits(halpha_fits)
    halpha = np.where(np.isfinite(halpha), halpha, np.nan)
    raw_halpha_path = _raw_halpha_map_path(field)
    if raw_halpha_path.exists():
        valid_halpha, _ = load_first_2d_image_from_fits(raw_halpha_path)
        valid_halpha = np.where(np.isfinite(valid_halpha), valid_halpha, np.nan)
    else:
        print(f"[domains] warning: raw Halpha map not found at {raw_halpha_path}; using boundary image for validity counts")
        valid_halpha = halpha
    pixel_scale_pc = estimate_pc_per_pixel_from_wcs(halpha_hdr, params.assume_distance_mpc)
    combined = _prepare_catalog(pd.read_csv(peak_csv), field)
    initial_region_count = int(len(combined))
    amp_map = halpha.copy()
    combined, zoi_label, contzoi_label = _rebuild_zoi_from_catalog(combined, amp_map, halpha_hdr, field, zoi_dir, params)
    removed_duplicate_zoi = 0
    removed_small_boundary = 0
    prune_iterations: list[dict[str, int]] = []

    for dedupe_iter in range(1, params.max_prune_iter + 1):
        deduped, removed = _dedupe_peaks_by_zoi_label(combined, zoi_label, amp_map)
        n_removed = int(len(removed))
        if n_removed == 0:
            break
        removed_duplicate_zoi += n_removed
        print(f"[domains] dedupe iteration {dedupe_iter}: removed {n_removed} peaks sharing the same ZoI center label")
        preview_cols = [c for c in ["region_id", "region_number", "x", "y", "zoi_center_label", "peak_halpha_value"] if c in removed.columns]
        if preview_cols:
            print(removed.loc[:, preview_cols].sort_values("zoi_center_label").to_string(index=False))
        combined = deduped.copy().reset_index(drop=True)
        combined["region_number"] = np.arange(1, len(combined) + 1, dtype=int)
        combined["region_id"] = [f"{field}_{int(n):04d}" for n in combined["region_number"]]
        combined, zoi_label, contzoi_label = _rebuild_zoi_from_catalog(combined, amp_map, halpha_hdr, field, zoi_dir, params)
    else:
        raise RuntimeError("ZoI duplicate-label pruning during domain setup did not converge.")

    for iteration in range(1, params.max_prune_iter + 1):
        print(f"[domains] iteration {iteration}: computing boundaries for {len(combined)} regions")
        boundary_label, metrics_df, region_diagnostics = compute_boundaries(combined, halpha, valid_halpha, zoi_label, field, pixel_scale_pc, params)
        small_mask = metrics_df["valid_halpha_pixels_in_boundary"] < int(params.min_domain_valid_pixels)
        n_small = int(small_mask.sum())
        prune_iterations.append(
            {
                "iteration": int(iteration),
                "regions_input": int(len(combined)),
                "small_regions_removed": int(n_small),
            }
        )
        print(f"[domains] iteration {iteration}: {n_small} regions with < {params.min_domain_valid_pixels} valid Halpha pixels in final boundary")
        if n_small == 0:
            break
        removed_small_boundary += n_small
        remove_ids = set(metrics_df.loc[small_mask, "region_id"].astype(str))
        print(metrics_df.loc[small_mask, ["region_id", "zoi_center_label", "valid_halpha_pixels_in_boundary"]].to_string(index=False))
        combined = combined.loc[~combined["region_id"].astype(str).isin(remove_ids)].copy().reset_index(drop=True)
        if combined.empty:
            raise RuntimeError("Domain-size pruning removed all regions.")
        combined["region_number"] = np.arange(1, len(combined) + 1, dtype=int)
        combined["region_id"] = [f"{field}_{int(n):04d}" for n in combined["region_number"]]
        combined, zoi_label, contzoi_label = _rebuild_zoi_from_catalog(combined, amp_map, halpha_hdr, field, zoi_dir, params)
    else:
        raise RuntimeError("Domain-size pruning did not converge.")

    combined["zoi_valid_halpha_pixels"] = [
        int(np.count_nonzero((zoi_label == int(row["zoi_center_label"])) & np.isfinite(valid_halpha)))
        for _, row in combined.iterrows()
    ]
    _assert_region_label_consistency(combined, zoi_label, boundary_label)
    combined_export = _export_final_peak_catalog(combined)
    combined_export.to_csv(peak_csv, index=False)
    fits.PrimaryHDU(boundary_label.astype(np.float32)).writeto(boundary_fits, overwrite=True)
    metrics_df.to_csv(metrics_csv, index=False)
    cont_map = make_contours_legacy(boundary_label, field=field, out_dir=boundary_fits.parent, header_info=halpha_hdr, filep="ContDomain_map")
    print(f"[domains] wrote updated final peaks: {peak_csv}")
    print(f"[domains] wrote boundary map: {boundary_fits}")
    print(f"[domains] wrote metrics table: {metrics_csv}")
    return {
        "combined": combined_export,
        "halpha": halpha,
        "halpha_hdr": halpha_hdr,
        "zoi_label": zoi_label,
        "contzoi_label": contzoi_label,
        "boundary_label": boundary_label,
        "metrics_df": metrics_df,
        "region_diagnostics": region_diagnostics,
        "cont_map": cont_map,
        "pixel_scale_pc": pixel_scale_pc,
        "prune_summary": {
            "initial_regions": int(initial_region_count),
            "removed_duplicate_zoi": int(removed_duplicate_zoi),
            "removed_small_boundary": int(removed_small_boundary),
            "final_regions": int(len(combined)),
            "min_domain_valid_pixels": int(params.min_domain_valid_pixels),
            "iterations": prune_iterations,
        },
    }
