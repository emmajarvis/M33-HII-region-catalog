from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/m33_peak_completeness_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/m33_peak_completeness_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp

from .injections import InjectionConfig, inject_gaussians, load_template_catalog, sample_injections
from .peak_detection import adopted_params_for_field, load_detection_inputs, repo_root, run_peak_detection


NUMERIC_BIAS_COLUMNS = [
    "injected_radius_px",
    "injected_radius_pc",
    "pc_per_px",
    "gaussian_3sigma_flux_fraction",
    "gaussian_3sigma_unit_sum",
    "sigma_px",
    "fwhm_px",
    "morphology_radius_px",
    "ring_radius_px",
    "ring_width_px",
    "component_separation_px",
    "n_components",
    "n_detected_peaks_in_region",
    "F_Halpha_sum_template",
    "F_OIII5007_sum_template",
    "L_Ha_sum",
    "L_Ha_sum_dered",
    "log_L_Ha_sum",
    "log_L_Ha_sum_dered",
    "log_oiii_ha",
    "oiii_ha",
    "log_ha_oiii",
    "ha_oiii",
    "ha_peak_amp",
    "oiii_peak_amp",
    "total_peak_amp",
    "surface_brightness_ha",
    "local_amp_background",
    "local_threshold",
    "local_noise",
    "injected_peak_snr_estimate",
    "threshold_contrast_estimate",
    "nearest_existing_peak_distance_px",
    "log_NII_Ha_sum_dered",
    "log_OIII_Hb_sum_dered",
    "log_SII_Ha_sum_dered",
    "Z_N2_Brazzini2024",
    "Z_O3N2_Brazzini2024",
    "Z_N2S2Halpha_Brazzini2024",
    "Z_R23_Maiolino2008",
    "logU_KK04",
    "ne_SII_cm3",
    "R_gal_kpc",
]

CATEGORICAL_BIAS_COLUMNS = [
    "sampling_mode",
    "morphology",
    "BPT_class_sum_dered",
]

LOG_DISPLAY_COLUMNS = {
    "F_Halpha_sum_template",
    "F_OIII5007_sum_template",
    "L_Ha_sum",
    "L_Ha_sum_dered",
    "oiii_ha",
    "ha_peak_amp",
    "oiii_peak_amp",
    "total_peak_amp",
    "surface_brightness_ha",
    "local_amp_background",
    "local_threshold",
    "local_noise",
    "ne_SII_cm3",
    "ha_oiii",
}

BINNED_DEPENDENCE_COLUMNS = [
    "log_L_Ha_sum",
    "surface_brightness_ha",
    "total_peak_amp",
    "injected_radius_px",
    "injected_radius_pc",
    "log_oiii_ha",
    "log_ha_oiii",
    "local_amp_background",
    "local_threshold",
    "nearest_existing_peak_distance_px",
]

COMPLETENESS_LEVELS = (0.5, 0.8, 0.9)


def read_existing_peaks(field: str) -> pd.DataFrame:
    path = repo_root() / "final_peak_info" / f"latest_peaks_{field}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["field", "x", "y"])
    return pd.read_csv(path, comment="#")


def match_recovered_sources(
    injections: pd.DataFrame,
    detections: pd.DataFrame,
    *,
    match_radius_px: float,
) -> pd.DataFrame:
    out = injections.copy()
    out["recovered"] = False
    out["n_detected_peaks_in_region"] = 0
    out["multiple_peaks_detected"] = False
    out["detected_peak_xs_in_region"] = ""
    out["detected_peak_ys_in_region"] = ""
    out["x_det"] = np.nan
    out["y_det"] = np.nan
    out["match_distance_px"] = np.nan
    out["peak_amp_det"] = np.nan
    out["peak_snr_det"] = np.nan
    out["threshold_excess_det"] = np.nan

    if detections.empty:
        return out

    det_xy = detections[["x", "y"]].to_numpy(dtype=float)
    for idx, row in out.iterrows():
        distances = np.hypot(det_xy[:, 0] - row["x_inj"], det_xy[:, 1] - row["y_inj"])
        region_radius = row.get("morphology_radius_px", np.nan)
        if not np.isfinite(region_radius):
            region_radius = match_radius_px
        count_radius = max(float(match_radius_px), float(region_radius))
        in_region = distances <= count_radius
        out.loc[idx, "n_detected_peaks_in_region"] = int(in_region.sum())
        out.loc[idx, "multiple_peaks_detected"] = bool(in_region.sum() >= 2)
        if in_region.any():
            region_dets = detections.loc[in_region, ["x", "y"]].round(3)
            out.loc[idx, "detected_peak_xs_in_region"] = ";".join(region_dets["x"].astype(str).tolist())
            out.loc[idx, "detected_peak_ys_in_region"] = ";".join(region_dets["y"].astype(str).tolist())
        if not in_region.any():
            continue
        order = np.argsort(distances)
        for det_idx in order:
            if distances[det_idx] <= count_radius:
                det = detections.iloc[int(det_idx)]
                out.loc[idx, "recovered"] = True
                out.loc[idx, "x_det"] = det["x"]
                out.loc[idx, "y_det"] = det["y"]
                out.loc[idx, "match_distance_px"] = distances[det_idx]
                out.loc[idx, "peak_amp_det"] = det.get("peak_amp", np.nan)
                out.loc[idx, "peak_snr_det"] = det.get("peak_snr", np.nan)
                out.loc[idx, "threshold_excess_det"] = det.get("threshold_excess", np.nan)
                break
    return out


def filter_baseline_detections(detections: pd.DataFrame, baseline: pd.DataFrame, radius_px: float = 2.0) -> pd.DataFrame:
    """Remove peaks already present in the non-injected map."""
    if detections.empty or baseline.empty:
        return detections
    base_xy = baseline[["x", "y"]].to_numpy(dtype=float)
    keep = []
    for det in detections[["x", "y"]].to_numpy(dtype=float):
        distances = np.hypot(base_xy[:, 0] - det[0], base_xy[:, 1] - det[1])
        keep.append(bool(np.all(distances > radius_px)))
    return detections.loc[np.array(keep, dtype=bool)].reset_index(drop=True)


def annotate_injection_environment(
    injections: pd.DataFrame,
    amp: np.ndarray,
    err: np.ndarray,
    threshold: np.ndarray,
    existing_peaks: pd.DataFrame,
) -> pd.DataFrame:
    out = injections.copy()
    ny, nx = amp.shape
    x_idx = np.clip(np.rint(out["x_inj"]).astype(int), 0, nx - 1)
    y_idx = np.clip(np.rint(out["y_inj"]).astype(int), 0, ny - 1)
    out["local_amp_background"] = amp[y_idx, x_idx]
    out["local_threshold"] = threshold[y_idx, x_idx]
    out["local_noise"] = err[y_idx, x_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["injected_peak_snr_estimate"] = out["total_peak_amp"] / out["local_noise"]
        out["threshold_contrast_estimate"] = out["total_peak_amp"] / out["local_threshold"]

    if not existing_peaks.empty and {"x", "y"} <= set(existing_peaks.columns):
        peak_xy = existing_peaks[["x", "y"]].to_numpy(dtype=float)
        distances = []
        for x, y in out[["x_inj", "y_inj"]].to_numpy(dtype=float):
            distances.append(float(np.nanmin(np.hypot(peak_xy[:, 0] - x, peak_xy[:, 1] - y))))
        out["nearest_existing_peak_distance_px"] = distances
    else:
        out["nearest_existing_peak_distance_px"] = np.nan
    return out


def summarize_recovery(recovered: pd.DataFrame) -> pd.DataFrame:
    grouped = recovered.groupby(["field", "trial"], dropna=False)
    rows = []
    for (field, trial), group in grouped:
        rows.append(
            {
                "field": field,
                "trial": trial,
                "n_injected": int(len(group)),
                "n_recovered": int(group["recovered"].sum()),
                "recovery_fraction": float(group["recovered"].mean()),
                "n_multiple_peak_regions": int(group["multiple_peaks_detected"].sum()) if "multiple_peaks_detected" in group else 0,
                "multiple_peak_fraction": float(group["multiple_peaks_detected"].mean()) if "multiple_peaks_detected" in group else 0.0,
                "median_log_oiii_ha_recovered": float(group.loc[group["recovered"], "log_oiii_ha"].median()),
                "median_log_oiii_ha_missed": float(group.loc[~group["recovered"], "log_oiii_ha"].median()),
                "median_surface_brightness_recovered": float(group.loc[group["recovered"], "surface_brightness_ha"].median()),
                "median_surface_brightness_missed": float(group.loc[~group["recovered"], "surface_brightness_ha"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall_recovery(recovered: pd.DataFrame) -> pd.DataFrame:
    if recovered.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in recovered.groupby(["sampling_mode", "morphology"], dropna=False):
        sampling_mode, morphology = keys
        n_injected = int(len(group))
        n_recovered = int(group["recovered"].sum())
        rows.append(
            {
                "sampling_mode": sampling_mode,
                "morphology": morphology,
                "n_injected": n_injected,
                "n_recovered": n_recovered,
                "n_missed": n_injected - n_recovered,
                "recovery_fraction": float(n_recovered / n_injected) if n_injected else np.nan,
                "recovery_percent": float(100.0 * n_recovered / n_injected) if n_injected else np.nan,
                "mean_detected_peaks_per_injected_region": float(group["n_detected_peaks_in_region"].mean())
                if "n_detected_peaks_in_region" in group
                else np.nan,
                "multiple_peak_fraction": float(group["multiple_peaks_detected"].mean()) if "multiple_peaks_detected" in group else np.nan,
                "median_match_distance_px": float(group.loc[group["recovered"], "match_distance_px"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_field_recovery_total(recovered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (field, morphology), group in recovered.groupby(["field", "morphology"], dropna=False):
        n_injected = int(len(group))
        n_recovered = int(group["recovered"].sum())
        rows.append(
            {
                "field": field,
                "morphology": morphology,
                "n_injected": n_injected,
                "n_recovered": n_recovered,
                "n_missed": n_injected - n_recovered,
                "recovery_fraction": float(n_recovered / n_injected) if n_injected else np.nan,
                "recovery_percent": float(100.0 * n_recovered / n_injected) if n_injected else np.nan,
                "mean_detected_peaks_per_injected_region": float(group["n_detected_peaks_in_region"].mean()),
                "multiple_peak_fraction": float(group["multiple_peaks_detected"].mean()),
                "median_match_distance_px": float(group.loc[group["recovered"], "match_distance_px"].median()),
            }
        )
    return pd.DataFrame(rows)


def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def summarize_numeric_bias(recovered: pd.DataFrame, *, include_trial: bool = False) -> pd.DataFrame:
    rows = []
    group_cols = ["field"]
    if include_trial and "trial" in recovered:
        group_cols.append("trial")

    for keys, group in recovered.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        rec = group[group["recovered"]]
        miss = group[~group["recovered"]]
        for column in NUMERIC_BIAS_COLUMNS:
            if column not in group:
                continue
            rec_values = _finite_values(rec, column)
            miss_values = _finite_values(miss, column)
            if rec_values.size == 0 and miss_values.size == 0:
                continue
            ks_pvalue = np.nan
            if rec_values.size >= 2 and miss_values.size >= 2:
                ks_pvalue = float(ks_2samp(rec_values, miss_values, nan_policy="omit").pvalue)
            rec_median = float(np.nanmedian(rec_values)) if rec_values.size else np.nan
            miss_median = float(np.nanmedian(miss_values)) if miss_values.size else np.nan
            rows.append(
                {
                    **key_values,
                    "property": column,
                    "n_recovered_finite": int(rec_values.size),
                    "n_missed_finite": int(miss_values.size),
                    "recovered_median": rec_median,
                    "missed_median": miss_median,
                    "missed_minus_recovered_median": float(miss_median - rec_median)
                    if np.isfinite(rec_median) and np.isfinite(miss_median)
                    else np.nan,
                    "missed_to_recovered_median_ratio": float(miss_median / rec_median)
                    if np.isfinite(rec_median) and np.isfinite(miss_median) and rec_median != 0
                    else np.nan,
                    "recovered_mean": float(np.nanmean(rec_values)) if rec_values.size else np.nan,
                    "missed_mean": float(np.nanmean(miss_values)) if miss_values.size else np.nan,
                    "recovered_p16": float(np.nanpercentile(rec_values, 16)) if rec_values.size else np.nan,
                    "recovered_p84": float(np.nanpercentile(rec_values, 84)) if rec_values.size else np.nan,
                    "missed_p16": float(np.nanpercentile(miss_values, 16)) if miss_values.size else np.nan,
                    "missed_p84": float(np.nanpercentile(miss_values, 84)) if miss_values.size else np.nan,
                    "ks_pvalue_recovered_vs_missed": ks_pvalue,
                }
            )
    return pd.DataFrame(rows)


def summarize_categorical_bias(recovered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, group in recovered.groupby("field", dropna=False):
        for column in CATEGORICAL_BIAS_COLUMNS:
            if column not in group:
                continue
            values = group[column].fillna("missing").astype(str)
            for value in sorted(values.unique()):
                mask = values == value
                n_injected = int(mask.sum())
                n_recovered = int((group.loc[mask, "recovered"]).sum())
                rows.append(
                    {
                        "field": field,
                        "property": column,
                        "value": value,
                        "n_injected": n_injected,
                        "n_recovered": n_recovered,
                        "n_missed": n_injected - n_recovered,
                        "recovery_fraction": float(n_recovered / n_injected) if n_injected else np.nan,
                        "injected_fraction": float(n_injected / len(group)) if len(group) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_morphology_recovery(recovered: pd.DataFrame) -> pd.DataFrame:
    if "morphology" not in recovered:
        return pd.DataFrame()
    rows = []
    for (field, morphology), group in recovered.groupby(["field", "morphology"], dropna=False):
        n_injected = int(len(group))
        n_recovered = int(group["recovered"].sum())
        rows.append(
            {
                "field": field,
                "morphology": morphology,
                "n_injected": n_injected,
                "n_recovered": n_recovered,
                "n_missed": n_injected - n_recovered,
                "recovery_fraction": float(n_recovered / n_injected) if n_injected else np.nan,
                "n_multiple_peak_regions": int(group["multiple_peaks_detected"].sum()),
                "multiple_peak_fraction_all": float(group["multiple_peaks_detected"].mean()),
                "multiple_peak_fraction_recovered": float(group.loc[group["recovered"], "multiple_peaks_detected"].mean())
                if n_recovered
                else np.nan,
                "median_detected_peaks_in_region": float(group["n_detected_peaks_in_region"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_overlap_groups(recovered: pd.DataFrame) -> pd.DataFrame:
    if "overlap_group_id" not in recovered:
        return pd.DataFrame()
    work = recovered[recovered["overlap_group_id"].fillna("").astype(str) != ""]
    if work.empty:
        return pd.DataFrame()
    rows = []
    for (field, trial, group_id), group in work.groupby(["field", "trial", "overlap_group_id"], dropna=False):
        matched = group.loc[group["recovered"], ["x_det", "y_det"]].dropna().drop_duplicates()
        n_unique_matched = int(len(matched))
        rows.append(
            {
                "field": field,
                "trial": trial,
                "overlap_group_id": group_id,
                "n_members": int(len(group)),
                "n_members_recovered": int(group["recovered"].sum()),
                "n_detected_peaks_near_members": int(group["n_detected_peaks_in_region"].sum()),
                "n_unique_matched_peaks": n_unique_matched,
                "all_members_recovered": bool(group["recovered"].all()),
                "any_member_recovered": bool(group["recovered"].any()),
                "members_merged_to_one_peak": bool(group["recovered"].sum() >= 2 and n_unique_matched == 1),
                "members_resolved_as_multiple_peaks": bool(n_unique_matched >= 2),
                "median_component_separation_px": float(group["component_separation_px"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_binned_dependence(recovered: pd.DataFrame, columns: list[str] | None = None, n_bins: int = 8) -> pd.DataFrame:
    columns = columns or BINNED_DEPENDENCE_COLUMNS
    rows = []
    for column in columns:
        if column not in recovered:
            continue
        values = pd.to_numeric(recovered[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.nunique() < 2:
            continue
        try:
            bins = pd.qcut(finite, q=min(n_bins, finite.nunique()), duplicates="drop")
        except ValueError:
            continue
        work = recovered.loc[finite.index].copy()
        work["bin"] = bins.astype(str)
        work["bin_left"] = [interval.left for interval in bins]
        work["bin_right"] = [interval.right for interval in bins]
        for (field, morphology, bin_label), group in work.groupby(["field", "morphology", "bin"], dropna=False):
            n_injected = int(len(group))
            n_recovered = int(group["recovered"].sum())
            rows.append(
                {
                    "property": column,
                    "field": field,
                    "morphology": morphology,
                    "bin": bin_label,
                    "bin_left": float(group["bin_left"].iloc[0]),
                    "bin_right": float(group["bin_right"].iloc[0]),
                    "bin_center": float(0.5 * (group["bin_left"].iloc[0] + group["bin_right"].iloc[0])),
                    "n_injected": n_injected,
                    "n_recovered": n_recovered,
                    "n_missed": n_injected - n_recovered,
                    "recovery_fraction": float(n_recovered / n_injected) if n_injected else np.nan,
                }
            )
        for (morphology, bin_label), group in work.groupby(["morphology", "bin"], dropna=False):
            n_injected = int(len(group))
            n_recovered = int(group["recovered"].sum())
            rows.append(
                {
                    "property": column,
                    "field": "ALL",
                    "morphology": morphology,
                    "bin": bin_label,
                    "bin_left": float(group["bin_left"].iloc[0]),
                    "bin_right": float(group["bin_right"].iloc[0]),
                    "bin_center": float(0.5 * (group["bin_left"].iloc[0] + group["bin_right"].iloc[0])),
                    "n_injected": n_injected,
                    "n_recovered": n_recovered,
                    "n_missed": n_injected - n_recovered,
                    "recovery_fraction": float(n_recovered / n_injected) if n_injected else np.nan,
                }
            )
    return pd.DataFrame(rows)


def estimate_luminosity_completeness_limits(recovered: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    if "log_L_Ha_sum" not in recovered:
        return pd.DataFrame()
    rows = []

    def add_rows(label_field: str, morphology: str, group: pd.DataFrame) -> None:
        values = pd.to_numeric(group["log_L_Ha_sum"], errors="coerce")
        finite_group = group.loc[np.isfinite(values)].copy()
        if finite_group.empty or finite_group["log_L_Ha_sum"].nunique() < 2:
            return
        try:
            finite_group["luminosity_bin"] = pd.qcut(
                finite_group["log_L_Ha_sum"],
                q=min(n_bins, finite_group["log_L_Ha_sum"].nunique()),
                duplicates="drop",
            )
        except ValueError:
            return
        bin_summary = (
            finite_group.groupby("luminosity_bin", observed=False)
            .agg(
                bin_left=("log_L_Ha_sum", "min"),
                bin_right=("log_L_Ha_sum", "max"),
                bin_center=("log_L_Ha_sum", "median"),
                n_injected=("recovered", "size"),
                recovery_fraction=("recovered", "mean"),
            )
            .reset_index(drop=True)
            .sort_values("bin_center")
        )
        for level in COMPLETENESS_LEVELS:
            above = bin_summary[bin_summary["recovery_fraction"] >= level]
            limit = float(above["bin_center"].iloc[0]) if not above.empty else np.nan
            rows.append(
                {
                    "field": label_field,
                    "morphology": morphology,
                    "completeness_level": level,
                    "log_L_Ha_limit": limit,
                    "L_Ha_limit": float(10.0**limit) if np.isfinite(limit) else np.nan,
                    "method": f"first luminosity bin with recovery_fraction >= {level}",
                }
            )

    for (field, morphology), group in recovered.groupby(["field", "morphology"], dropna=False):
        add_rows(str(field), str(morphology), group)
    for morphology, group in recovered.groupby("morphology", dropna=False):
        add_rows("ALL", str(morphology), group)
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def summarize_peak_offsets(recovered: pd.DataFrame) -> pd.DataFrame:
    if "match_distance_px" not in recovered:
        return pd.DataFrame()
    rows = []
    for (field, morphology), group in recovered.groupby(["field", "morphology"], dropna=False):
        distances = pd.to_numeric(group.loc[group["recovered"], "match_distance_px"], errors="coerce")
        distances = distances[np.isfinite(distances)]
        if distances.empty:
            continue
        rows.append(
            {
                "field": field,
                "morphology": morphology,
                "n_recovered": int(len(distances)),
                "mean_match_distance_px": float(distances.mean()),
                "median_match_distance_px": float(distances.median()),
                "p16_match_distance_px": float(np.nanpercentile(distances, 16)),
                "p84_match_distance_px": float(np.nanpercentile(distances, 84)),
                "p95_match_distance_px": float(np.nanpercentile(distances, 95)),
                "fraction_within_1px": float((distances <= 1).mean()),
                "fraction_within_3px": float((distances <= 3).mean()),
                "fraction_within_5px": float((distances <= 5).mean()),
                "fraction_within_10px": float((distances <= 10).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_summary_manifest(output_dir: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "question": "What percent of injected regions get detected?",
                "result_file": "summary_overall_recovery.csv",
                "columns_to_use": "recovery_percent, recovery_fraction, n_injected, n_recovered",
            },
            {
                "question": "What percent are detected per field?",
                "result_file": "summary_recovery_by_field_total.csv",
                "columns_to_use": "field, recovery_percent, n_injected, n_recovered",
            },
            {
                "question": "Which injected properties affect recovery?",
                "result_file": "summary_property_dependence_bins.csv",
                "columns_to_use": "property, bin_left, bin_right, recovery_fraction",
            },
            {
                "question": "What is the Halpha luminosity completeness limit?",
                "result_file": "summary_luminosity_completeness_limits.csv",
                "columns_to_use": "completeness_level, log_L_Ha_limit, L_Ha_limit",
            },
            {
                "question": "How does environment affect recovery?",
                "result_file": "summary_property_dependence_bins.csv",
                "columns_to_use": "local_amp_background, local_threshold, nearest_existing_peak_distance_px rows",
            },
            {
                "question": "How close is the detected peak to the injected center?",
                "result_file": "summary_peak_offset_statistics.csv",
                "columns_to_use": "median_match_distance_px, p84_match_distance_px, fraction_within_5px",
            },
            {
                "question": "How does morphology impact detection and segmentation?",
                "result_file": "recovery_summary_by_morphology.csv",
                "columns_to_use": "morphology, recovery_fraction, multiple_peak_fraction_all, median_detected_peaks_in_region",
            },
            {
                "question": "Which morphologies are primarily missed?",
                "result_file": "recovery_summary_by_morphology.csv",
                "columns_to_use": "morphology, n_missed, recovery_fraction",
            },
            {
                "question": "For each morphology, how many peaks does one region get segmented into?",
                "result_file": "recovery_summary_by_morphology.csv",
                "columns_to_use": "median_detected_peaks_in_region, multiple_peak_fraction_all",
            },
        ]
    )
    manifest.to_csv(output_dir / "summary_results_manifest.csv", index=False)


def plot_injection_map(
    injected_amp: np.ndarray,
    recovered: pd.DataFrame,
    *,
    field: str,
    trial: int,
    output_path: Path,
    show: bool = False,
) -> None:
    display_map = injected_amp
    positive = display_map[np.isfinite(display_map) & (display_map > 0)]
    if positive.size:
        vmin = max(np.nanpercentile(positive, 2), np.nextafter(0, 1))
        vmax = np.nanpercentile(positive, 99.7)
        norm = LogNorm(vmin=vmin, vmax=vmax) if np.isfinite(vmax) and vmax > vmin else None
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(display_map, origin="lower", cmap="gray", norm=norm)
    missed = recovered[~recovered["recovered"]]
    ax.scatter(missed["x_inj"], missed["y_inj"], s=36, facecolors="none", edgecolors="tab:red", linewidths=1.2, label="missed injected center")
    rec = recovered[recovered["recovered"]]
    ax.scatter(rec["x_inj"], rec["y_inj"], s=42, facecolors="none", edgecolors="tab:green", linewidths=1.5, label="recovered injected center")
    footprint_xs: list[float] = []
    footprint_ys: list[float] = []
    if {"detected_peak_xs_in_region", "detected_peak_ys_in_region"} <= set(recovered.columns):
        for row in recovered.itertuples(index=False):
            xs = str(getattr(row, "detected_peak_xs_in_region", "") or "")
            ys = str(getattr(row, "detected_peak_ys_in_region", "") or "")
            if xs.lower() == "nan" or ys.lower() == "nan" or not xs or not ys:
                continue
            try:
                x_values = [float(value) for value in xs.split(";") if value]
                y_values = [float(value) for value in ys.split(";") if value]
            except ValueError:
                continue
            if len(x_values) != len(y_values):
                continue
            footprint_xs.extend(x_values)
            footprint_ys.extend(y_values)
    if footprint_xs:
        ax.scatter(
            footprint_xs,
            footprint_ys,
            s=36,
            marker="x",
            color="tab:orange",
            linewidths=1.2,
            label="all detected peaks in injected footprint",
            zorder=5,
        )
    matched = rec[np.isfinite(rec["x_det"]) & np.isfinite(rec["y_det"])]
    ax.scatter(
        matched["x_det"],
        matched["y_det"],
        s=55,
        marker="+",
        color="tab:blue",
        linewidths=1.6,
        label="matched detected peak",
        zorder=6,
    )
    for row in matched.itertuples(index=False):
        ax.plot(
            [row.x_inj, row.x_det],
            [row.y_inj, row.y_det],
            color="tab:blue",
            linewidth=0.6,
            alpha=0.6,
            zorder=4,
        )
    ax.set_title(f"{field} trial {trial:03d}: injected peak recovery")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def _parse_detected_peak_lists(row) -> tuple[list[float], list[float]]:
    xs = str(getattr(row, "detected_peak_xs_in_region", "") or "")
    ys = str(getattr(row, "detected_peak_ys_in_region", "") or "")
    if xs.lower() == "nan" or ys.lower() == "nan" or not xs or not ys:
        return [], []
    try:
        x_values = [float(value) for value in xs.split(";") if value]
        y_values = [float(value) for value in ys.split(";") if value]
    except ValueError:
        return [], []
    if len(x_values) != len(y_values):
        return [], []
    return x_values, y_values


def plot_multiple_peak_zooms(
    injected_amp: np.ndarray,
    recovered: pd.DataFrame,
    *,
    field: str,
    trial: int,
    output_path: Path,
    max_panels: int = 12,
) -> None:
    if "multiple_peaks_detected" not in recovered:
        return
    multi = recovered[recovered["multiple_peaks_detected"]].head(max_panels)
    if multi.empty:
        return

    ncols = min(4, len(multi))
    nrows = int(np.ceil(len(multi) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, row in zip(axes_flat, multi.itertuples(index=False)):
        radius = float(getattr(row, "morphology_radius_px", np.nan))
        if not np.isfinite(radius):
            radius = 25.0
        half_width = int(max(35, np.ceil(radius * 1.5)))
        x0 = float(row.x_inj)
        y0 = float(row.y_inj)
        ny, nx = injected_amp.shape
        x_min = max(0, int(round(x0)) - half_width)
        x_max = min(nx, int(round(x0)) + half_width + 1)
        y_min = max(0, int(round(y0)) - half_width)
        y_max = min(ny, int(round(y0)) + half_width + 1)
        cutout = injected_amp[y_min:y_max, x_min:x_max]
        positive = cutout[np.isfinite(cutout) & (cutout > 0)]
        norm = None
        if positive.size:
            vmin = max(np.nanpercentile(positive, 2), np.nextafter(0, 1))
            vmax = np.nanpercentile(positive, 99.7)
            norm = LogNorm(vmin=vmin, vmax=vmax) if np.isfinite(vmax) and vmax > vmin else None
        ax.imshow(cutout, origin="lower", cmap="gray", norm=norm, extent=[x_min, x_max, y_min, y_max])
        ax.scatter([x0], [y0], s=55, facecolors="none", edgecolors="tab:green", linewidths=1.5)
        xs, ys = _parse_detected_peak_lists(row)
        if xs:
            ax.scatter(xs, ys, s=45, marker="x", color="tab:orange", linewidths=1.4)
        if np.isfinite(getattr(row, "x_det", np.nan)) and np.isfinite(getattr(row, "y_det", np.nan)):
            ax.scatter([row.x_det], [row.y_det], s=75, marker="+", color="tab:blue", linewidths=1.8)
        ax.set_title(f"{row.injection_id} {row.morphology}\n{int(row.n_detected_peaks_in_region)} detected peaks")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    for ax in axes_flat[len(multi) :]:
        ax.set_axis_off()
    fig.suptitle(f"{field} trial {trial:03d}: injected regions with multiple detected peaks", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _safe_filename(name: str) -> str:
    return (
        name.replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _plot_values(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, str] | None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return None
    if column in LOG_DISPLAY_COLUMNS:
        positive = values > 0
        values = values[positive]
        if values.size == 0:
            return None
        return np.log10(values), f"log10({column})"
    return values, column


def _hist_bins(values: np.ndarray, max_bins: int = 35) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.nanmin(values) == np.nanmax(values):
        center = float(values[0]) if values.size else 0.0
        width = max(abs(center) * 0.05, 0.5)
        return np.array([center - width, center + width])
    return np.histogram_bin_edges(values, bins=min(max_bins, max(8, int(np.sqrt(values.size)))))


def plot_numeric_distribution(
    recovered: pd.DataFrame,
    column: str,
    *,
    output_path: Path,
    by_field: bool,
) -> None:
    available = recovered.dropna(subset=["recovered"]).copy()
    if column not in available:
        return

    if by_field:
        fields = sorted(available["field"].dropna().astype(str).unique())
        if not fields:
            return
        ncols = min(3, len(fields))
        nrows = int(np.ceil(len(fields) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, field in zip(axes_flat, fields):
            group = available[available["field"].astype(str) == field]
            plotted = _plot_values(group, column)
            if plotted is None:
                ax.set_axis_off()
                continue
            all_values, xlabel = plotted
            bins = _hist_bins(all_values)
            for flag, label, color in ((True, "recovered", "tab:green"), (False, "missed", "tab:red")):
                subset = group[group["recovered"] == flag]
                plotted_subset = _plot_values(subset, column)
                if plotted_subset is None:
                    continue
                values, _ = plotted_subset
                ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.35, color=color, density=True, label=label)
                ax.hist(values, bins=bins, histtype="step", linewidth=1.3, color=color, density=True)
            ax.set_title(field)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
        for ax in axes_flat[len(fields) :]:
            ax.set_axis_off()
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{column}: recovered vs missed injected sources by field", y=0.995)
    else:
        plotted = _plot_values(available, column)
        if plotted is None:
            return
        all_values, xlabel = plotted
        bins = _hist_bins(all_values)
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for flag, label, color in ((True, "recovered", "tab:green"), (False, "missed", "tab:red")):
            subset = available[available["recovered"] == flag]
            plotted_subset = _plot_values(subset, column)
            if plotted_subset is None:
                continue
            values, _ = plotted_subset
            ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.35, color=color, density=True, label=label)
            ax.hist(values, bins=bins, histtype="step", linewidth=1.5, color=color, density=True)
        ax.set_title(f"{column}: recovered vs missed injected sources")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_categorical_distribution(
    recovered: pd.DataFrame,
    column: str,
    *,
    output_path: Path,
    by_field: bool,
) -> None:
    if column not in recovered:
        return
    work = recovered[["field", "recovered", column]].copy()
    work[column] = work[column].fillna("missing").astype(str)
    if work.empty:
        return

    if by_field:
        fields = sorted(work["field"].dropna().astype(str).unique())
        ncols = min(3, len(fields))
        nrows = int(np.ceil(len(fields) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, field in zip(axes_flat, fields):
            group = work[work["field"].astype(str) == field]
            counts = pd.crosstab(group[column], group["recovered"]).reindex(columns=[False, True], fill_value=0)
            x = np.arange(len(counts))
            ax.bar(x, counts.get(True, 0), color="tab:green", alpha=0.75, label="recovered")
            ax.bar(x, counts.get(False, 0), bottom=counts.get(True, 0), color="tab:red", alpha=0.75, label="missed")
            ax.set_title(field)
            ax.set_xticks(x)
            ax.set_xticklabels(counts.index, rotation=35, ha="right")
            ax.set_ylabel("count")
        for ax in axes_flat[len(fields) :]:
            ax.set_axis_off()
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{column}: recovered vs missed injected sources by field", y=0.995)
    else:
        counts = pd.crosstab(work[column], work["recovered"]).reindex(columns=[False, True], fill_value=0)
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        x = np.arange(len(counts))
        ax.bar(x, counts.get(True, 0), color="tab:green", alpha=0.75, label="recovered")
        ax.bar(x, counts.get(False, 0), bottom=counts.get(True, 0), color="tab:red", alpha=0.75, label="missed")
        ax.set_title(f"{column}: recovered vs missed injected sources")
        ax.set_xticks(x)
        ax.set_xticklabels(counts.index, rotation=35, ha="right")
        ax.set_ylabel("count")
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_distribution_plots(recovered: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for column in NUMERIC_BIAS_COLUMNS:
        if column not in recovered:
            continue
        stem = _safe_filename(column)
        plot_numeric_distribution(recovered, column, output_path=output_dir / "all_fields" / f"{stem}.png", by_field=False)
        plot_numeric_distribution(recovered, column, output_path=output_dir / "by_field" / f"{stem}_by_field.png", by_field=True)
    for column in CATEGORICAL_BIAS_COLUMNS:
        if column not in recovered:
            continue
        stem = _safe_filename(column)
        plot_categorical_distribution(recovered, column, output_path=output_dir / "all_fields" / f"{stem}.png", by_field=False)
        plot_categorical_distribution(recovered, column, output_path=output_dir / "by_field" / f"{stem}_by_field.png", by_field=True)


def run_field_trial(
    field: str,
    trial: int,
    *,
    output_dir: Path,
    rng: np.random.Generator,
    template_catalog: pd.DataFrame,
    injection_config: InjectionConfig,
    match_radius_px: float,
    save_injected_fits: bool,
    save_injection_plots: bool,
    show_injection_plots: bool,
    plot_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    amp, err, threshold, params, header = load_detection_inputs(field)
    existing = read_existing_peaks(field)
    injections = sample_injections(field, amp, params, template_catalog, rng, injection_config, existing_peaks=existing)
    injections = annotate_injection_environment(injections, amp, err, threshold, existing)
    baseline_detections = run_peak_detection(amp, err, threshold, field=field, params=params)
    injected_amp = inject_gaussians(amp, injections, rng=rng, config=injection_config)
    detections = run_peak_detection(injected_amp, err, threshold, field=field, params=params)
    new_detections = filter_baseline_detections(detections, baseline_detections)

    recovered = match_recovered_sources(injections, new_detections, match_radius_px=match_radius_px)
    recovered.insert(1, "trial", trial)
    detections.insert(1, "trial", trial)
    new_detections.insert(1, "trial", trial)

    field_dir = output_dir / field / f"trial_{trial:03d}"
    field_dir.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(field_dir / "injected_sources_recovery.csv", index=False)
    detections.to_csv(field_dir / "detected_peaks.csv", index=False)
    new_detections.to_csv(field_dir / "new_detected_peaks_after_baseline_filter.csv", index=False)
    if save_injected_fits:
        fits.writeto(field_dir / f"{field}_HaOIII_amp_injected.fits", injected_amp, header=header, overwrite=True)
    if save_injection_plots or show_injection_plots:
        map_dir = plot_dir if plot_dir is not None else output_dir / "injection_maps"
        plot_injection_map(
            injected_amp,
            recovered,
            field=field,
            trial=trial,
            output_path=map_dir / f"{field}_trial_{trial:03d}_injected_recovery_map.png",
            show=show_injection_plots,
        )
        plot_multiple_peak_zooms(
            injected_amp,
            recovered,
            field=field,
            trial=trial,
            output_path=map_dir / f"{field}_trial_{trial:03d}_multiple_peak_zooms.png",
        )

    return recovered, detections


def run_trials(
    fields: list[str],
    *,
    n_trials: int,
    output_dir: Path,
    seed: int,
    injection_config: InjectionConfig,
    match_radius_px: float = 10.0,
    save_injected_fits: bool = False,
    save_injection_plots: bool = False,
    show_injection_plots: bool = False,
    plot_dir: str | Path | None = None,
    save_distribution_plots_flag: bool = False,
    distribution_plot_dir: str | Path | None = None,
    template_catalog_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(plot_dir) if plot_dir is not None else None
    distribution_plot_dir = Path(distribution_plot_dir) if distribution_plot_dir is not None else output_dir / "distribution_plots"
    template_catalog = load_template_catalog(template_catalog_path)
    all_recovered = []
    all_detections = []

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(fields) * n_trials)
    seed_idx = 0
    for field in fields:
        # Validate field early, before starting the long loop.
        amp = fits.getdata(repo_root() / "peak_files" / f"data_for_visualisation_OIII+Ha_{field}" / f"M33_{field}_HaOIII_amp_nonan.fits")
        adopted_params_for_field(field, amp.shape)
        for trial in range(1, n_trials + 1):
            rng = np.random.default_rng(child_seeds[seed_idx])
            seed_idx += 1
            recovered, detections = run_field_trial(
                field,
                trial,
                output_dir=output_dir,
                rng=rng,
                template_catalog=template_catalog,
                injection_config=injection_config,
                match_radius_px=match_radius_px,
                save_injected_fits=save_injected_fits,
                save_injection_plots=save_injection_plots,
                show_injection_plots=show_injection_plots,
                plot_dir=plot_dir,
            )
            all_recovered.append(recovered)
            all_detections.append(detections)
            print(
                f"{field} trial {trial:03d}: "
                f"{int(recovered['recovered'].sum())}/{len(recovered)} recovered, "
                f"{len(detections)} total detected peaks"
            )

    recovered_df = pd.concat(all_recovered, ignore_index=True) if all_recovered else pd.DataFrame()
    detections_df = pd.concat(all_detections, ignore_index=True) if all_detections else pd.DataFrame()
    summary_df = summarize_recovery(recovered_df) if not recovered_df.empty else pd.DataFrame()

    recovered_df.to_csv(output_dir / "all_injected_sources_recovery.csv", index=False)
    detections_df.to_csv(output_dir / "all_detected_peaks.csv", index=False)
    summary_df.to_csv(output_dir / "recovery_summary_by_field_trial.csv", index=False)
    if not summary_df.empty:
        summary_df.groupby("field", as_index=False).agg(
            n_injected=("n_injected", "sum"),
            n_recovered=("n_recovered", "sum"),
            recovery_fraction=("recovery_fraction", "mean"),
        ).to_csv(output_dir / "recovery_summary_by_field.csv", index=False)
    if not recovered_df.empty:
        overall_recovery = summarize_overall_recovery(recovered_df)
        field_recovery = summarize_field_recovery_total(recovered_df)
        numeric_bias_by_trial = summarize_numeric_bias(recovered_df, include_trial=True)
        numeric_bias = summarize_numeric_bias(recovered_df, include_trial=False)
        categorical_bias = summarize_categorical_bias(recovered_df)
        morphology_summary = summarize_morphology_recovery(recovered_df)
        overlap_summary = summarize_overlap_groups(recovered_df)
        property_bins = summarize_binned_dependence(recovered_df)
        luminosity_limits = estimate_luminosity_completeness_limits(recovered_df)
        peak_offsets = summarize_peak_offsets(recovered_df)
        overall_recovery.to_csv(output_dir / "summary_overall_recovery.csv", index=False)
        field_recovery.to_csv(output_dir / "summary_recovery_by_field_total.csv", index=False)
        numeric_bias_by_trial.to_csv(output_dir / "recovery_bias_numeric_by_field_trial.csv", index=False)
        numeric_bias.to_csv(output_dir / "recovery_bias_numeric_by_field.csv", index=False)
        categorical_bias.to_csv(output_dir / "recovery_bias_categorical_by_field.csv", index=False)
        morphology_summary.to_csv(output_dir / "recovery_summary_by_morphology.csv", index=False)
        overlap_summary.to_csv(output_dir / "overlap_pair_recovery_summary.csv", index=False)
        property_bins.to_csv(output_dir / "summary_property_dependence_bins.csv", index=False)
        luminosity_limits.to_csv(output_dir / "summary_luminosity_completeness_limits.csv", index=False)
        peak_offsets.to_csv(output_dir / "summary_peak_offset_statistics.csv", index=False)
        write_summary_manifest(output_dir)
        if not numeric_bias.empty:
            ranked = numeric_bias.copy()
            ranked["abs_missed_minus_recovered_median"] = ranked["missed_minus_recovered_median"].abs()
            ranked = ranked.sort_values(
                ["field", "ks_pvalue_recovered_vs_missed", "abs_missed_minus_recovered_median"],
                ascending=[True, True, False],
                na_position="last",
            )
            ranked.to_csv(output_dir / "recovery_bias_numeric_ranked.csv", index=False)
    if save_distribution_plots_flag and not recovered_df.empty:
        save_distribution_plots(recovered_df, distribution_plot_dir)
    return recovered_df, detections_df, summary_df
