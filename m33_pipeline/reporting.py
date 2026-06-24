from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from . import paths
from .config import FIELDS


_DIGIT_WORDS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


def latex_command_name(value: str) -> str:
    """Return a TeX-safe command name containing letters only."""
    return "".join(_DIGIT_WORDS.get(char, char) for char in str(value) if char.isalnum())


def load_wr_catalog(path: str | Path = "CATALOGS/WR_catalog.csv") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_snr_catalog(path: str | Path = "CATALOGS/snr_table.csv") -> pd.DataFrame:
    path = Path(path)
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean = re.sub(r"\\rm|\\pm|\$", "", line)
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            ra, dec = None, None
            for part in parts:
                if ":" in part and ra is None:
                    ra = part
                elif (part.startswith("+") or part.startswith("-")) and ":" in part and dec is None:
                    dec = part
            if ra and dec:
                rows.append((ra, dec))
    return pd.DataFrame(rows, columns=["RA", "Dec"])


def latex_escape(value):
    if not isinstance(value, str):
        return value
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def format_value(value, fmt: str | None = None) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        formatted = format(value, fmt) if fmt is not None else str(value)
        if "e" in formatted.lower():
            mantissa, exponent = re.split("[eE]", formatted)
            return rf"${mantissa}\times10^{{{int(exponent)}}}$"
        return formatted
    return latex_escape(str(value))


def write_latex_commands(filename: str | Path, values: dict, formats: dict | None = None) -> Path:
    filename = Path(filename)
    formats = formats or {}
    lines = [
        "% This file is auto-generated from the Jupyter notebook.",
        "% Do not edit by hand.",
        "",
    ]
    for cmd, value in values.items():
        formatted = format_value(value, formats.get(cmd))
        lines.append(rf"\newcommand{{\{latex_command_name(cmd)}}}{{{formatted}}}")
    filename.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filename


def add_summary_stats(values: dict, formats: dict, df: pd.DataFrame, col: str, prefix: str, fmt: str = ".2f") -> None:
    if col not in df.columns:
        return
    data = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(data) == 0:
        return
    p5, p16, p25, p50, p75, p84, p95 = np.percentile(data, [5, 16, 25, 50, 75, 84, 95])
    values[f"n{prefix}"] = len(data)
    values[f"median{prefix}"] = p50
    values[f"pfive{prefix}"] = p5
    values[f"psixteen{prefix}"] = p16
    values[f"ptwentyfive{prefix}"] = p25
    values[f"pseventyfive{prefix}"] = p75
    values[f"peightyfour{prefix}"] = p84
    values[f"pninetyfive{prefix}"] = p95
    values[f"min{prefix}"] = data.min()
    values[f"max{prefix}"] = data.max()
    for key in [
        f"median{prefix}",
        f"pfive{prefix}",
        f"psixteen{prefix}",
        f"ptwentyfive{prefix}",
        f"pseventyfive{prefix}",
        f"peightyfour{prefix}",
        f"pninetyfive{prefix}",
        f"min{prefix}",
        f"max{prefix}",
    ]:
        formats[key] = fmt


def add_fit_values(
    values: dict,
    formats: dict,
    fit: dict | None,
    prefix: str,
    fmt: str = ".4f",
) -> None:
    """Add a named linear fit and its uncertainties to a LaTeX-value mapping."""
    if not fit:
        return
    keys = {
        "slope": f"{prefix}Slope",
        "intercept": f"{prefix}Intercept",
        "slope_stderr": f"{prefix}SlopeErr",
        "intercept_stderr": f"{prefix}InterceptErr",
        "n_points": f"{prefix}N",
    }
    for source, command in keys.items():
        if source not in fit:
            continue
        values[command] = fit[source]
        if source != "n_points":
            formats[command] = fmt


def add_region_removal_values(
    values: dict,
    summary_dir: str | Path = Path("01_region_identification") / "summary",
) -> None:
    """Add totals removed and remaining after each peak/region pruning stage."""
    summary_paths = sorted(Path(summary_dir).glob("*_region_summary.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    if not rows:
        return

    def total(key: str) -> int:
        return sum(int(row.get(key, 0)) for row in rows)

    input_peaks = total("input_peaks")
    saddle_removed = total("saddle_removed")
    edge_removed = total("edge_removed")
    candidate_peaks = total("candidate_peaks")
    zoi_small_removed = total("zoi_small_removed")
    zoi_duplicate_removed = total("zoi_duplicate_removed")
    final_peaks_after_zoi = total("final_peaks_after_zoi")
    boundary_small_removed = total("boundary_small_removed")
    final_regions_after_boundary = total("final_regions_after_boundary")

    values.update(
        {
            "nRegionSummaryFields": len(rows),
            "nPeaksInput": input_peaks,
            "nPeaksRemovedSaddle": saddle_removed,
            "nPeaksAfterSaddle": input_peaks - saddle_removed,
            "nPeaksRemovedEdge": edge_removed,
            "nPeaksAfterEdge": candidate_peaks,
            "nPeaksRemovedSmallZoI": zoi_small_removed,
            "nPeaksAfterSmallZoI": candidate_peaks - zoi_small_removed,
            "nPeaksRemovedDuplicateZoI": zoi_duplicate_removed,
            "nPeaksAfterDuplicateZoI": final_peaks_after_zoi,
            "nRegionsRemovedSmallBoundary": boundary_small_removed,
            "nRegionsAfterSmallBoundary": final_regions_after_boundary,
        }
    )


def add_boundary_edge_method_values(
    values: dict,
    boundary_dir: str | Path = Path("Boundary_maps") / "Boundary_map_100pc",
) -> None:
    """Add counts of the dominant ray edge-finding method per final boundary region."""
    boundary_paths = sorted(Path(boundary_dir).glob("Boundary_metrics_*.csv"))
    if not boundary_paths:
        return

    frames = []
    for path in boundary_paths:
        df = pd.read_csv(path, usecols=lambda col: col == "edge_flag_primary")
        if "edge_flag_primary" in df.columns:
            frames.append(df)
    if not frames:
        return

    primary_counts = pd.concat(frames, ignore_index=True)["edge_flag_primary"].value_counts()
    values.update(
        {
            "nBoundaryPrimaryFractional": int(primary_counts.get("RAY_CROSS_FRAC_PROM", 0)),
            "nBoundaryPrimaryFallbackSigma": int(primary_counts.get("RAY_CROSS_BG_KSIG", 0)),
            "nBoundaryPrimaryZoIClamp": int(primary_counts.get("RAY_CLAMP_TO_ZOI", 0)),
            "nBoundaryPrimaryLocalMinimum": int(primary_counts.get("RAY_LOCAL_MIN", 0)),
        }
    )


def add_duplicate_region_values(values: dict, cat: pd.DataFrame) -> None:
    """Add overlap-duplicate counts before the primary-only catalog filter."""
    duplicate_col = None
    for candidate in ("duplicated", "is_duplicate_overlap"):
        if candidate in cat.columns:
            duplicate_col = candidate
            break

    values["nRegionsBeforeDuplicateRemoval"] = int(len(cat))
    if duplicate_col is None:
        values["nRegionsDuplicated"] = 0
        values["nRegionsAfterDuplicateRemoval"] = int(len(cat))
        return

    duplicated = cat[duplicate_col].fillna(False).astype(bool)
    values["nRegionsDuplicated"] = int(duplicated.sum())
    if "duplicate_peak_overlap" in cat.columns:
        values["nRegionsDuplicatedByPeakOverlap"] = int(cat["duplicate_peak_overlap"].fillna(False).astype(bool).sum())
    else:
        values["nRegionsDuplicatedByPeakOverlap"] = 0
    if "duplicate_boundary_overlap" in cat.columns:
        values["nRegionsDuplicatedByBoundaryOverlap"] = int(cat["duplicate_boundary_overlap"].fillna(False).astype(bool).sum())
    else:
        values["nRegionsDuplicatedByBoundaryOverlap"] = 0
    if "primary" in cat.columns:
        primary = cat["primary"].fillna(True).astype(bool)
        values["nRegionsAfterDuplicateRemoval"] = int(primary.sum())
        values["nRegionsRemovedAsDuplicateNonPrimary"] = int((~primary).sum())
        if "duplicate_boundary_overlap" in cat.columns:
            boundary_duplicate = cat["duplicate_boundary_overlap"].fillna(False).astype(bool)
            values["nRegionsRemovedAsBoundaryOverlapNonPrimary"] = int((boundary_duplicate & ~primary).sum())
        else:
            values["nRegionsRemovedAsBoundaryOverlapNonPrimary"] = 0
    else:
        values["nRegionsAfterDuplicateRemoval"] = int(len(cat))
        values["nRegionsRemovedAsDuplicateNonPrimary"] = 0
        values["nRegionsRemovedAsBoundaryOverlapNonPrimary"] = 0


def add_halpha_flux_partition_values(
    values: dict,
    formats: dict,
    fields: list[str] | tuple[str, ...] = FIELDS,
    max_zoi_pc: int = 100,
    max_halpha_flux: float = 1.0e-12,
    halpha_path_template: str | None = None,
    boundary_path_template: str | None = None,
    zoi_path_template: str | None = None,
    prefix: str = "",
) -> None:
    """Add Halpha flux percentages inside HII boundaries, ZoIs, and outside ZoIs."""
    flux_hii = 0.0
    flux_zoi_not_hii = 0.0
    flux_outside_zoi = 0.0
    excluded_high_pixels = 0

    for field in fields:
        halpha_path = (
            Path(halpha_path_template.format(field=field, FIELD=field))
            if halpha_path_template is not None
            else paths.calibrated_field_map_dir(field) / f"M33{field}-Haflux.fits"
        )
        boundary_path = (
            Path(boundary_path_template.format(field=field, FIELD=field))
            if boundary_path_template is not None
            else paths.boundary_fits(field, max_zoi_pc)
        )
        zoi_path = (
            Path(zoi_path_template.format(field=field, FIELD=field))
            if zoi_path_template is not None
            else paths.zoi_fits(field, max_zoi_pc)
        )
        if not (halpha_path.exists() and boundary_path.exists() and zoi_path.exists()):
            continue

        halpha = np.squeeze(fits.getdata(halpha_path)).astype(float)
        boundary = np.squeeze(fits.getdata(boundary_path))
        zoi = np.squeeze(fits.getdata(zoi_path))
        if halpha.shape != boundary.shape or halpha.shape != zoi.shape:
            raise ValueError(
                f"Shape mismatch for {field}: Halpha={halpha.shape}, "
                f"boundary={boundary.shape}, ZoI={zoi.shape}"
            )

        finite_positive = np.isfinite(halpha) & (halpha > 0)
        high_outlier = finite_positive & (halpha >= max_halpha_flux)
        excluded_high_pixels += int(np.count_nonzero(high_outlier))
        valid = finite_positive & (~high_outlier)
        hii_mask = valid & np.isfinite(boundary) & (boundary > 0)
        zoi_not_hii_mask = valid & (~hii_mask) & np.isfinite(zoi) & (zoi > 0)
        outside_zoi_mask = valid & (~hii_mask) & (~zoi_not_hii_mask)

        flux_hii += float(np.nansum(halpha[hii_mask]))
        flux_zoi_not_hii += float(np.nansum(halpha[zoi_not_hii_mask]))
        flux_outside_zoi += float(np.nansum(halpha[outside_zoi_mask]))

    total_flux = flux_hii + flux_zoi_not_hii + flux_outside_zoi
    if total_flux <= 0 or not np.isfinite(total_flux):
        return

    values.update(
        {
            f"{prefix}halphaFluxTotalForPartition": total_flux,
            f"{prefix}halphaFluxPercentInHIIRegions": 100.0 * flux_hii / total_flux,
            f"{prefix}halphaFluxPercentNotInHIIRegions": 100.0 * (1 - flux_hii / total_flux),
            f"{prefix}halphaFluxPercentInZoINotHIIRegions": 100.0 * flux_zoi_not_hii / total_flux,
            f"{prefix}halphaFluxPercentOutsideZoI": 100.0 * flux_outside_zoi / total_flux,
            f"{prefix}nHalphaFluxOutlierPixelsExcluded": excluded_high_pixels,
        }
    )
    formats.update(
        {
            f"{prefix}halphaFluxTotalForPartition": ".3e",
            f"{prefix}halphaFluxPercentInHIIRegions": ".2f",
            f"{prefix}halphaFluxPercentNotInHIIRegions": ".2f",
            f"{prefix}halphaFluxPercentInZoINotHIIRegions": ".2f",
            f"{prefix}halphaFluxPercentOutsideZoI": ".2f",
        }
    )


def build_catalog_number_values(
    cat: pd.DataFrame,
    snr_cut: float = 3.0,
    include_region_removal: bool = True,
    include_boundary_edge_methods: bool = True,
    include_halpha_flux_partition: bool = True,
) -> tuple[dict, dict]:
    values: dict[str, object] = {}
    formats: dict[str, str] = {}

    values["nregions"] = len(cat)
    bpt_class = cat["BPT_class_sum_dered"]
    values["nstarforming"] = len(cat[bpt_class == "Star-forming"])
    values["ncomposite"] = len(cat[bpt_class == "Composite"])
    values["nagn"] = len(cat[bpt_class == "AGN/Shock"])
    classified = bpt_class.isin(["Star-forming", "Composite", "AGN/Shock"])
    values["nUnclassified"] = int((~classified).sum())

    snr_three_mask = (
        (cat["SNR_Halpha_sum"] > snr_cut)
        & (cat["SNR_[OIII]5007_sum"] > snr_cut)
        & (cat["SNR_[SII]6716_sum"] > snr_cut)
        & (cat["SNR_[NII]6583_sum"] > snr_cut)
        & (cat["SNR_Hbeta_sum"] > snr_cut)
    )
    values["nregionssnrthree"] = int(snr_three_mask.sum())
    values["nStarformingSNRThree"] = int(((bpt_class == "Star-forming") & snr_three_mask).sum())
    values["nCompositeSNRThree"] = int(((bpt_class == "Composite") & snr_three_mask).sum())
    values["nAgnSNRThree"] = int(((bpt_class == "AGN/Shock") & snr_three_mask).sum())
    values["nregionsdensity"] = len(cat[~pd.isna(cat["ne_SII_cm3"])])
    if {"has_snr_in_boundary", "has_wr_in_boundary"}.issubset(cat.columns):
        snr_hosts = cat["has_snr_in_boundary"].fillna(False).astype(bool)
        wr_hosts = cat["has_wr_in_boundary"].fillna(False).astype(bool)
        values["nRegionsContainingSNR"] = int(snr_hosts.sum())
        values["nRegionsContainingWR"] = int(wr_hosts.sum())
        values["nRegionsContainingSNRAndWR"] = int((snr_hosts & wr_hosts).sum())
        values["nRegionsContainingSNROrWR"] = int((snr_hosts | wr_hosts).sum())
    if "has_pn_in_boundary" in cat.columns:
        pn_hosts = cat["has_pn_in_boundary"].fillna(False).astype(bool)
        values["nRegionsContainingPN"] = int(pn_hosts.sum())
    if "n_pn_in_boundary" in cat.columns:
        values["nCrossMatchedPN"] = int(
            pd.to_numeric(cat["n_pn_in_boundary"], errors="coerce").fillna(0).sum()
        )

    add_summary_stats(values, formats, cat, "log_L_Ha_sum_dered", "logLHa", ".2f")
    add_summary_stats(values, formats, cat, "radius_p50_pc", "radiuspfifty", ".2f")
    add_summary_stats(values, formats, cat, "radius_p16_pc", "radiuspsixteen", ".2f")
    add_summary_stats(values, formats, cat, "radius_p84_pc", "radiuspeightyfour", ".2f")
    add_summary_stats(values, formats, cat, "radius_areaeq_pc", "radiusareaequiv", ".2f")
    add_summary_stats(values, formats, cat, "ne_SII_cm3", "ne", ".1f")
    if "log_P_thermal_SII_over_k" in cat.columns:
        add_summary_stats(values, formats, cat, "log_P_thermal_SII_over_k", "logPthermal", ".2f")
    add_summary_stats(values, formats, cat, "logU_KK04", "logU", ".3f")
    add_summary_stats(values, formats, cat, "Z_N2S2Halpha_Brazzini2024", "metallicity", ".3f")
    add_summary_stats(values, formats, cat, "sum_A_V", "Av", ".2f")
    add_summary_stats(values, formats, cat, "DIG_fraction", "DIGfraction", ".3f")
    add_summary_stats(values, formats, cat, "distance_5th_closest_pc_deproj", "dfive", ".2f")
    add_summary_stats(values, formats, cat, "L_Ha_sum_dered", "LHaLinear", ".2e")
    add_summary_stats(values, formats, cat, "sum_E_BV", "EBV", ".2f")
    add_summary_stats(values, formats, cat, "nearest_neighbor_pc_deproj", "nearestneighbour", ".2f")
    if "sigma5_per_pc2_deproj" in cat.columns:
        add_summary_stats(values, formats, cat, "sigma5_per_pc2_deproj", "sigmafive", ".3e")
    if include_region_removal:
        add_region_removal_values(values)
    if include_boundary_edge_methods:
        add_boundary_edge_method_values(values)
    if include_halpha_flux_partition:
        add_halpha_flux_partition_values(values, formats)

    return values, formats
