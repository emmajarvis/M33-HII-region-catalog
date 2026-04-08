from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


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
        return format(value, fmt) if fmt is not None else str(value)
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
        lines.append(rf"\newcommand{{\{cmd}}}{{{formatted}}}")
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


def build_catalog_number_values(cat: pd.DataFrame, snr_cut: float = 3.0) -> tuple[dict, dict]:
    values: dict[str, object] = {}
    formats: dict[str, str] = {}

    values["nregions"] = len(cat)
    values["nstarforming"] = len(cat[cat["BPT_class_sum_dered"] == "Star-forming"])
    values["ncomposite"] = len(cat[cat["BPT_class_sum_dered"] == "Composite"])
    values["nagn"] = len(cat[cat["BPT_class_sum_dered"] == "AGN/Shock"])
    values["nregionssnrthree"] = len(
        cat[
            (cat["SNR_Halpha_sum"] > snr_cut)
            & (cat["SNR_[OIII]5007_sum"] > snr_cut)
            & (cat["SNR_[SII]6716_sum"] > snr_cut)
            & (cat["SNR_[NII]6583_sum"] > snr_cut)
            & (cat["SNR_Hbeta_sum"] > snr_cut)
        ]
    )
    values["nregionsdensity"] = len(cat[~pd.isna(cat["ne_SII_cm3"])])

    add_summary_stats(values, formats, cat, "log_L_Ha_sum_dered", "logLHa", ".2f")
    add_summary_stats(values, formats, cat, "radius_p50_pc", "radiuspfifty", ".2f")
    add_summary_stats(values, formats, cat, "radius_p16_pc", "radiuspsixteen", ".2f")
    add_summary_stats(values, formats, cat, "radius_p84_pc", "radiuspeightyfour", ".2f")
    add_summary_stats(values, formats, cat, "radius_areaeq_pc", "radiusareaequiv", ".2f")
    add_summary_stats(values, formats, cat, "ne_SII_cm3", "ne", ".1f")
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

    return values, formats
