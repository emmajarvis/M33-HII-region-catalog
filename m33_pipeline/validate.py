from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> list[str]:
    missing = [col for col in cols if col not in df.columns]
    if not missing:
        return []
    return [f"{context}: missing columns: {', '.join(missing)}"]


def assert_unique(df: pd.DataFrame, col: str, context: str) -> list[str]:
    if col not in df.columns:
        return [f"{context}: missing uniqueness column: {col}"]
    if df[col].duplicated().any():
        return [f"{context}: duplicate values found in {col}"]
    return []


def finite_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    values = pd.to_numeric(series, errors="coerce")
    return float(np.isfinite(values).sum() / len(values))


def validate_field_flux_catalog(df: pd.DataFrame, field: str) -> list[str]:
    warnings = []
    warnings.extend(require_columns(df, ["region_id", "x", "y", "field"], f"{field} flux catalog"))
    warnings.extend(assert_unique(df, "region_id", f"{field} flux catalog"))
    if "field" in df.columns and not (df["field"] == field).all():
        warnings.append(f"{field} flux catalog: unexpected mixed field values")
    if "npix_region" in df.columns and (pd.to_numeric(df["npix_region"], errors="coerce") <= 0).any():
        warnings.append(f"{field} flux catalog: non-positive region sizes detected")
    required_flux_cols = ["F_Halpha_sum", "F_Halpha_e_sum", "SNR_Halpha_sum"]
    warnings.extend(require_columns(df, required_flux_cols, f"{field} flux catalog"))
    return warnings


def validate_total_catalog(df: pd.DataFrame) -> list[str]:
    warnings = []
    warnings.extend(require_columns(df, ["field", "region_id"], "total catalog"))
    if "field" in df.columns and df["field"].isna().any():
        warnings.append("total catalog: null field values detected")
    return warnings


def write_qc_report(path: Path, stage: str, field: str | None, warnings: list[str], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "field": field,
        "warnings": warnings,
        "summary": summary,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
