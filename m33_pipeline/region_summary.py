from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("01_region_identification") / "summary"

SUMMARY_KEYS_ORDER = [
    "input_peaks",
    "saddle_removed",
    "edge_removed",
    "candidate_peaks",
    "zoi_small_removed",
    "zoi_duplicate_removed",
    "final_peaks_after_zoi",
    "boundary_small_removed",
    "final_regions_after_boundary",
]


def _summary_dir(summary_dir: str | Path | None = None) -> Path:
    path = Path(summary_dir) if summary_dir is not None else DEFAULT_SUMMARY_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _field_json_path(field: str, summary_dir: str | Path | None = None) -> Path:
    return _summary_dir(summary_dir) / f"{field}_region_summary.json"


def _field_txt_path(field: str, summary_dir: str | Path | None = None) -> Path:
    return _summary_dir(summary_dir) / f"{field}_region_summary.txt"


def _overall_txt_path(summary_dir: str | Path | None = None) -> Path:
    return _summary_dir(summary_dir) / "all_fields_region_summary.txt"


def _coerce_jsonable(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _render_field_summary(field: str, stats: dict[str, Any]) -> str:
    lines = [
        f"Field: {field}",
        "",
        "Region Identification Summary",
        "",
    ]
    for key in SUMMARY_KEYS_ORDER:
        if key in stats:
            lines.append(f"- {key}: {stats[key]}")
    extra_keys = [
        key for key in stats.keys() if key not in {"field", *SUMMARY_KEYS_ORDER, "summary_version"}
    ]
    for key in sorted(extra_keys):
        lines.append(f"- {key}: {stats[key]}")
    lines.append("")
    return "\n".join(lines)


def write_field_region_summary(
    field: str,
    stats: dict[str, Any],
    summary_dir: str | Path | None = None,
) -> dict[str, Any]:
    json_path = _field_json_path(field, summary_dir)
    existing: dict[str, Any] = {}
    if json_path.exists():
        existing = json.loads(json_path.read_text())
    merged = {**existing, **_coerce_jsonable(stats)}
    merged["field"] = field
    merged["summary_version"] = 1
    json_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    _field_txt_path(field, summary_dir).write_text(_render_field_summary(field, merged))
    return merged


def write_overall_region_summary(summary_dir: str | Path | None = None) -> Path:
    summary_path = _summary_dir(summary_dir)
    field_json_paths = sorted(summary_path.glob("*_region_summary.json"))
    rows = [json.loads(path.read_text()) for path in field_json_paths]

    totals = {key: 0 for key in SUMMARY_KEYS_ORDER}
    lines = [
        "All Fields Region Identification Summary",
        "",
        f"Fields included: {len(rows)}",
        "",
    ]
    for row in rows:
        field = row.get("field", "UNKNOWN")
        parts = [f"{field}"]
        for key in SUMMARY_KEYS_ORDER:
            value = row.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
            if value is not None:
                parts.append(f"{key}={value}")
        lines.append("- " + ", ".join(parts))

    lines.extend(["", "Totals", ""])
    for key in SUMMARY_KEYS_ORDER:
        lines.append(f"- {key}: {totals[key]}")
    lines.append("")

    out_path = _overall_txt_path(summary_dir)
    out_path.write_text("\n".join(lines))
    return out_path
