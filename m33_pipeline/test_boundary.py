from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from astropy.io import fits

from .config import FIELDS, PhotometryConfig
from .io import write_catalog
from .photometry import build_field_flux_catalog
from .region_domains import (
    DomainParams,
    _prepare_catalog,
    _raw_halpha_map_path,
    compute_boundaries,
    estimate_pc_per_pixel_from_wcs,
    label_at_xy,
    load_first_2d_image_from_fits,
    make_contours_legacy,
)


DEFAULT_OUTPUT_ROOT = Path("test_boundary")
DEFAULT_MAX_ZOI_PC = 100
DEFAULT_DISTANCE_MPC = 0.84

BOUNDARY_VARIANTS = {
    "strict_small": {
        "description": "Small-boundary stress test: higher peak-background fraction and high fallback sigma.",
        "edge_frac": 0.7,
        "fallback_sig_k": 20.0,
        "global_p95_cap": True,
    },
    "loose_large": {
        "description": "Large-boundary stress test: lower peak-background fraction, lower fallback sigma, no p95 radius cap.",
        "edge_frac": 0.02,
        "fallback_sig_k": 3.0,
        "global_p95_cap": False,
    },
}


def now_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def iso_now_string() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def log(message: str) -> None:
    print(f"[{now_string()}] [test_boundary] {message}", flush=True)


def haoiii_boundary_image(field: str) -> Path:
    return Path("peak_files") / f"data_for_visualisation_OIII+Ha_{field}" / f"M33_{field}_HaOIII_amp_nonan.fits"


def zoi_map_path(field: str, max_zoi_pc: int = DEFAULT_MAX_ZOI_PC) -> Path:
    return Path("ZOI_maps") / f"ZOI_map_{max_zoi_pc}pc" / f"ZoI_map_{field}.fits"


def peak_catalog_path(field: str) -> Path:
    return Path("CATALOGS") / f"final_peaks_{field}.csv"


def variant_boundary_dir(output_root: Path, variant: str) -> Path:
    return output_root / "boundary_maps" / variant


def variant_catalog_dir(output_root: Path, variant: str, dig_mode: str = "no_dig") -> Path:
    return output_root / "flux_catalogs" / dig_mode / variant


def base_domain_params(max_zoi_pc: int, distance_mpc: float) -> DomainParams:
    return DomainParams(
        max_zoi_pc=max_zoi_pc,
        assume_distance_mpc=distance_mpc,
        n_theta=72,
        r_bin=2.0,
        smooth_sigma=1.2,
        edge_frac=0.1,
        fallback_sig_k=10.0,
        rmax_default=float(max_zoi_pc),
        ang_win_sectors=max(5, (72 // 8) * 2 + 1),
        mad_k_clip=2.2,
        slope_alpha=0.3,
        slope_min_px=4.0,
        spike_k_neigh=2.0,
        sg_window=7,
        sg_poly=2,
        global_p95_cap=True,
        min_domain_valid_pixels=20,
    )


def build_variant_params(variant: str, max_zoi_pc: int, distance_mpc: float) -> DomainParams:
    if variant not in BOUNDARY_VARIANTS:
        raise KeyError(f"Unknown boundary variant {variant!r}; choose from {sorted(BOUNDARY_VARIANTS)}")
    overrides = {k: v for k, v in BOUNDARY_VARIANTS[variant].items() if k != "description"}
    return replace(base_domain_params(max_zoi_pc=max_zoi_pc, distance_mpc=distance_mpc), **overrides)


def load_boundary_inputs(field: str, max_zoi_pc: int, params: DomainParams):
    halpha, halpha_hdr = load_first_2d_image_from_fits(haoiii_boundary_image(field))
    halpha = np.where(np.isfinite(halpha), halpha, np.nan)
    raw_halpha_path = _raw_halpha_map_path(field)
    if raw_halpha_path.exists():
        valid_halpha, _ = load_first_2d_image_from_fits(raw_halpha_path)
        valid_halpha = np.where(np.isfinite(valid_halpha), valid_halpha, np.nan)
    else:
        valid_halpha = halpha
    pixel_scale_pc = estimate_pc_per_pixel_from_wcs(halpha_hdr, params.assume_distance_mpc)
    peaks = _prepare_catalog(pd.read_csv(peak_catalog_path(field)), field)
    zoi_label = fits.getdata(zoi_map_path(field, max_zoi_pc)).astype(float)
    peaks = peaks.copy()
    peaks["zoi_center_label"] = [label_at_xy(zoi_label, x, y) for x, y in zip(peaks["x"], peaks["y"])]
    return peaks, halpha, halpha_hdr, valid_halpha, zoi_label, pixel_scale_pc


def add_observed_halpha_luminosity(df: pd.DataFrame, distance_mpc: float) -> pd.DataFrame:
    out = df.copy()
    distance_cm = float(distance_mpc) * 1.0e6 * 3.085677581491367e18
    luminosity_factor = 4.0 * np.pi * distance_cm**2
    ha_flux = pd.to_numeric(out.get("F_Halpha_sum_nodig", out.get("F_Halpha_sum")), errors="coerce")
    out["L_Ha_observed"] = luminosity_factor * ha_flux
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_L_Ha_observed"] = np.log10(out["L_Ha_observed"])
    out["boundary_test_distance_mpc"] = float(distance_mpc)
    return out


def run_boundary_variant_for_field(
    field: str,
    variant: str,
    dig_mode: str,
    output_root: Path,
    max_zoi_pc: int,
    distance_mpc: float,
    *,
    overwrite: bool = True,
) -> pd.DataFrame:
    params = build_variant_params(variant, max_zoi_pc=max_zoi_pc, distance_mpc=distance_mpc)
    peaks, halpha, halpha_hdr, valid_halpha, zoi_label, pixel_scale_pc = load_boundary_inputs(field, max_zoi_pc, params)

    boundary_label, metrics_df, _region_diagnostics = compute_boundaries(
        peaks,
        halpha,
        valid_halpha,
        zoi_label,
        field,
        pixel_scale_pc,
        params,
    )

    boundary_dir = variant_boundary_dir(output_root, variant)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    boundary_fits = boundary_dir / f"Boundary_map_{field}.fits"
    metrics_csv = boundary_dir / f"Boundary_metrics_{field}.csv"
    if boundary_fits.exists() and not overwrite:
        raise FileExistsError(boundary_fits)
    fits.PrimaryHDU(boundary_label.astype(np.float32), header=halpha_hdr).writeto(boundary_fits, overwrite=overwrite)
    metrics_df.to_csv(metrics_csv, index=False)
    make_contours_legacy(boundary_label, field=field, out_dir=boundary_dir, header_info=halpha_hdr, filep="ContDomain_map")

    photometry_config = PhotometryConfig(max_zoi_pc=max_zoi_pc)
    field_catalog = build_field_flux_catalog(
        field,
        max_zoi_pc,
        photometry_config,
        method="summed_map",
        dig_mode=dig_mode,
        peaks_csv=peak_catalog_path(field),
        zoi_fits=zoi_map_path(field, max_zoi_pc),
        boundary_fits=boundary_fits,
        boundary_metrics_csv=metrics_csv,
    )
    field_catalog["boundary_test_variant"] = variant
    field_catalog["boundary_test_dig_mode"] = dig_mode
    field_catalog["boundary_edge_frac"] = float(params.edge_frac)
    field_catalog["boundary_fallback_sig_k"] = float(params.fallback_sig_k)
    field_catalog["boundary_global_p95_cap"] = bool(params.global_p95_cap)
    field_catalog = add_observed_halpha_luminosity(field_catalog, distance_mpc=distance_mpc)

    catalog_dir = variant_catalog_dir(output_root, variant, dig_mode=dig_mode)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    write_catalog(field_catalog, catalog_dir / f"flux_catalog_{field}.csv")
    return field_catalog


def run_boundary_variant(
    variant: str,
    dig_mode: str,
    fields: list[str],
    output_root: Path,
    max_zoi_pc: int,
    distance_mpc: float,
    *,
    overwrite: bool = True,
) -> pd.DataFrame:
    field_catalogs = []
    variant_start = perf_counter()
    total_fields = len(fields)
    log(f"{variant}/{dig_mode}: starting {total_fields} field(s)")
    for index, field in enumerate(fields, start=1):
        field_start = perf_counter()
        completed = index - 1
        if completed:
            elapsed = field_start - variant_start
            eta = elapsed / completed * (total_fields - completed)
            eta_text = format_duration(eta)
        else:
            eta_text = "estimating"
        log(f"{variant}/{dig_mode}: field {index}/{total_fields} {field} starting; ETA {eta_text}")
        field_catalogs.append(
            run_boundary_variant_for_field(
                field,
                variant,
                dig_mode,
                output_root,
                max_zoi_pc,
                distance_mpc,
                overwrite=overwrite,
            )
        )
        field_elapsed = perf_counter() - field_start
        completed_elapsed = perf_counter() - variant_start
        remaining = total_fields - index
        eta = completed_elapsed / index * remaining
        log(
            f"{variant}/{dig_mode}: field {index}/{total_fields} {field} finished in "
            f"{format_duration(field_elapsed)}; remaining ETA {format_duration(eta)}"
        )
    total = pd.concat(field_catalogs, axis=0, ignore_index=True, sort=False)
    total_path = variant_catalog_dir(output_root, variant, dig_mode=dig_mode) / "total_flux_catalog.csv"
    write_catalog(total, total_path)
    log(f"{variant}/{dig_mode}: wrote {len(total)} rows to {total_path} in {format_duration(perf_counter() - variant_start)}")
    return total


def write_manifest(output_root: Path, variants: list[str], dig_modes: list[str], fields: list[str], max_zoi_pc: int, distance_mpc: float) -> Path:
    payload = {
        "created_at": iso_now_string(),
        "fields": fields,
        "dig_modes": dig_modes,
        "max_zoi_pc": max_zoi_pc,
        "distance_mpc": distance_mpc,
        "variants": {
            name: {
                **BOUNDARY_VARIANTS[name],
                "domain_params": asdict(build_variant_params(name, max_zoi_pc=max_zoi_pc, distance_mpc=distance_mpc)),
            }
            for name in variants
        },
        "notes": [
            "Peak catalog and ZoI maps are reused from the current pipeline products.",
            "Domain-size pruning is not rerun, so peak identification is fixed for these tests.",
            "Flux catalogs are summed-map catalogs for each requested DIG mode, intended for Halpha luminosity and luminosity-radius comparisons.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run boundary-parameter stress tests without changing the main catalog.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for test outputs.")
    parser.add_argument("--fields", nargs="+", default=FIELDS, help="Fields to process.")
    parser.add_argument("--variants", nargs="+", default=["strict_small", "loose_large"], choices=sorted(BOUNDARY_VARIANTS))
    parser.add_argument("--dig-modes", nargs="+", default=["no_dig", "dig_subtracted"], choices=["no_dig", "dig_subtracted"])
    parser.add_argument("--max-zoi-pc", type=int, default=DEFAULT_MAX_ZOI_PC)
    parser.add_argument("--distance-mpc", type=float, default=DEFAULT_DISTANCE_MPC)
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if an output already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_start = perf_counter()
    output_root = Path(args.output_root)
    fields = list(args.fields)
    variants = list(args.variants)
    dig_modes = list(args.dig_modes)
    log(
        f"run started with {len(variants)} variant(s), {len(dig_modes)} DIG mode(s), {len(fields)} field(s), "
        f"output_root={output_root}"
    )
    manifest_path = write_manifest(output_root, variants, dig_modes, fields, args.max_zoi_pc, args.distance_mpc)
    log(f"wrote manifest: {manifest_path}")
    jobs = [(variant, dig_mode) for dig_mode in dig_modes for variant in variants]
    for job_index, (variant, dig_mode) in enumerate(jobs, start=1):
        completed_jobs = job_index - 1
        if completed_jobs:
            elapsed = perf_counter() - run_start
            eta = elapsed / completed_jobs * (len(jobs) - completed_jobs)
            log(f"job {job_index}/{len(jobs)} starting: {variant}/{dig_mode}; run ETA {format_duration(eta)}")
        else:
            log(f"job {job_index}/{len(jobs)} starting: {variant}/{dig_mode}; run ETA estimating")
        total = run_boundary_variant(
            variant,
            dig_mode,
            fields,
            output_root,
            args.max_zoi_pc,
            args.distance_mpc,
            overwrite=not args.no_overwrite,
        )
        log(f"job {job_index}/{len(jobs)} complete: {variant}/{dig_mode}, rows={len(total)}")
    log(f"run finished in {format_duration(perf_counter() - run_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
