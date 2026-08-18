from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import papermill as pm
from astropy.io import fits

from m33_pipeline.config import FIELDS
from m33_pipeline.region_domains import DomainParams, run_domains_with_final_prune


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("papermill").setLevel(logging.INFO)


REPO_ROOT = Path(__file__).resolve().parent
OUT_ROOT = REPO_ROOT / "executed"
REGION_TEST_ROOT = REPO_ROOT / "region_identification_testing"

REGION_NOTEBOOKS = [
    "01_region_identification/2a_make_laplace_maps.ipynb",
    "01_region_identification/2b_make_threshold_maps.ipynb",
    "01_region_identification/3_finalize_peak_detection.ipynb",
    "01_region_identification/4_ZoI.ipynb",
    "01_region_identification/5_domains.ipynb",
]

CATALOG_NOTEBOOKS = [
    "02_catalog_values/6_flux_catalog_individual_field.ipynb",
]

FINAL_NOTEBOOKS = [
    "02_catalog_values/7_create_total_M33_catalog.ipynb",
]

PLOT_NOTEBOOKS = [
    "03_paper_plots_and_muse/19_catalog_comparison_plot_maps.ipynb",
    "03_paper_plots_and_muse/8_catalog_plots.ipynb",
    "03_paper_plots_and_muse/.ipynb",
]

RESOLUTION_NOTEBOOKS = [
    "04_resolution_degradation/1-NWMake_map_M33-resolution.ipynb",
    "04_resolution_degradation/10_resolution_comparison.ipynb",
]

STAGE_NOTEBOOKS = {
    "regions": REGION_NOTEBOOKS,
    "catalog": CATALOG_NOTEBOOKS,
    "plots": PLOT_NOTEBOOKS,
    "resolution": RESOLUTION_NOTEBOOKS,
}

FIELD_STAGES = {"regions", "catalog"}


def banner(message: str) -> None:
    logging.info("")
    logging.info("*****")
    logging.info("%s", message)
    logging.info("*****")
    logging.info("")


def variant_label(parameters: dict[str, object]) -> str:
    parts = []
    if "field" in parameters:
        parts.append(f"field={parameters['field']}")
    if "sizedet" in parameters:
        parts.append(f"sizedet={parameters['sizedet']}")
    if "flux_method" in parameters:
        parts.append(f"flux_method={parameters['flux_method']}")
    if "dig_mode" in parameters:
        parts.append(f"dig_mode={parameters['dig_mode']}")
    return " ".join(parts) if parts else "no parameters"


def execute_notebook(input_path: Path, output_path: Path, parameters: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner(f"Running {input_path} {variant_label(parameters)}")
    pm.execute_notebook(
        input_path=str(input_path),
        output_path=str(output_path),
        parameters=parameters,
        cwd=str(REPO_ROOT),
        log_output=False,
        progress_bar=True,
    )
    logging.info("Completed notebook %s -> %s", input_path.name, output_path)


def _safe_value(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def region_test_tag(*, lap: int, lap_sizebox: int, threshold_sizebox: int, snrlim: float, bgbox: int, stdbox: int, signoi: float, threshold_signal_mode: str) -> str:
    return (
        f"L{lap}_lapsize{lap_sizebox}_thsize{threshold_sizebox}"
        f"_snr{_safe_value(float(snrlim))}_bg{bgbox}_std{stdbox}"
        f"_signoi{_safe_value(float(signoi))}_{threshold_signal_mode}"
    )


def region_test_field_paths(field: str, *, lap: int, lap_sizebox: int, threshold_sizebox: int, bgbox: int, stdbox: int, signoi: float) -> dict[str, Path]:
    visual_dir = REPO_ROOT / "peak_files" / f"data_for_visualisation_OIII+Ha_{field}"
    threshold_dir = REPO_ROOT / "peak_files" / f"2-BG_Noi_Th_data_OIII+Ha_{field}"
    maps_dir = REPO_ROOT.parent / "M33-Maps" / f"M33-{field}"
    return {
        "candidate_peak": visual_dir / f"A_PicL{lap}_sizd{lap_sizebox}.fits",
        "threshold_signal_sized": visual_dir / f"AMP_sizd{threshold_sizebox}.fits",
        "threshold_signal_raw": visual_dir / f"M33_{field}_HaOIII_amp_nonan.fits",
        "threshold": threshold_dir / f"Th_box{bgbox}_std{stdbox}_signoi{float(signoi):.1f}.fits",
        "amp": maps_dir / f"M33-{field}_SN3.LineMaps.map.Ha+OIII.1x1.amplitude.fits",
        "amp_err": maps_dir / f"M33-{field}_SN3.LineMaps.map.Ha+OIII.1x1.amplitude-err.fits",
        "halpha_oiii": visual_dir / f"M33_{field}_HaOIII_amp_nonan.fits",
    }


def region_test_threshold_signal_path(field: str, paths: dict[str, Path], threshold_signal_mode: str) -> Path:
    if threshold_signal_mode == "amp_sized":
        return paths["threshold_signal_sized"]
    if threshold_signal_mode == "match_latest":
        return paths["threshold_signal_raw"] if field == "F6" else paths["threshold_signal_sized"]
    raise ValueError("threshold_signal_mode must be 'amp_sized' or 'match_latest'")


def identify_region_test_peaks(field: str, *, lap: int, lap_sizebox: int, threshold_sizebox: int, snrlim: float, bgbox: int, stdbox: int, signoi: float, threshold_signal_mode: str) -> tuple[pd.DataFrame, dict[str, str]]:
    paths = region_test_field_paths(
        field,
        lap=lap,
        lap_sizebox=lap_sizebox,
        threshold_sizebox=threshold_sizebox,
        bgbox=bgbox,
        stdbox=stdbox,
        signoi=signoi,
    )
    threshold_signal = region_test_threshold_signal_path(field, paths, threshold_signal_mode)
    required = [paths["candidate_peak"], threshold_signal, paths["threshold"], paths["amp"], paths["amp_err"], paths["halpha_oiii"]]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing region-test input files:\n" + "\n".join(str(path) for path in missing))

    peak_map = fits.getdata(paths["candidate_peak"]).astype(float)
    amp = fits.getdata(paths["amp"]).astype(float)
    amp_err = fits.getdata(paths["amp_err"]).astype(float)
    threshold = fits.getdata(paths["threshold"]).astype(float)
    threshold_signal_data = fits.getdata(threshold_signal).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = amp / amp_err

    selected = peak_map.copy()
    selected[snr < snrlim] = np.nan
    selected[(threshold_signal_data - threshold) <= 0] = np.nan
    y, x = np.where(selected == 1)
    peaks = pd.DataFrame(
        {
            "field": field,
            "y": y.astype(int),
            "x": x.astype(int),
            "peak_snr": snr[y, x],
            "threshold_excess": (threshold_signal_data - threshold)[y, x],
        }
    )
    metadata = {key: str(value) for key, value in paths.items()}
    metadata["threshold_signal"] = str(threshold_signal)
    return peaks, metadata


def maybe_generate_laplacian_map(field: str, lap_sizebox: int, test_out_dir: Path, rebuild: bool) -> None:
    # Notebook 2a writes a parameter-specific candidate map into peak_files, e.g. A_PicL15_sizd30.fits.
    # That file name is separate from the production sizd10 map, so creating it does not overwrite final catalogs.
    in_path = REPO_ROOT / "01_region_identification/2a_make_laplace_maps.ipynb"
    expected = REPO_ROOT / "peak_files" / f"data_for_visualisation_OIII+Ha_{field}" / f"A_PicL15_sizd{lap_sizebox}.fits"
    if expected.exists() and not rebuild:
        return
    params = {"field": field, "sizedet": lap_sizebox}
    out_path = test_out_dir / "executed_laplace" / field / f"2a_make_laplace_maps__{field}__sizedet{lap_sizebox}.ipynb"
    execute_notebook(in_path, out_path, params)


def run_region_identification_test_variant(
    *,
    fields: tuple[str, ...],
    lap: int,
    lap_sizebox: int,
    threshold_sizebox: int,
    snrlim: float,
    bgbox: int,
    stdbox: int,
    signoi: float,
    threshold_signal_mode: str,
    max_zoi_pc: int,
    rebuild_laplacian: bool,
) -> pd.DataFrame:
    tag = region_test_tag(
        lap=lap,
        lap_sizebox=lap_sizebox,
        threshold_sizebox=threshold_sizebox,
        snrlim=snrlim,
        bgbox=bgbox,
        stdbox=stdbox,
        signoi=signoi,
        threshold_signal_mode=threshold_signal_mode,
    )
    test_root = REGION_TEST_ROOT / tag
    peaks_dir = test_root / "peaks"
    zoi_dir = test_root / f"ZOI_map_{max_zoi_pc}pc"
    boundary_dir = test_root / f"Boundary_map_{max_zoi_pc}pc"
    tables_dir = test_root / "tables"
    for path in [peaks_dir, zoi_dir, boundary_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for field in fields:
        banner(f"Region-identification test {tag}: field {field}")
        maybe_generate_laplacian_map(field, lap_sizebox, test_root, rebuild_laplacian)
        peaks, peak_metadata = identify_region_test_peaks(
            field,
            lap=lap,
            lap_sizebox=lap_sizebox,
            threshold_sizebox=threshold_sizebox,
            snrlim=snrlim,
            bgbox=bgbox,
            stdbox=stdbox,
            signoi=signoi,
            threshold_signal_mode=threshold_signal_mode,
        )
        peak_csv = peaks_dir / f"peaks_{field}_{tag}.csv"
        peaks.to_csv(peak_csv, index=False)
        logging.info("%s: %d input peaks -> %s", field, len(peaks), peak_csv)

        domain_params = DomainParams(max_zoi_pc=max_zoi_pc, sizebox=1, min_domain_valid_pixels=20)
        boundary_fits = boundary_dir / f"Boundary_map_{field}.fits"
        metrics_csv = boundary_dir / f"Boundary_metrics_{field}.csv"
        result = run_domains_with_final_prune(
            field=field,
            peak_csv=peak_csv,
            halpha_fits=peak_metadata["halpha_oiii"],
            zoi_fits=zoi_dir / f"ZoI_map_{field}.fits",
            contzoi_fits=zoi_dir / f"ContZoI_map_{field}.fits",
            boundary_fits=boundary_fits,
            metrics_csv=metrics_csv,
            params=domain_params,
        )
        metrics = result["metrics_df"].copy()
        final_regions = int(result["prune_summary"]["final_regions"])
        radius_col = "radius_areaeq_pc_after_carve" if "radius_areaeq_pc_after_carve" in metrics.columns else "radius_areaeq_pc"
        radii = pd.to_numeric(metrics.get(radius_col, pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "variant": tag,
                "field": field,
                "lap": lap,
                "lap_sizebox": lap_sizebox,
                "threshold_sizebox": threshold_sizebox,
                "snrlim": snrlim,
                "bgbox": bgbox,
                "stdbox": stdbox,
                "signoi": signoi,
                "threshold_signal_mode": threshold_signal_mode,
                "input_peaks": int(len(peaks)),
                "final_regions_after_boundary": final_regions,
                "mean_radius_areaeq_pc": float(radii.mean()) if len(radii) else np.nan,
                "median_radius_areaeq_pc": float(radii.median()) if len(radii) else np.nan,
                "mean_area_px_after_carve": float(pd.to_numeric(metrics.get("area_px_after_carve", pd.Series(dtype=float)), errors="coerce").mean()),
                "peak_csv": str(peak_csv),
                "boundary_fits": str(boundary_fits),
                "metrics_csv": str(metrics_csv),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = tables_dir / f"region_test_summary_{tag}.csv"
    summary.to_csv(summary_path, index=False)
    total = {
        "variant": tag,
        "n_fields": len(fields),
        "total_input_peaks": int(summary["input_peaks"].sum()),
        "total_final_regions_after_boundary": int(summary["final_regions_after_boundary"].sum()),
        "mean_radius_areaeq_pc_all_regions": np.nan,
        "summary_csv": str(summary_path),
    }
    all_metrics = []
    for metrics_path in summary["metrics_csv"]:
        metrics = pd.read_csv(metrics_path)
        all_metrics.append(metrics)
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        radius_col = "radius_areaeq_pc_after_carve" if "radius_areaeq_pc_after_carve" in combined_metrics.columns else "radius_areaeq_pc"
        total["mean_radius_areaeq_pc_all_regions"] = float(pd.to_numeric(combined_metrics[radius_col], errors="coerce").mean())
        total["median_radius_areaeq_pc_all_regions"] = float(pd.to_numeric(combined_metrics[radius_col], errors="coerce").median())
    total_path = tables_dir / f"region_test_totals_{tag}.csv"
    pd.DataFrame([total]).to_csv(total_path, index=False)
    logging.info("Wrote region-test summary: %s", summary_path)
    logging.info("Wrote region-test totals: %s", total_path)
    logging.info(
        "%s totals: input peaks=%d, final regions=%d, mean radius=%.3f pc",
        tag,
        total["total_input_peaks"],
        total["total_final_regions_after_boundary"],
        total["mean_radius_areaeq_pc_all_regions"],
    )
    return summary


def run_region_identification_tests(args: argparse.Namespace) -> list[tuple[str, str, dict[str, object], str]]:
    failures: list[tuple[str, str, dict[str, object], str]] = []
    fields = tuple(args.fields)
    for lap_sizebox in args.lap_sizeboxes:
        params = {
            "lap": args.lap,
            "lap_sizebox": lap_sizebox,
            "threshold_sizebox": args.threshold_sizebox,
            "snrlim": args.snr_limit,
            "bgbox": args.bgbox,
            "stdbox": args.stdbox,
            "signoi": args.signoi,
            "threshold_signal_mode": args.threshold_signal_mode,
            "max_zoi_pc": args.max_zoi_pc,
            "rebuild_laplacian": args.rebuild_laplacian,
        }
        try:
            run_region_identification_test_variant(fields=fields, **params)
        except Exception as exc:  # pragma: no cover
            logging.error("FAILED region-identification test %s", params)
            failures.append(("REGION_TEST", "region-tests", params, str(exc).strip() or "(no message)"))
    return failures


def run_stage(
    stage: str,
    fields: tuple[str, ...],
    flux_methods: tuple[str, ...],
    dig_modes: tuple[str, ...],
) -> list[tuple[str, str, dict[str, object], str]]:
    failures: list[tuple[str, str, dict[str, object], str]] = []
    notebooks = STAGE_NOTEBOOKS[stage]

    if stage in FIELD_STAGES:
        logging.info("Stage %s: processing %d field(s)", stage, len(fields))
        for field in fields:
            banner(f"Running {stage} stage for field {field}")
            field_failed = False
            for notebook in notebooks:
                in_path = REPO_ROOT / notebook
                if not in_path.exists():
                    failures.append((field, notebook, {}, f"Notebook file not found: {in_path}"))
                    field_failed = True
                    break

                if in_path.name == "2a_make_laplace_maps.ipynb":
                    for sizedet in (1, 10):
                        params = {"field": field, "sizedet": sizedet}
                        out_path = OUT_ROOT / stage / field / f"{in_path.stem}__{field}__sizedet{sizedet}.ipynb"
                        try:
                            execute_notebook(in_path, out_path, params)
                        except Exception as exc:  # pragma: no cover
                            logging.error("FAILED %s %s", notebook, variant_label(params))
                            logging.error("Skipping remaining notebooks for field %s", field)
                            failures.append((field, notebook, params, str(exc).strip() or "(no message)"))
                            field_failed = True
                            break
                    if failures and failures[-1][0] == field and failures[-1][1] == notebook:
                        break
                else:
                    methods = flux_methods if stage == "catalog" else ("summed_map",)
                    for flux_method in methods:
                        stage_dig_modes = dig_modes if stage == "catalog" else ("no_dig",)
                        for dig_mode in stage_dig_modes:
                            params = {"field": field}
                            if stage == "catalog":
                                params["flux_method"] = flux_method
                                params["dig_mode"] = dig_mode
                            suffix = f"__{field}"
                            if stage == "catalog":
                                suffix += f"__{flux_method}__{dig_mode}"
                            out_path = OUT_ROOT / stage / field / f"{in_path.stem}{suffix}.ipynb"
                            try:
                                execute_notebook(in_path, out_path, params)
                            except Exception as exc:  # pragma: no cover
                                logging.error("FAILED %s %s", notebook, variant_label(params))
                                logging.error("Skipping remaining notebooks for field %s", field)
                                failures.append((field, notebook, params, str(exc).strip() or "(no message)"))
                                field_failed = True
                                break
                        if failures and failures[-1][0] == field and failures[-1][1] == notebook:
                            break
                    if failures and failures[-1][0] == field and failures[-1][1] == notebook:
                        break
            if field_failed:
                logging.info("Field %s did not complete for stage %s", field, stage)
            else:
                banner(f"Completed field {field} for stage {stage}")
    else:
        banner(f"Running {stage} stage")
        for notebook in notebooks:
            in_path = REPO_ROOT / notebook
            if not in_path.exists():
                failures.append((stage, notebook, {}, f"Notebook file not found: {in_path}"))
                break
            out_path = OUT_ROOT / stage / "final" / f"{in_path.stem}.ipynb"
            methods = flux_methods if stage == "catalog" else ("summed_map",)
            for flux_method in methods:
                stage_dig_modes = dig_modes if stage == "catalog" else ("no_dig",)
                for dig_mode in stage_dig_modes:
                    params = {"flux_method": flux_method, "dig_mode": dig_mode} if stage == "catalog" else {}
                    final_out_path = out_path
                    if stage == "catalog":
                        final_out_path = OUT_ROOT / stage / "final" / f"{in_path.stem}__{flux_method}__{dig_mode}.ipynb"
                    try:
                        execute_notebook(in_path, final_out_path, params)
                    except Exception as exc:  # pragma: no cover
                        logging.error("FAILED %s %s", notebook, variant_label(params))
                        failures.append((stage, notebook, params, str(exc).strip() or "(no message)"))
                        break
                if failures and failures[-1][0] == stage and failures[-1][1] == notebook:
                    break
            if failures and failures[-1][0] == stage and failures[-1][1] == notebook:
                break
        if not failures:
            banner(f"Completed {stage} stage")

    for notebook in FINAL_NOTEBOOKS:
        if stage != "catalog" or failures:
            continue
        banner("Running final notebooks after field catalogs completed")
        for flux_method in flux_methods:
            for dig_mode in dig_modes:
                in_path = REPO_ROOT / notebook
                if not in_path.exists():
                    failures.append(("ALL_FIELDS", notebook, {}, f"Notebook file not found: {in_path}"))
                    break
                params = {"flux_method": flux_method, "dig_mode": dig_mode}
                out_path = OUT_ROOT / stage / "final" / f"{in_path.stem}__final__{flux_method}__{dig_mode}.ipynb"
                try:
                    execute_notebook(in_path, out_path, params)
                except Exception as exc:  # pragma: no cover
                    logging.error("FAILED final notebook %s %s", notebook, variant_label(params))
                    failures.append(("ALL_FIELDS", notebook, params, str(exc).strip() or "(no message)"))
                    break
            if failures:
                break
        if failures:
            break
        banner("Completed final notebooks")

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged M33 notebook pipeline.")
    parser.add_argument(
        "--stage",
        choices=("regions", "region-tests", "catalog", "plots", "resolution", "all"),
        default="all",
        help="Pipeline stage to run. 'all' runs regions, catalog, and plots. 'region-tests' writes only to region_identification_testing/.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(FIELDS),
        help="Fields to process for field-level stages.",
    )
    parser.add_argument(
        "--flux-methods",
        nargs="+",
        default=["summed_map", "integrated_spectrum"],
        choices=("summed_map", "integrated_spectrum"),
        help="Flux catalog methods to build for the catalog stage.",
    )
    parser.add_argument(
        "--dig-modes",
        nargs="+",
        default=["no_dig", "dig_subtracted"],
        choices=("no_dig", "dig_subtracted"),
        help="DIG/background handling modes to build for the catalog stage.",
    )
    parser.add_argument("--lap", type=int, default=15, help="Laplacian candidate map tag for region-tests, e.g. 15 for A_PicL15.")
    parser.add_argument(
        "--lap-sizeboxes",
        nargs="+",
        type=int,
        default=[10],
        help="Detection/Laplacian sizebox values to test with --stage region-tests. Example: --lap-sizeboxes 1 10 30",
    )
    parser.add_argument("--threshold-sizebox", type=int, default=1, help="AMP_sizd<sizebox> map used for the threshold signal in region-tests.")
    parser.add_argument("--snr-limit", type=float, default=6.0, help="Peak S/N cut for region-tests.")
    parser.add_argument("--bgbox", type=int, default=25, help="Threshold map background box for region-tests.")
    parser.add_argument("--stdbox", type=int, default=3, help="Threshold map noise box for region-tests.")
    parser.add_argument("--signoi", type=float, default=3.0, help="Threshold-map sigma-noise multiplier for region-tests.")
    parser.add_argument("--max-zoi-pc", type=int, default=100, help="Maximum ZoI radius in pc for region-tests.")
    parser.add_argument(
        "--threshold-signal-mode",
        choices=("amp_sized", "match_latest"),
        default="match_latest",
        help="Signal map used for thresholding in region-tests. match_latest reproduces current F6 handoff behavior.",
    )
    parser.add_argument(
        "--rebuild-laplacian",
        action="store_true",
        help="Run notebook 2a to rebuild candidate Laplacian maps for each requested lap sizebox before region-tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields = tuple(args.fields)
    flux_methods = tuple(args.flux_methods)
    dig_modes = tuple(args.dig_modes)
    stages = ("regions", "catalog", "plots") if args.stage == "all" else (args.stage,)
    failures: list[tuple[str, str, dict[str, object], str]] = []

    logging.info("Starting M33 notebook pipeline")
    logging.info("Stages: %s", ", ".join(stages))
    logging.info("Fields: %s", ", ".join(fields))
    logging.info("Flux methods: %s", ", ".join(flux_methods))
    logging.info("DIG modes: %s", ", ".join(dig_modes))

    if args.stage == "region-tests":
        failures = run_region_identification_tests(args)
        if failures:
            logging.error("--- Failure summary ---")
            for index, (field, notebook, params, message) in enumerate(failures, start=1):
                logging.error("%02d) field=%s notebook=%s params=%s", index, field, notebook, params)
                logging.error("    error=%s", message)
            return 1
        logging.info("No failures. Region-identification test products are in %s", REGION_TEST_ROOT)
        return 0

    for stage in stages:
        stage_failures = run_stage(stage, fields, flux_methods, dig_modes)
        failures.extend(stage_failures)
        if stage_failures and args.stage == "all":
            logging.error("Stopping pipeline after failures in %s stage", stage)
            break

    if failures:
        logging.error("--- Failure summary ---")
        for index, (field, notebook, params, message) in enumerate(failures, start=1):
            logging.error("%02d) field=%s notebook=%s params=%s", index, field, notebook, params)
            logging.error("    error=%s", message)
        return 1

    logging.info("No failures. Yay!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
