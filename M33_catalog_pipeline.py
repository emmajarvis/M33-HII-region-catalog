from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import papermill as pm

from m33_pipeline.config import FIELDS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("papermill").setLevel(logging.INFO)


REPO_ROOT = Path(__file__).resolve().parent
OUT_ROOT = REPO_ROOT / "executed"

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
    "03_paper_plots_and_muse/1_plot_maps.ipynb",
    "03_paper_plots_and_muse/8_catalog_plots.ipynb",
    "03_paper_plots_and_muse/9_catalog_comparison.ipynb",
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
        choices=("regions", "catalog", "plots", "resolution", "all"),
        default="all",
        help="Pipeline stage to run. 'all' runs regions, catalog, and plots.",
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
