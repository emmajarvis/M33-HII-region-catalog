#NOTE THAT THIS IS RUN ON CANFAR, NOT INTEGRATED INTO OF PIPELINE
#Run this for each field on CANFAR, then download the .ascii file for each field and then run the notebook 6 choosing flux_methods = integrated_spectrum

import logging
# logging.disable(logging.CRITICAL)
import argparse
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime

from integrated_spectra_functions import (
    build_empty_flux_table,
    load_context,
    run_region_fit,
    normalize_field,
)

WORKER_CTX = None
WORKER_PLOTSHOW = False
WORKER_MAKE_SPECTRUM_PLOT = False


def init_worker(
    field,
    base_output_folder,
    output_timestamp,
    plotshow,
    alignment_folder,
    alignment_diagnostic_regions,
    make_spectrum_plot,
):
    global WORKER_CTX, WORKER_PLOTSHOW, WORKER_MAKE_SPECTRUM_PLOT
    WORKER_CTX = load_context(
        field,
        base_output_folder=base_output_folder,
        output_timestamp=output_timestamp,
        alignment_folder=alignment_folder,
        alignment_diagnostic_regions=alignment_diagnostic_regions,
    )
    WORKER_PLOTSHOW = plotshow
    WORKER_MAKE_SPECTRUM_PLOT = make_spectrum_plot


def process_region(ireg):
    global WORKER_CTX, WORKER_PLOTSHOW, WORKER_MAKE_SPECTRUM_PLOT

    try:
        result = run_region_fit(
            WORKER_CTX,
            ireg,
            plotshow=WORKER_PLOTSHOW,
            make_spectrum_plot=WORKER_MAKE_SPECTRUM_PLOT,
        )
        return {
            "ok": True,
            "result": result,
            "error": None,
            "ireg": ireg,
        }
    except Exception as e:
        return {
            "ok": False,
            "result": None,
            "error": str(e),
            "ireg": ireg,
        }


VALID_FIELDS = {'NW', 'NE', 'SE', 'SW', '5', '6', '7', '8', '9'}


def parse_region_list(text):
    if not text:
        return None
    return {int(item.strip()) for item in text.split(',') if item.strip()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit integrated spectra for one M33 field on CANFAR."
    )
    parser.add_argument("field", help="Field name, e.g. SE, NW, 7, or F7")
    parser.add_argument(
        "--alignment-folder",
        default=None,
        help=(
            "Folder containing parametres_align_cube_corrige_SN1_FIELD.txt and "
            "parametres_align_cube_corrige_SN2_FIELD.txt. '~' is supported."
        ),
    )
    parser.add_argument(
        "--alignment-diagnostic-regions",
        default=None,
        help="Comma-separated region IDs for boundary-over-flux alignment plots, e.g. 12,45,103.",
    )
    parser.add_argument(
        "--make-spectrum-plots",
        action="store_true",
        help="Save the full integrated-spectrum plot for every fitted region.",
    )
    parser.add_argument(
        "--plotshow",
        action="store_true",
        help="Show interactive plot windows in addition to saving plots.",
    )
    args = parser.parse_args()

    args.field = normalize_field(args.field)

    if args.field not in VALID_FIELDS:
        parser.error(f"Invalid field: {args.field}. Valid fields: {', '.join(sorted(VALID_FIELDS))}")

    args.alignment_diagnostic_regions = parse_region_list(args.alignment_diagnostic_regions)
    return args


def main():
    
    args = parse_args()
    field = args.field

    # One timestamp for the whole run, shared by the main process and all workers.
    output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_date = output_timestamp.split("_")[0]

    # This folder name is now automatic. No manual date edits needed.
    base_output_folder = f"5-Intspec_Updated_{output_date}"

    ctx = load_context(
        field,
        base_output_folder=base_output_folder,
        output_timestamp=output_timestamp,
        alignment_folder=args.alignment_folder,
        alignment_diagnostic_regions=args.alignment_diagnostic_regions,
    )
    
    data_table_flux = build_empty_flux_table()
    
    indices = ctx.ipic

    can_fit = []
    cant_fit = []

    nproc = 4   # change this
    write_every = 25
    
    with mp.Pool(
        processes=nproc,
        initializer=init_worker,
        initargs=(
            field,
            base_output_folder,
            output_timestamp,
            args.plotshow,
            args.alignment_folder,
            args.alignment_diagnostic_regions,
            args.make_spectrum_plots,
        ),
    ) as pool:
        for n_done, out in enumerate(
            tqdm(pool.imap_unordered(process_region, indices),
                 total=len(indices),
                 desc=f"Field {field}",
                 unit="region"),
            start=1
        ):
            ireg = out["ireg"]

            if out["ok"]:
                result = out["result"]

                if result["fit_success"]:
                    can_fit.append(ireg)
                else:
                    cant_fit.append(ireg)

                data_table_flux.add_row(result["row"])
            else:
                print(f"\nRegion {ireg} failed: {out['error']}")
                cant_fit.append(ireg)

            if n_done % write_every == 0:
                data_table_flux.write(
                    f"{ctx.folder}/{ctx.field}_Data_flux.ascii",
                    overwrite=True,
                    format='ascii'
                )

    data_table_flux.write(
        f"{ctx.folder}/{ctx.field}_Data_flux.ascii",
        overwrite=True,
        format='ascii'
    )

    print()
    print(f"Finished field {ctx.field}")
    print(f"Successful fits: {len(can_fit)}")
    print(f"Failed fits: {len(cant_fit)}")
    print(f"Output table: {ctx.folder}/{ctx.field}_Data_flux.ascii")


if __name__ == "__main__":
    main()
