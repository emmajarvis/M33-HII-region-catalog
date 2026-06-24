#NOTE THAT THIS IS RUN ON CANFAR, NOT INTEGRATED INTO OF PIPELINE
#Run this for each field on CANFAR, then download the .ascii file for each field and then run the notebook 6 choosing flux_methods = integrated_spectrum

import logging
# logging.disable(logging.CRITICAL)
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime

import sys

from integrated_spectra_functions import (
    build_empty_flux_table,
    load_context,
    run_region_fit,
)

WORKER_CTX = None
WORKER_PLOTSHOW = False


def init_worker(field, base_output_folder, output_timestamp, plotshow):
    global WORKER_CTX, WORKER_PLOTSHOW
    WORKER_CTX = load_context(
        field,
        base_output_folder=base_output_folder,
        output_timestamp=output_timestamp
    )
    WORKER_PLOTSHOW = plotshow


def process_region(ireg):
    global WORKER_CTX, WORKER_PLOTSHOW

    try:
        result = run_region_fit(WORKER_CTX, ireg, plotshow=WORKER_PLOTSHOW)
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


VALID_FIELDS = {'NW', 'NE', 'SE', 'SW', '7', '8', '9'}


def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python integrated_spectra_run.py <FIELD>")
        print("Example: python integrated_spectra_run.py SE")
        print("Valid fields:", ", ".join(sorted(VALID_FIELDS)))
        sys.exit(1)

    field = str(sys.argv[1])

    if field not in VALID_FIELDS:
        print(f"Invalid field: {field}")
        print("Valid fields:", ", ".join(sorted(VALID_FIELDS))) 
        sys.exit(1)

    return field


def main():
    
    field = parse_args()

    # change this if you do not want interactive plot windows
    plotshow = False

    # One timestamp for the whole run, shared by the main process and all workers.
    output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_date = output_timestamp.split("_")[0]

    # This folder name is now automatic. No manual date edits needed.
    base_output_folder = f"5-Intspec_Updated_{output_date}"

    ctx = load_context(
        field,
        base_output_folder=base_output_folder,
        output_timestamp=output_timestamp
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
        initargs=(field, base_output_folder, output_timestamp, plotshow),
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


    print()
    print(f"Finished field {ctx.field}")
    print(f"Successful fits: {len(can_fit)}")
    print(f"Failed fits: {len(cant_fit)}")
    print(f"Output table: {ctx.folder}/{ctx.field}_Data_flux.ascii")


if __name__ == "__main__":
    main()