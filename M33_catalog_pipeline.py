from pathlib import Path
import papermill as pm

import logging
import sys

# --- Logging setup: show papermill streamed output (for your checkpoints)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("papermill").setLevel(logging.INFO)

FIELDS = [
    "NW", "NE", "SW", "SE",
    "F5", "F6", "F7", "F8", "F9"
]

# FIELDS = [
#      "F7"
# ]


NOTEBOOKS = [
    "2a_make_laplace_maps.ipynb",
    "2b_make_threshold_maps.ipynb",
    "3_finalize_peak_detection.ipynb",
    "4_ZoI.ipynb",
    "5_domains.ipynb",
    "6_flux_catalog_individual_field.ipynb",
    "7_create_total_M33_catalog.ipynb",
]

OUT_ROOT = Path("executed")
OUT_ROOT.mkdir(exist_ok=True)

failures = []  # (field, notebook, params, error_message)

for field in FIELDS:
    print(f"\n=== Running pipeline for {field} ===\n")
    field_out = OUT_ROOT / field
    field_out.mkdir(parents=True, exist_ok=True)

    field_failed = False  # if True, skip remaining notebooks for this field

    for nb in NOTEBOOKS:
        if field_failed:
            break

        in_path = Path(nb)

        # Optional: fail fast if notebook file is missing
        if not in_path.exists():
            msg = f"Notebook file not found: {in_path}"
            print(f"{msg}")
            failures.append((field, nb, {}, msg))
            field_failed = True
            break

        # ---- Special handling for 2a ----
        if in_path.name == "2a_make_laplace_maps.ipynb":
            for sizedet in [1, 10]:
                out_path = field_out / f"{in_path.stem}__{field}__sizedet{sizedet}.ipynb"
                params = {"field": field, "sizedet": sizedet}

                print(f"\n*****\nRunning {in_path} (sizedet={sizedet})\n*****\n")

                try:
                    pm.execute_notebook(
                        input_path=str(in_path),
                        output_path=str(out_path),
                        parameters=params,
                        # Turn this ON if you want checkpoint prints to appear in terminal:
                        log_output=False,
                        # Optional: show cell count progress like "7/20"
                        progress_bar=True,
                    )
                except Exception as e:
                    msg = str(e).strip() or "(no message)"
                    print(f"FAILED field={field} notebook={nb} sizedet={sizedet}")
                    print(f"Skipping remaining notebooks for this field :(")
                    failures.append((field, nb, params, msg))
                    field_failed = True
                    break  # stop sizedet loop and move to next field

        # ---- All other notebooks ----
        else:
            out_path = field_out / f"{in_path.stem}__{field}.ipynb"
            params = {"field": field}

            print(f"\n*****\nRunning {in_path}\n*****\n")

            try:
                pm.execute_notebook(
                    input_path=str(in_path),
                    output_path=str(out_path),
                    parameters=params,
                    # Turn this ON if you want checkpoint prints to appear in terminal:
                    log_output=False,
                    progress_bar=True,
                )
            except Exception as e:
                msg = str(e).strip() or "(no message)"
                print(f"\nFAILED field={field} notebook={nb}")
                print(f"Skipping remaining notebooks for this field :(")
                failures.append((field, nb, params, msg))
                field_failed = True
                break  # move to next field

    if not field_failed:
        print(f"\n*******\nCompleted field {field}. Yippee\n*******\n")

print("\nAll fields attempted.")

if failures:
    print("\n--- Failure summary ---")
    for i, (field, nb, params, msg) in enumerate(failures, start=1):
        print(f"{i:02d}) field={field} notebook={nb} params={params}")
        print(f"    error={msg}")
else:
    print("\nNo failures. Woohoo!!!!!")