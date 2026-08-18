# Generate the HII Region Catalog for M33

This repository contains the notebook and terminal pipeline used to identify M33
emission regions, build flux catalogs, make paper plots, compare to external
data, and run the resolution/completeness tests.

Most production steps can be run from the terminal through wrapper scripts.
Some calibration, visual inspection, and CANFAR cube-extraction steps are still
intentionally notebook/manual driven.

## Repository Layout

- `01_region_identification/`: field-by-field peak finding, zones of influence,
  and final boundary maps.
- `02_catalog_values/`: per-field flux catalogs, integrated-spectrum tools, and
  merged final catalogs.
- `03_paper_plots_and_muse/`: paper figures, MUSE comparison, BPT and
  metallicity exploration.
- `04_resolution_degradation/`: degraded-resolution maps, peaks, regions,
  catalogs, and plots.
- `peak_completeness/`: injection/recovery completeness experiments.
- `m33_pipeline/`: shared Python helpers used by the notebooks and terminal
  runners.
- `region_identification_testing/`: outputs from peak/boundary parameter tests.

Each notebook resets its working directory to the repository root when it
starts, so notebooks can be launched from their stage folder without breaking
relative paths.

## Terminal vs Notebook

Run these from the terminal:

- Full notebook pipeline: `M33_catalog_pipeline.py`
- Region-parameter tests: `M33_catalog_pipeline.py --stage region-tests`
- CANFAR integrated-spectrum extraction: `02_catalog_values/6a_get_integrated_spectrum/integrated_spectra_run.py`
- Resolution-degradation build stages: `run_resolution_degradation_pipeline.py`
- Peak-completeness injection/recovery tests: `python -m peak_completeness.run_peak_completeness`

Run these interactively as notebooks:

- `_00_calibrate_maps.ipynb`: calibration/checking workflow.
- `01_region_identification/0_align_maps_callibrate.ipynb`: map alignment and
  calibration checks, including the special F9 map alignment.
- `01_region_identification/0_aligment_cubes.ipynb`: cube alignment parameter
  measurement.
- `04_resolution_degradation/05_plot_resolution_results.ipynb`: resolution
  figures after terminal products exist.
- Exploratory paper/MUSE/metallicity notebooks in `03_paper_plots_and_muse/`.
- `peak_completeness/build_average_region_profile.ipynb`: build empirical
  average profiles before `custom` completeness injections.

## Main Pipeline

DO NOT run the whole pipeline at once, has to be done in stages.

Run one stage:

```bash
python M33_catalog_pipeline.py --stage regions
python M33_catalog_pipeline.py --stage catalog
python M33_catalog_pipeline.py --stage plots
python M33_catalog_pipeline.py --stage resolution
```

First, the regions stage gets the boundaries. Then, upload those files to CANFAR to get the integrated fluxes from the cubes. Then, upload those .ascii files to the CATALOGS folder.

Second, run the catlog stage to get the full catalog with extinction corrected fluxes and then all the derived quantities.

Then, for plotting it is best to just run the plotting notebook (8) and 9 for MUSE comparison.

You can also run only selected fields for field-level stages, here are some examples of this:

```bash
python M33_catalog_pipeline.py --stage regions --fields NW NE SE SW F5 F6 F7 F8 F9
python M33_catalog_pipeline.py --stage regions --fields NW NE
python M33_catalog_pipeline.py --stage catalog --fields F7 F8 F9
```

There are a few catalog-stage variations. If you don't specify it will do all of them. Probably just want to use integrated_spectrum and dig_subtracted. The summed_map is just for quick tests when you don't want to get the final integrated fluxes by fitting cubes on canfar, if just sums the fluxes within the boundaries (this is just for testing code, not final results).

```bash
python M33_catalog_pipeline.py --stage catalog --flux-methods summed_map
python M33_catalog_pipeline.py --stage catalog --flux-methods integrated_spectrum
python M33_catalog_pipeline.py --stage catalog --dig-modes no_dig
python M33_catalog_pipeline.py --stage catalog --dig-modes dig_subtracted
python M33_catalog_pipeline.py --stage catalog --flux-methods summed_map integrated_spectrum --dig-modes no_dig dig_subtracted
```

The region stage runs these notebooks per field:

1. `01_region_identification/2a_make_laplace_maps.ipynb` with `sizedet=1` and
   `sizedet=10`
2. `01_region_identification/2b_make_threshold_maps.ipynb`
3. `01_region_identification/3_finalize_peak_detection.ipynb`
4. `01_region_identification/4_ZoI.ipynb`
5. `01_region_identification/5_domains.ipynb`

If you want to tweak parameters, you can just run these notebooks individually.

The catalog stage runs:

1. `02_catalog_values/6_flux_catalog_individual_field.ipynb` per field
2. `02_catalog_values/7_create_total_M33_catalog.ipynb` after all field catalogs
   complete

## Region-Identification Parameter Tests

These tests write only to `region_identification_testing/` and do not overwrite
the production catalog products.

Default test:

```bash
python M33_catalog_pipeline.py --stage region-tests
```

Test several Laplacian/detection box sizes:

```bash
python M33_catalog_pipeline.py --stage region-tests --lap-sizeboxes 1 5 10 15
```

Rebuild the candidate Laplacian maps before testing:

```bash
python M33_catalog_pipeline.py --stage region-tests --lap-sizeboxes 5 15 --rebuild-laplacian
```

Vary threshold and pruning parameters:

```bash
python M33_catalog_pipeline.py --stage region-tests \
  --fields NW F5 F6 \
  --lap 15 \
  --lap-sizeboxes 5 10 \
  --threshold-sizebox 1 \
  --snr-limit 6 \
  --bgbox 25 \
  --stdbox 3 \
  --signoi 3 \
  --max-zoi-pc 100 \
  --threshold-signal-mode match_latest
```

Use `--threshold-signal-mode amp_sized` to force the sized amplitude map for
all fields. `match_latest` reproduces the current F6 handoff behavior.

## Cube Alignment and Integrated Spectra on CANFAR

Cube alignment parameters are measured manually with this file, but this is not run locally:

```bash
jupyter notebook 01_region_identification/0_aligment_cubes.ipynb
```

The notebook writes files like:

```text
M33_alignments/parametres_align_cube_corrige_SN1_F7.txt
M33_alignments/parametres_align_cube_corrige_SN2_F7.txt
```

Upload the boundary maps and alignment-parameter folder to CANFAR, then run the
integrated-spectrum terminal runner from:

```bash
cd 02_catalog_values/6a_get_integrated_spectrum
```

Basic run with cube-boundary alignment (DO THIS ON CANFAR, or wherever cubes and ORCS are installed):

```bash
python integrated_spectra_run.py F7 --alignment-folder ~/M33_alignments
```

Run only selected alignment diagnostic plots:

```bash
python integrated_spectra_run.py F7 \
  --alignment-folder ~/M33_alignments \
  --alignment-diagnostic-regions 1,12,45
```

Save full integrated-spectrum plots for every region:

```bash
python integrated_spectra_run.py F7 \
  --alignment-folder ~/M33_alignments \
  --make-spectrum-plots
```

Save both selected alignment diagnostics and all spectrum plots:

Valid field inputs are `NW`, `NE`, `SE`, `SW`, `F5`, `F6`, `F7`, `F8`, and
`F9`. Bare numeric fields such as `7` also work.


Region IDs in `--alignment-diagnostic-regions` are the zero-based internal
`ireg` indices used by the code.

### F9 Special Case

F9 has a large map/cube offset that was measured in
`01_region_identification/0_align_maps_callibrate.ipynb` using `astroalign`.
For integrated spectra, the F9 correction is automatic. The code samples the
SN3 boundary map back onto the original SN1 and SN2 cube grids, then keeps only
regions that have boundary pixels in SN1, SN2, and SN3.

Run F9 like this:

```bash
python integrated_spectra_run.py F9
```
You do not need `--alignment-folder` for the F9 special map shift.

## Resolution-Degradation Pipeline

The preferred terminal runner is:

```bash
python run_resolution_degradation_pipeline.py --stage all
```

Run individual resolution stages:

```bash
python run_resolution_degradation_pipeline.py --stage degrade
python run_resolution_degradation_pipeline.py --stage peaks
python run_resolution_degradation_pipeline.py --stage regions
python run_resolution_degradation_pipeline.py --stage catalogs
```

Restrict fields or distances:

```bash
python run_resolution_degradation_pipeline.py --stage all --fields NW NE
python run_resolution_degradation_pipeline.py --stage all --distances 1.0 2.0 5.0 10.0
python run_resolution_degradation_pipeline.py --stage catalogs --fields F7 F8 F9 --distances 5.0
```

Terminal stages write products under:

- `04_resolution_degradation/degraded_fits/`
- `04_resolution_degradation/degraded_peaks/`
- `04_resolution_degradation/region_products/`
- `04_resolution_degradation/catalog_products/`
- `04_resolution_degradation/qc/`

After terminal products exist, open this notebook to make/edit figures:

```bash
jupyter notebook 04_resolution_degradation/05_plot_resolution_results.ipynb
```

Do not use these notebooks, they were old tests:

- `04_resolution_degradation/1-NWMake_map_M33-resolution.ipynb`
- `04_resolution_degradation/10_resolution_comparison.ipynb`

## Peak Completeness Tests

Build the empirical average profile first if using `custom` morphology:

```bash
jupyter notebook peak_completeness/build_average_region_profile.ipynb
```

Full default experiment:

```bash
python -m peak_completeness.run_peak_completeness --trials 100 --sources-per-field 100
```

Mixed morphologies (only do gaussian morphology for final one, others are just test):

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies gaussian ring clumpy overlap_pair
```

Empirical-profile injections (these are based on the data, again just a test):

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies custom \
  --output-dir peak_completeness/results_custom
```

Pseudo-Voigt injections:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies pseudo_voigt \
  --output-dir peak_completeness/results_pseudo_voigt
```

Save injection maps for inspection:

```bash
python -m peak_completeness.run_peak_completeness \
  --fields NW \
  --trials 1 \
  --sources-per-field 100 \
  --save-injection-plots
```

Save recovered-vs-missed distribution plots:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --save-distribution-plots
```

Completeness outputs are written under `peak_completeness/results/` by default.

## Other Notebooks and Checks

Manual/interactive notebooks:

```bash
jupyter notebook _00_calibrate_maps.ipynb
jupyter notebook 01_region_identification/0_align_maps_callibrate.ipynb
jupyter notebook 01_region_identification/0_aligment_cubes.ipynb
jupyter notebook 01_region_identification/completeness-test.ipynb
jupyter notebook 03_paper_plots_and_muse/8_catalog_plots.ipynb
jupyter notebook 03_paper_plots_and_muse/9_catalog_comparison.ipynb
jupyter notebook 03_paper_plots_and_muse/BPT_comparison.ipynb
jupyter notebook 03_paper_plots_and_muse/metallicities.ipynb
```

The only one needed is the metallicities.ipynb which does updated metallicity calibrations for plotting rather than the full set of calibrations used in the main 8_catalog_plots.ipynb file.

Run the Python t

## Notes

- The repo currently contains many generated products and executed notebooks.
  Check `git status` before committing to avoid mixing regenerated outputs with
  code changes.
- On local machines without CANFAR packages, the integrated-spectrum code may
  not be able to load cubes. Run that extraction on CANFAR where `orcs`, `orb`,
  and the `/arc/projects/signals/M33/...` cube paths exist.
