# Resolution Degradation Pipeline

This folder now has a four-stage build pipeline plus one plotting notebook:

1. `01_degrade_maps.ipynb`
2. `02_identify_peaks.ipynb`
3. `03_build_regions.ipynb`
4. `04_build_catalogs.ipynb`
5. `05_plot_resolution_results.ipynb`

The first four stages are thin wrappers around `m33_pipeline.resolution_degradation`, so each stage can be run independently as long as its inputs already exist.

Plotting is now intentionally separated from the terminal pipeline. The command-line runner only builds degraded maps, peaks, regions, and catalogs. All figure-making code lives in `05_plot_resolution_results.ipynb` so plots can be edited interactively in one place.

Outputs are written under:

- `04_resolution_degradation/degraded_fits/`
- `04_resolution_degradation/degraded_peaks/`
- `04_resolution_degradation/region_products/`
- `04_resolution_degradation/catalog_products/`
- `04_resolution_degradation/plots/`
- `04_resolution_degradation/qc/`

You can also run the pipeline from the command line:

```bash
python run_resolution_degradation_pipeline.py --stage all
python run_resolution_degradation_pipeline.py --stage catalogs
```

Then open `05_plot_resolution_results.ipynb` to load the saved products and make:

- peak subplot figures for each field
- boundary subplot figures for each field
- luminosity-radius comparisons
- radial gradients in metallicity, density, and ionization parameter

Legacy exploratory notebooks are still kept in this folder:

- `1-NWMake_map_M33-resolution.ipynb`
- `10_resolution_comparison.ipynb`
