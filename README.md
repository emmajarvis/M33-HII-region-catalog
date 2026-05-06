# Generate the HII region catalog for M33

The notebooks are organized into four pipeline stages:

- `01_region_identification/`: field-by-field region finding through final boundary maps
- `02_catalog_values/`: field catalogs plus the merged catalog products
- `03_paper_plots_and_muse/`: paper figures and MUSE comparison notebooks
- `04_resolution_degradation/`: degraded-resolution experiments

Each notebook now resets its working directory to the repository root when it starts, so the notebooks can be run from their stage folder without breaking relative paths.

To run the full staged pipeline:

```bash
python M33_catalog_pipeline.py
```

To run one stage independently:

```bash
python M33_catalog_pipeline.py --stage regions
python M33_catalog_pipeline.py --stage catalog
python M33_catalog_pipeline.py --stage plots
python M33_catalog_pipeline.py --stage resolution
```

To restrict field-based stages to a subset of fields:

```bash
python M33_catalog_pipeline.py --stage regions --fields NW NE
```

You can still run notebooks individually; for field-level notebooks, change the `field` parameter at the top or use Papermill parameters.
