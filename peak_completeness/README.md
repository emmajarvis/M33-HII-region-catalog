# Peak Completeness Tests

This folder contains terminal-run tools for injecting false HII-region-like
sources into the Halpha+OIII amplitude maps used by the peak-identification
notebooks.

The code mirrors the final peak-identification notebook:

- build Laplacian candidate peaks from the injected Halpha+OIII amplitude map
- require `SNR >= 6`
- require positive excess over the existing threshold map
- match detected peaks back to the injected source centers

By default, injected sources are symmetric 2D Gaussians. The runner can also
inject non-Gaussian morphologies:

- `ring`: annular shell-like emission
- `clumpy`: several compact clumps within one region
- `overlap_pair`: two separate injected regions with partially overlapping
  footprints, to test line-of-sight blending
- `custom`: circularized empirical radial profile measured from the final
  catalog regions rather than an analytic Gaussian
- `pseudo_voigt`: circular pseudo-Voigt radial profile using the average-profile
  fit parameters by default

Build the empirical profile before using `custom`:

```bash
jupyter notebook peak_completeness/build_average_region_profile.ipynb
```

The notebook writes `peak_completeness/average_region_profile/average_region_profile.csv`,
model-fit comparisons, steepest/flattest examples, and simple profile-shape
groups.

Run a quick smoke test:

```bash
python -m peak_completeness.run_peak_completeness --fields NW --trials 1 --sources-per-field 5
```

Run the full requested experiment:

```bash
python -m peak_completeness.run_peak_completeness --trials 100 --sources-per-field 100
```

Run a mixed-morphology experiment:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies gaussian ring clumpy overlap_pair
```

Run custom empirical-profile injections:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies custom \
  --output-dir peak_completeness/results_custom
```

Compare Gaussian and empirical-profile injections in the same run:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies gaussian custom \
  --output-dir peak_completeness/results_gaussian_custom
```

Run pseudo-Voigt profile injections:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --morphologies pseudo_voigt \
  --output-dir peak_completeness/results_pseudo_voigt
```

The default pseudo-Voigt parameters come from the average-profile fit. Override
them with:

```bash
--pseudo-voigt-sigma-over-r50 0.96852
--pseudo-voigt-gamma-over-r50 0.14786
--pseudo-voigt-eta 0.10307
--pseudo-voigt-rmax-over-r50 4.0
```

Run the simplified experiment where the injected sources vary only in radius,
Halpha luminosity, and OIII/Halpha:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --sampling-mode independent \
  --morphologies gaussian \
  --log-oiii-ha-min -1.5 \
  --log-oiii-ha-max 1.0 \
  --save-distribution-plots
```

In `independent` mode, the default radius and Halpha-luminosity ranges are
estimated from the observed catalog. Override them with:

```bash
--radius-min-px 2 --radius-max-px 25
--log-lha-min 34.5 --log-lha-max 37.5
```

For Gaussian regions, the physical-radius options define the 3-sigma contour
radius. The Halpha peak amplitude is normalized so the discrete pixel sum
inside that 3-sigma radius has the sampled total Halpha luminosity:

```bash
python -m peak_completeness.run_peak_completeness \
  --fields NW \
  --trials 100 \
  --sources-per-field 100 \
  --sampling-mode independent \
  --morphologies gaussian \
  --radius-min-pc 0 \
  --radius-max-pc 50 \
  --log-lha-min 30 \
  --log-lha-max 42 \
  --output-dir peak_completeness/results_gaussian_pc_lum
```

Use uniform OIII/Halpha sampling instead of catalog-sampled ratios:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --oiii-ha-mode uniform \
  --log-oiii-ha-min -1.5 \
  --log-oiii-ha-max 1.0
```

Main outputs are written to `peak_completeness/results/`:

- `all_injected_sources_recovery.csv`: one row per injected source, with size,
  surface brightness, OIII/Halpha ratio, BPT class, metallicity columns, and
  recovery flag. It also includes `morphology`,
  `n_detected_peaks_in_region`, and `multiple_peaks_detected`.
- `all_detected_peaks.csv`: all peaks detected in the injected maps
- `recovery_summary_by_field_trial.csv`: recovery fraction per field and trial
- `recovery_summary_by_field.csv`: aggregate recovery by field
- `recovery_summary_by_morphology.csv`: recovery fractions and multiple-peak
  fractions by morphology
- `overlap_pair_recovery_summary.csv`: pair-level outcomes for partially
  overlapping injected regions, including whether the pair was merged into one
  detected peak or resolved as multiple peaks
- `recovery_bias_numeric_by_field.csv`: recovered-vs-missed medians and
  distribution tests for size, luminosity, surface brightness, OIII/Halpha,
  BPT-line ratios, metallicity, ionization parameter, density, and radius
- `recovery_bias_numeric_ranked.csv`: the same numeric comparisons sorted to
  highlight the strongest recovered-vs-missed differences within each field
- `recovery_bias_categorical_by_field.csv`: recovery fractions by categorical
  properties such as BPT class
- `summary_results_manifest.csv`: quick lookup table mapping each science
  question to the CSV file and columns to inspect
- `summary_overall_recovery.csv`: total recovery percentage across all fields
  and trials for the run
- `summary_recovery_by_field_total.csv`: total recovery percentage per field
  across all trials
- `summary_property_dependence_bins.csv`: binned recovery fractions versus
  Halpha luminosity, surface brightness, peak intensity, size, OIII/Halpha,
  Halpha/OIII, local background/threshold, and distance to nearest existing peak
- `summary_luminosity_completeness_limits.csv`: estimated Halpha luminosity
  limits for 50%, 80%, and 90% recovery
- `summary_peak_offset_statistics.csv`: statistics for how far detected peaks
  lie from injected centers

Injected FITS maps are not saved by default because a full run would produce
900 maps. Add `--save-injected-fits` if you want per-trial maps for inspection.

Save map figures showing missed and recovered injected sources:

```bash
python -m peak_completeness.run_peak_completeness \
  --fields NW \
  --trials 1 \
  --sources-per-field 100 \
  --save-injection-plots
```

By default, map PNGs are saved in `peak_completeness/results/injection_maps/`.
Use `--plot-dir some/folder` to choose a different folder:

```bash
python -m peak_completeness.run_peak_completeness \
  --fields NW \
  --trials 1 \
  --sources-per-field 100 \
  --save-injection-plots \
  --plot-dir peak_completeness/injection_map_plots
```

The plotted markers are green circles for recovered injected centers, red
circles for missed injected centers, orange x markers for every detected peak
inside an injected source footprint, and blue plus signs for the nearest
matched detected peak. Thin blue lines connect each recovered injected center
to its matched detected peak. If any injected regions produce multiple detected
peaks, an additional `*_multiple_peak_zooms.png` file is saved for that trial.

Save recovered-vs-missed distribution plots for every varied property:

```bash
python -m peak_completeness.run_peak_completeness \
  --trials 100 \
  --sources-per-field 100 \
  --save-distribution-plots
```

By default, these plots are saved under
`peak_completeness/results/distribution_plots/`. Each property gets an
all-fields histogram/bar plot and a by-field faceted plot. Use
`--distribution-plot-dir some/folder` to choose a different folder.
