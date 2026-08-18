from __future__ import annotations

import argparse
from pathlib import Path

from m33_pipeline.config import FIELDS

from .injections import InjectionConfig
from .recovery import run_trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject false HII regions and measure peak-identification recovery.")
    parser.add_argument("--fields", nargs="+", default=FIELDS, help="Fields to process.")
    parser.add_argument("--trials", type=int, default=100, help="Number of repeated trials per field.")
    parser.add_argument("--sources-per-field", type=int, default=100, help="False sources injected per field per trial.")
    parser.add_argument("--seed", type=int, default=20260731, help="Random seed for reproducible injections.")
    parser.add_argument("--output-dir", default="peak_completeness/results", help="Directory for recovery tables and optional injected FITS maps.")
    parser.add_argument("--template-catalog", default=None, help="Catalog used to sample sizes, fluxes, ratios, BPT class, and metallicities.")
    parser.add_argument("--match-radius-px", type=float, default=10.0, help="Maximum distance between injected center and detected peak.")
    parser.add_argument("--min-distance-px", type=float, default=25.0, help="Minimum placement distance from existing/injected peaks.")
    parser.add_argument(
        "--sampling-mode",
        choices=["catalog", "independent"],
        default="catalog",
        help="catalog samples whole observed regions; independent varies only radius, Halpha luminosity, and OIII/Halpha.",
    )
    parser.add_argument("--radius-min-px", type=float, default=None, help="Minimum injected radius in pixels for independent sampling.")
    parser.add_argument("--radius-max-px", type=float, default=None, help="Maximum injected radius in pixels for independent sampling.")
    parser.add_argument("--radius-min-pc", type=float, default=None, help="Minimum injected 3-sigma radius in pc for independent sampling.")
    parser.add_argument("--radius-max-pc", type=float, default=None, help="Maximum injected 3-sigma radius in pc for independent sampling.")
    parser.add_argument("--pc-per-px", type=float, default=None, help="Physical pixel scale for pc radius conversion. Defaults to the template catalog median.")
    parser.add_argument("--log-lha-min", type=float, default=None, help="Minimum log10 Halpha luminosity for independent sampling.")
    parser.add_argument("--log-lha-max", type=float, default=None, help="Maximum log10 Halpha luminosity for independent sampling.")
    parser.add_argument("--oiii-ha-mode", choices=["sample", "uniform"], default="sample", help="Sample OIII/Halpha from catalog or uniformly in log space.")
    parser.add_argument("--log-oiii-ha-min", type=float, default=-1.5, help="Minimum log10(OIII/Halpha) for uniform/clipped sampling.")
    parser.add_argument("--log-oiii-ha-max", type=float, default=1.0, help="Maximum log10(OIII/Halpha) for uniform/clipped sampling.")
    parser.add_argument(
        "--morphologies",
        nargs="+",
        default=["gaussian"],
        choices=["gaussian", "ring", "clumpy", "overlap_pair", "custom", "pseudo_voigt", "pseudo_voight", "pseudo-voigt", "pseudo-voight"],
        help="Injected source morphologies to sample. Multiple values are sampled with equal probability.",
    )
    parser.add_argument(
        "--custom-profile-path",
        default=None,
        help="CSV profile for custom morphology. Defaults to peak_completeness/average_region_profile/average_region_profile.csv.",
    )
    parser.add_argument("--pseudo-voigt-sigma-over-r50", type=float, default=0.96852, help="Gaussian sigma term for pseudo-Voigt injections, in r/r50 units.")
    parser.add_argument("--pseudo-voigt-gamma-over-r50", type=float, default=0.14786, help="Lorentzian gamma term for pseudo-Voigt injections, in r/r50 units.")
    parser.add_argument("--pseudo-voigt-eta", type=float, default=0.10307, help="Lorentzian mixing fraction for pseudo-Voigt injections; 0 is Gaussian, 1 is Lorentzian.")
    parser.add_argument("--pseudo-voigt-rmax-over-r50", type=float, default=4.0, help="Radial cutoff for pseudo-Voigt injections, in r/r50 units.")
    parser.add_argument("--clump-min-count", type=int, default=3, help="Minimum number of clumps for clumpy injected regions.")
    parser.add_argument("--clump-max-count", type=int, default=6, help="Maximum number of clumps for clumpy injected regions.")
    parser.add_argument("--overlap-sep-min-sigma", type=float, default=0.7, help="Minimum overlap-pair separation in units of source sigma.")
    parser.add_argument("--overlap-sep-max-sigma", type=float, default=1.8, help="Maximum overlap-pair separation in units of source sigma.")
    parser.add_argument("--save-injected-fits", action="store_true", help="Save injected Halpha+OIII amplitude maps for each trial.")
    parser.add_argument("--save-injection-plots", action="store_true", help="Save PNG maps showing injected sources, recovered sources, and missed sources.")
    parser.add_argument("--show-injection-plots", action="store_true", help="Display injection maps interactively while running.")
    parser.add_argument("--plot-dir", default=None, help="Folder for injection recovery PNG maps. Defaults to <output-dir>/injection_maps.")
    parser.add_argument("--save-distribution-plots", action="store_true", help="Save recovered-vs-missed distribution plots for all varied properties.")
    parser.add_argument("--distribution-plot-dir", default=None, help="Folder for distribution plots. Defaults to <output-dir>/distribution_plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    injection_config = InjectionConfig(
        n_sources=args.sources_per_field,
        min_distance_px=args.min_distance_px,
        sampling_mode=args.sampling_mode,
        radius_min_px=args.radius_min_px,
        radius_max_px=args.radius_max_px,
        radius_min_pc=args.radius_min_pc,
        radius_max_pc=args.radius_max_pc,
        pc_per_px=args.pc_per_px,
        log_lha_min=args.log_lha_min,
        log_lha_max=args.log_lha_max,
        oiii_ha_mode=args.oiii_ha_mode,
        log_oiii_ha_min=args.log_oiii_ha_min,
        log_oiii_ha_max=args.log_oiii_ha_max,
        morphologies=tuple(args.morphologies),
        clump_min_count=args.clump_min_count,
        clump_max_count=args.clump_max_count,
        overlap_sep_min_sigma=args.overlap_sep_min_sigma,
        overlap_sep_max_sigma=args.overlap_sep_max_sigma,
        custom_profile_path=args.custom_profile_path,
        pseudo_voigt_sigma_over_r50=args.pseudo_voigt_sigma_over_r50,
        pseudo_voigt_gamma_over_r50=args.pseudo_voigt_gamma_over_r50,
        pseudo_voigt_eta=args.pseudo_voigt_eta,
        pseudo_voigt_rmax_over_r50=args.pseudo_voigt_rmax_over_r50,
    )
    run_trials(
        list(args.fields),
        n_trials=args.trials,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        injection_config=injection_config,
        match_radius_px=args.match_radius_px,
        save_injected_fits=args.save_injected_fits,
        save_injection_plots=args.save_injection_plots,
        show_injection_plots=args.show_injection_plots,
        plot_dir=args.plot_dir,
        save_distribution_plots_flag=args.save_distribution_plots,
        distribution_plot_dir=args.distribution_plot_dir,
        template_catalog_path=args.template_catalog,
    )


if __name__ == "__main__":
    main()
