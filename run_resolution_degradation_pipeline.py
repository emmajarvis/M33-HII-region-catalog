from __future__ import annotations

import argparse

from m33_pipeline.resolution_degradation import (
    DEFAULT_DISTANCES_MPC,
    FIELDS,
    build_regions_for_all_fields,
    build_resolution_catalogs,
    degrade_all_fields,
    identify_degraded_peaks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged resolution degradation pipeline.")
    parser.add_argument(
        "--stage",
        choices=("degrade", "peaks", "regions", "catalogs", "all"),
        default="all",
    )
    parser.add_argument("--fields", nargs="+", default=list(FIELDS))
    parser.add_argument("--distances", nargs="+", type=float, default=list(DEFAULT_DISTANCES_MPC))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields = tuple(args.fields)
    distances = tuple(args.distances)

    if args.stage in {"degrade", "all"}:
        degrade_all_fields(fields=fields, distances_mpc=distances)
    if args.stage in {"peaks", "all"}:
        identify_degraded_peaks(fields=fields, distances_mpc=distances)
    if args.stage in {"regions", "all"}:
        build_regions_for_all_fields(fields=fields, distances_mpc=distances)
    if args.stage in {"catalogs", "all"}:
        build_resolution_catalogs(fields=fields, distances_mpc=distances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
