from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


def deduplicate_pn_catalog(pn_catalog: pd.DataFrame, max_sep_arcsec: float = 1.5) -> pd.DataFrame:
    """Merge PN entries from multiple source catalogs by sky position."""
    required = {"pn_name", "source_catalog", "RA_deg", "Dec_deg"}
    missing = sorted(required.difference(pn_catalog.columns))
    if missing:
        raise KeyError(f"PN catalog is missing columns: {missing}")

    rows: list[dict] = []
    for _, row in pn_catalog.sort_values(["source_catalog", "pn_name"]).iterrows():
        coord = SkyCoord(float(row["RA_deg"]) * u.deg, float(row["Dec_deg"]) * u.deg)
        if rows:
            existing = SkyCoord(
                [item["RA_deg"] for item in rows] * u.deg,
                [item["Dec_deg"] for item in rows] * u.deg,
            )
            idx, sep, _ = coord.match_to_catalog_sky(existing)
            if sep.arcsec <= max_sep_arcsec:
                item = rows[int(idx)]
                item["source_catalogs"] = ";".join(
                    sorted(set(item["source_catalogs"].split(";")) | {str(row["source_catalog"])})
                )
                item["source_names"] = ";".join(
                    sorted(set(item["source_names"].split(";")) | {str(row["pn_name"])})
                )
                continue

        rows.append(
            {
                "pn_id": len(rows) + 1,
                "pn_name": str(row["pn_name"]),
                "RA_deg": float(row["RA_deg"]),
                "Dec_deg": float(row["Dec_deg"]),
                "source_catalogs": str(row["source_catalog"]),
                "source_names": str(row["pn_name"]),
            }
        )
    return pd.DataFrame(rows)


def crossmatch_pne_to_hii_boundaries(
    hii_catalog: pd.DataFrame,
    pn_catalog: pd.DataFrame,
    boundary_dir: str | Path = "Boundary_maps/Boundary_map_100pc",
    map_root: str | Path = "../M33-Maps-Calibrated",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign PNe to HII regions by testing their coordinates against boundary maps."""
    required_hii = {"field", "region_id", "zoi_center_label"}
    required_pn = {"pn_id", "pn_name", "RA_deg", "Dec_deg"}
    missing_hii = sorted(required_hii.difference(hii_catalog.columns))
    missing_pn = sorted(required_pn.difference(pn_catalog.columns))
    if missing_hii:
        raise KeyError(f"HII catalog is missing columns: {missing_hii}")
    if missing_pn:
        raise KeyError(f"PN catalog is missing columns: {missing_pn}")

    boundary_dir = Path(boundary_dir)
    map_root = Path(map_root)
    pn_coords = SkyCoord(
        pn_catalog["RA_deg"].to_numpy(dtype=float) * u.deg,
        pn_catalog["Dec_deg"].to_numpy(dtype=float) * u.deg,
    )
    match_rows: list[dict] = []

    for field in sorted(hii_catalog["field"].dropna().astype(str).unique()):
        boundary_path = boundary_dir / f"Boundary_map_{field}.fits"
        wcs_candidates = [
            map_root / f"M33-{field}" / f"M33{field}-Haflux.fits",
            Path("../M33-Maps") / f"M33-{field}" / f"M33{field}-Haflux.fits",
        ]
        wcs_path = next((path for path in wcs_candidates if path.exists()), None)
        if not boundary_path.exists() or wcs_path is None:
            continue

        boundary = np.asarray(fits.getdata(boundary_path))
        wcs = WCS(fits.getheader(wcs_path))
        x, y = wcs.world_to_pixel(pn_coords)
        xi = np.rint(x).astype(int)
        yi = np.rint(y).astype(int)
        inside = (
            np.isfinite(x)
            & np.isfinite(y)
            & (xi >= 0)
            & (yi >= 0)
            & (xi < boundary.shape[1])
            & (yi < boundary.shape[0])
        )

        field_hii = hii_catalog.loc[hii_catalog["field"].astype(str) == field].copy()
        label_to_rows = {}
        for idx, label in pd.to_numeric(field_hii["zoi_center_label"], errors="coerce").items():
            if np.isfinite(label) and int(label) > 0:
                label_to_rows.setdefault(int(label), []).append(idx)

        for pn_pos in np.flatnonzero(inside):
            label = int(np.nan_to_num(boundary[yi[pn_pos], xi[pn_pos]], nan=0))
            for hii_idx in label_to_rows.get(label, []):
                pn = pn_catalog.iloc[pn_pos]
                hii = hii_catalog.loc[hii_idx]
                match_rows.append(
                    {
                        "pn_id": int(pn["pn_id"]),
                        "pn_name": pn["pn_name"],
                        "pn_RA_deg": float(pn["RA_deg"]),
                        "pn_Dec_deg": float(pn["Dec_deg"]),
                        "field": field,
                        "region_id": hii["region_id"],
                        "zoi_center_label": label,
                        "pn_x": float(x[pn_pos]),
                        "pn_y": float(y[pn_pos]),
                    }
                )

    matches = pd.DataFrame(match_rows)
    augmented = hii_catalog.copy()
    augmented["has_pn_in_boundary"] = False
    augmented["n_pn_in_boundary"] = 0
    augmented["pn_names_in_boundary"] = ""
    if not matches.empty:
        grouped = matches.groupby(["field", "region_id"], dropna=False)
        counts = grouped.size()
        names = grouped["pn_name"].apply(lambda values: ";".join(sorted(set(map(str, values)))))
        keys = pd.MultiIndex.from_frame(augmented[["field", "region_id"]])
        augmented["n_pn_in_boundary"] = counts.reindex(keys, fill_value=0).to_numpy(dtype=int)
        augmented["has_pn_in_boundary"] = augmented["n_pn_in_boundary"] > 0
        augmented["pn_names_in_boundary"] = names.reindex(keys, fill_value="").to_numpy()
    return matches, augmented
