from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def field_map_dir(field: str) -> Path:
    return repo_root().parent / "M33-maps" / f"M33-{field}"


def calibrated_field_map_dir(field: str) -> Path:
    return repo_root().parent / "M33-Maps-Calibrated" / f"M33-{field}"


def final_peaks_csv(field: str) -> Path:
    return repo_root() / "CATALOGS" / f"final_peaks_{field}.csv"


def zoi_fits(field: str, max_zoi_pc: int) -> Path:
    return repo_root() / "ZOI_maps" / f"ZOI_map_{max_zoi_pc}pc" / f"ZoI_map_{field}.fits"


def boundary_fits(field: str, max_zoi_pc: int) -> Path:
    return repo_root() / "Boundary_maps" / f"Boundary_map_{max_zoi_pc}pc" / f"Boundary_map_{field}.fits"


def boundary_metrics_csv(field: str, max_zoi_pc: int) -> Path:
    return repo_root() / "Boundary_maps" / f"Boundary_map_{max_zoi_pc}pc" / f"Boundary_metrics_{field}.csv"


def fit_int_spec_ascii(field: str) -> Path:
    return repo_root() / "CATALOGS" / "fit_int_spec" / f"{field}_Data_flux.ascii"


def flux_catalog_dir() -> Path:
    return repo_root() / "CATALOGS" / "flux_catalogs"


def flux_catalog_csv(field: str) -> Path:
    return flux_catalog_dir() / f"flux_catalog_{field}.csv"


def total_flux_catalog_csv() -> Path:
    return flux_catalog_dir() / "total_flux_catalog.csv"


def derived_catalog_csv(stage: str) -> Path:
    return flux_catalog_dir() / f"total_flux_catalog_{stage}.csv"
