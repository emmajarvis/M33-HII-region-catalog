from pathlib import Path

import pandas as pd
from astropy.io import fits


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_fits_data(path: Path):
    with fits.open(path) as hdul:
        return hdul[0].data


def read_fits_data_header(path: Path):
    with fits.open(path) as hdul:
        return hdul[0].data, hdul[0].header


def write_fits(path: Path, data, header=None, overwrite: bool = True) -> None:
    safe_mkdir(path.parent)
    fits.writeto(path, data, header=header, overwrite=overwrite)


def read_catalog(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_catalog(df: pd.DataFrame, path: Path) -> None:
    safe_mkdir(path.parent)
    df.to_csv(path, index=False)
