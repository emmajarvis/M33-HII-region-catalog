from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_laplace, minimum_filter, uniform_filter


@dataclass(frozen=True)
class PeakDetectionParams:
    lap_sigma: float = 1.5
    lap_size: int = 10
    threshold_size: int = 1
    snr_limit: float = 6.0
    bgbox: int = 25
    stdbox: int = 3
    signoi: float = 3.0
    xi: int = 50
    xf: int = 2000
    yi: int = 50
    yf: int = 2000

    @property
    def lap_tag(self) -> int:
        return int(round(self.lap_sigma * 10))


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_param_section(field: str, sizebox: int, default_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    path = repo_root() / "peak_files" / "param_sections" / f"param_section_{sizebox}_{field}.txt"
    if path.exists():
        xi, xf, yi, yf = np.loadtxt(path, unpack=True)
        return int(xi), int(xf), int(yi), int(yf)

    ny, nx = default_shape
    return 50, min(nx - 50, 2000), 50, min(ny - 50, 2000)


def adopted_params_for_field(field: str, amp_shape: tuple[int, int]) -> PeakDetectionParams:
    xi, xf, yi, yf = read_param_section(field, 1, amp_shape)
    signoi = 4.5 if field == "F5" else 3.0
    return PeakDetectionParams(signoi=signoi, xi=xi, xf=xf, yi=yi, yf=yf)


def threshold_map_path(field: str, params: PeakDetectionParams) -> Path:
    signoi = f"{params.signoi:.1f}"
    return (
        repo_root()
        / "peak_files"
        / f"2-BG_Noi_Th_data_OIII+Ha_{field}"
        / f"Th_box{params.bgbox}_std{params.stdbox}_signoi{signoi}.fits"
    )


def amplitude_map_path(field: str) -> Path:
    return (
        repo_root()
        / "peak_files"
        / f"data_for_visualisation_OIII+Ha_{field}"
        / f"M33_{field}_HaOIII_amp_nonan.fits"
    )


def amplitude_error_path(field: str) -> Path:
    return (
        repo_root().parent
        / "M33-Maps"
        / f"M33-{field}"
        / f"M33-{field}_SN3.LineMaps.map.Ha+OIII.1x1.amplitude-err.fits"
    )


def _finite_window(output: np.ndarray, data: np.ndarray, params: PeakDetectionParams, trim: int = 1) -> np.ndarray:
    output[:] = np.nan
    output[params.yi + trim : params.yf - trim, params.xi + trim : params.xf - trim] = data[
        params.yi + trim : params.yf - trim, params.xi + trim : params.xf - trim
    ]
    return output


def local_minima_laplacian_peaks(laplacian: np.ndarray, params: PeakDetectionParams) -> np.ndarray:
    work = np.asarray(laplacian, dtype=float)
    local_min = np.copy(work)
    size = params.lap_size * 2 + 1
    roi = np.nan_to_num(work[params.yi : params.yf, params.xi : params.xf], nan=np.inf)
    local_min[params.yi : params.yf, params.xi : params.xf] = minimum_filter(roi, size=size)

    peak_map = np.zeros(work.shape, dtype=np.uint8)
    y_idx, x_idx = np.where((work - local_min) == 0)
    peak_map[y_idx, x_idx] = 1
    peak_map[work >= 0] = 0

    y_idx, x_idx = np.where(peak_map == 1)
    ny, nx = peak_map.shape
    for y, x in zip(y_idx, x_idx):
        if y <= 0 or y >= ny - 1 or x <= 0 or x >= nx - 1:
            peak_map[y, x] = 0
            continue
        if np.count_nonzero(work[y - 1 : y + 2, x - 1 : x + 2] >= 0) == 8:
            peak_map[y, x] = 0

    out = np.zeros(work.shape, dtype=np.uint8)
    out[params.yi + 1 : params.yf - 1, params.xi + 1 : params.xf - 1] = peak_map[
        params.yi + 1 : params.yf - 1, params.xi + 1 : params.xf - 1
    ]
    return out


def run_peak_detection(
    amp_map: np.ndarray,
    err_map: np.ndarray,
    threshold_map: np.ndarray,
    *,
    field: str,
    params: PeakDetectionParams,
) -> pd.DataFrame:
    laplacian = np.copy(amp_map)
    laplacian[params.yi : params.yf, params.xi : params.xf] = gaussian_laplace(
        amp_map[params.yi : params.yf, params.xi : params.xf],
        params.lap_sigma,
    )
    peak_map = local_minima_laplacian_peaks(laplacian, params)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err_map > 0, amp_map / err_map, np.nan)
    peak_map = peak_map.astype(float)
    peak_map[snr < params.snr_limit] = np.nan

    if field in {"F5", "F6"}:
        threshold_signal = amp_map
    else:
        threshold_signal = np.copy(amp_map)
        threshold_signal[params.yi : params.yf, params.xi : params.xf] = uniform_filter(
            amp_map[params.yi : params.yf, params.xi : params.xf],
            size=params.threshold_size * 2 + 1,
        )
        threshold_signal = _finite_window(np.empty_like(amp_map, dtype=float), threshold_signal, params)

    peak_map[threshold_signal - threshold_map <= 0] = np.nan
    y_idx, x_idx = np.where(peak_map == 1)

    return pd.DataFrame(
        {
            "field": field,
            "x": x_idx.astype(int),
            "y": y_idx.astype(int),
            "peak_amp": amp_map[y_idx, x_idx],
            "peak_snr": snr[y_idx, x_idx],
            "threshold_excess": (threshold_signal - threshold_map)[y_idx, x_idx],
        }
    )


def load_detection_inputs(field: str, params: PeakDetectionParams | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, PeakDetectionParams, fits.Header]:
    amp_path = amplitude_map_path(field)
    amp, header = fits.getdata(amp_path, header=True)
    params = params or adopted_params_for_field(field, amp.shape)
    err = fits.getdata(amplitude_error_path(field))
    threshold = fits.getdata(threshold_map_path(field, params))
    return amp.astype(float), err.astype(float), threshold.astype(float), params, header

