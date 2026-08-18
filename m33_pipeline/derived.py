from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import astropy.constants as c
import astropy.units as u
import numpy as np
import pandas as pd
from scipy import odr
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from . import paths
from .config import DerivedConfig
from .io import read_catalog, write_catalog


C_CM_S = c.c.to_value(u.cm / u.s)
KK04_LOGQ_RANGE = (6.5, 8.5)
KK04_METALLICITY_RANGE = (7.1, 9.4)


def merge_field_flux_catalogs(catalog_dir: Path | None = None, method: str = "summed_map", dig_mode: str = "no_dig") -> pd.DataFrame:
    catalog_dir = catalog_dir or paths.flux_method_dir(method, dig_mode=dig_mode)
    files = sorted(glob.glob(os.path.join(catalog_dir, "flux_catalog_*.csv")))
    dfs = []
    for fp in files:
        df = read_catalog(Path(fp))
        if "field" not in df.columns:
            base = os.path.basename(fp)
            field = base.replace("flux_catalog_", "").replace(".csv", "")
            df["field"] = field
        dfs.append(df)
    if not dfs:
        raise ValueError(f"No flux catalogs found in {catalog_dir}")
    return pd.concat(dfs, axis=0, ignore_index=True, sort=False)


def write_total_flux_catalog(df: pd.DataFrame, method: str = "summed_map", dig_mode: str = "no_dig") -> Path:
    out_path = paths.total_flux_catalog_csv(method=method, dig_mode=dig_mode)
    write_catalog(df, out_path)
    return out_path


def write_combined_catalog(df: pd.DataFrame, method: str = "summed_map", dig_mode: str = "no_dig") -> Path:
    out_path = paths.combined_catalog_csv(method=method, dig_mode=dig_mode)
    write_catalog(df, out_path)
    return out_path


def write_derived_stage_catalog(df: pd.DataFrame, stage: str, method: str = "summed_map", dig_mode: str = "no_dig") -> Path:
    out_path = paths.derived_catalog_csv(stage, method=method, dig_mode=dig_mode)
    write_catalog(df, out_path)
    return out_path


def _col(prefix: str, name: str, suffix: str):
    return f"{prefix}_{name}_{suffix}", f"{prefix}_{name}_e_{suffix}"


def safe_normal(rng, mu, sigma, n):
    x = rng.normal(mu, sigma, n)
    return np.where(np.isfinite(x), x, np.nan)


def kk04_logq_from_logO32_and_Z(logO32, Z_12logOH):
    y = logO32
    z = Z_12logOH
    num = 32.81 - 1.153 * (y**2) + z * (-3.396 - 0.025 * y + 0.1444 * (y**2))
    den = 4.603 - 0.3119 * y - 0.163 * (y**2) + z * (-0.48 + 0.0271 * y + 0.02037 * (y**2))
    return num / den


def kk04_metallicity_from_logR23_and_logq(logR23, logq, branch):
    """Return the KK04 R23 metallicity for the selected upper/lower branch."""
    x = np.asarray(logR23, dtype=float)
    q = np.asarray(logq, dtype=float)
    branch = np.asarray(branch)
    lower = 9.40 + 4.65 * x - 3.17 * x**2 - q * (0.272 + 0.547 * x - 0.513 * x**2)
    upper = (
        9.72
        - 0.777 * x
        - 0.951 * x**2
        - 0.072 * x**3
        - 0.811 * x**4
        - q * (0.0737 - 0.0713 * x - 0.141 * x**2 + 0.0373 * x**3 - 0.058 * x**4)
    )
    return np.where(branch == "lower", lower, upper)


def kk04_iterative_logq_and_metallicity(
    logO32,
    logR23,
    branch,
    convergence_tolerance: float = 1.0e-3,
    max_iterations: int = 20,
):
    """Jointly solve KK04 ionization parameter and R23 metallicity."""
    logO32, logR23, branch = np.broadcast_arrays(
        np.asarray(logO32, dtype=float),
        np.asarray(logR23, dtype=float),
        np.asarray(branch),
    )
    valid = np.isfinite(logO32) & np.isfinite(logR23) & np.isin(branch, ("lower", "upper"))
    metallicity = np.where(branch == "lower", 8.2, 8.7).astype(float)
    metallicity[~valid] = np.nan
    iterations = np.zeros(metallicity.shape, dtype=int)
    converged = np.zeros(metallicity.shape, dtype=bool)

    for iteration in range(1, max_iterations + 1):
        logq = kk04_logq_from_logO32_and_Z(logO32, metallicity)
        updated = kk04_metallicity_from_logR23_and_logq(logR23, logq, branch)
        finite = valid & np.isfinite(logq) & np.isfinite(updated)
        newly_converged = finite & ~converged & (np.abs(updated - metallicity) <= convergence_tolerance)
        iterations[newly_converged] = iteration
        converged |= newly_converged
        metallicity = np.where(finite, updated, np.nan)
        if np.all(converged | ~valid):
            break

    logq = kk04_logq_from_logO32_and_Z(logO32, metallicity)
    within_calibration_grid = (
        np.isfinite(logq)
        & np.isfinite(metallicity)
        & (logq >= KK04_LOGQ_RANGE[0])
        & (logq <= KK04_LOGQ_RANGE[1])
        & (metallicity >= KK04_METALLICITY_RANGE[0])
        & (metallicity <= KK04_METALLICITY_RANGE[1])
    )
    converged &= within_calibration_grid
    iterations[valid & ~converged] = max_iterations
    return logq, metallicity, converged, iterations


def add_logU_KK04(
    df: pd.DataFrame,
    n_mc: int = 2000,
    seed: int = 123,
    branch_logN2O2_threshold: float = -1.2,
    convergence_tolerance: float = 1.0e-3,
    max_iterations: int = 20,
) -> pd.DataFrame:
    """Add self-consistent KK04 R23/O32 metallicity and ionization parameter."""
    out = df.copy()
    prefix = "F"
    oii, oii_e = _col(prefix, "[OII]3727", "sum_dered")
    o3_5007, o3_5007_e = _col(prefix, "[OIII]5007", "sum_dered")
    hb, hb_e = _col(prefix, "Hbeta", "sum_dered")
    ha, ha_e = _col(prefix, "Halpha", "sum_dered")
    nii, nii_e = _col(prefix, "[NII]6583", "sum_dered")

    if n_mc <= 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            total_oiii = (1.0 + 1.0 / 2.98) * out[o3_5007].to_numpy(dtype=float)
            O32 = total_oiii / out[oii].to_numpy(dtype=float)
            R23 = (out[oii].to_numpy(dtype=float) + total_oiii) / out[hb].to_numpy(dtype=float)
            logO32 = np.log10(O32)
            logR23 = np.log10(R23)
            logN2O2 = np.log10(out[nii].to_numpy(dtype=float) / out[oii].to_numpy(dtype=float))
            O3N2 = np.log10(
                (out[o3_5007].to_numpy(dtype=float) / out[hb].to_numpy(dtype=float))
                * (out[ha].to_numpy(dtype=float) / out[nii].to_numpy(dtype=float))
            )
        branch = np.where(logN2O2 < branch_logN2O2_threshold, "lower", "upper")
        logq, Z, converged, iterations = kk04_iterative_logq_and_metallicity(
            logO32,
            logR23,
            branch,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
        )
        logU = logq - np.log10(C_CM_S)
        good = (
            np.isfinite(O32)
            & np.isfinite(R23)
            & np.isfinite(logO32)
            & np.isfinite(logR23)
            & np.isfinite(logN2O2)
            & np.isfinite(Z)
            & np.isfinite(logq)
            & np.isfinite(logU)
            & converged
            & (out[oii].to_numpy(dtype=float) > 0)
            & (out[o3_5007].to_numpy(dtype=float) > 0)
            & (out[hb].to_numpy(dtype=float) > 0)
            & (out[nii].to_numpy(dtype=float) > 0)
        )
        out["O32"] = np.where(good, O32, np.nan)
        out["O32_e"] = np.nan
        out["logO32"] = np.where(good, logO32, np.nan)
        out["logO32_e"] = np.nan
        out["R23"] = np.where(good, R23, np.nan)
        out["logR23"] = np.where(good, logR23, np.nan)
        out["logN2O2"] = np.where(good, logN2O2, np.nan)
        out["O3N2"] = np.where(good & np.isfinite(O3N2), O3N2, np.nan)
        out["O3N2_e"] = np.nan
        out["Z_12logOH"] = np.where(good, Z, np.nan)
        out["Z_12logOH_e"] = np.nan
        out["logq_KK04"] = np.where(good, logq, np.nan)
        out["logq_KK04_e"] = np.nan
        out["logU_KK04"] = np.where(good, logU, np.nan)
        out["logU_KK04_e"] = np.nan
        out["logU_flag"] = np.where(np.isfinite(out["logU_KK04"]), "ok", "invalid")
        out["logU_KK04_branch"] = np.where(good, branch, "invalid")
        out["logU_KK04_converged"] = converged
        out["logU_KK04_iterations"] = iterations
        out["logU_meta_cal"] = "KK04_iterative_R23_O32"
        out["logU_meta_branch_threshold_logN2O2"] = branch_logN2O2_threshold
        return out

    rng = np.random.default_rng(seed)
    O32_med = np.full(len(out), np.nan)
    O32_sig = np.full(len(out), np.nan)
    logO32_med = np.full(len(out), np.nan)
    logO32_sig = np.full(len(out), np.nan)
    R23_med = np.full(len(out), np.nan)
    R23_sig = np.full(len(out), np.nan)
    logR23_med = np.full(len(out), np.nan)
    logR23_sig = np.full(len(out), np.nan)
    logN2O2_med = np.full(len(out), np.nan)
    logN2O2_sig = np.full(len(out), np.nan)
    O3N2_med = np.full(len(out), np.nan)
    O3N2_sig = np.full(len(out), np.nan)
    Z_med = np.full(len(out), np.nan)
    Z_sig = np.full(len(out), np.nan)
    logq_med = np.full(len(out), np.nan)
    logq_sig = np.full(len(out), np.nan)
    logU_med = np.full(len(out), np.nan)
    logU_sig = np.full(len(out), np.nan)
    branch_result = np.full(len(out), "invalid", dtype=object)
    converged_fraction = np.full(len(out), np.nan)
    iteration_med = np.full(len(out), np.nan)

    def med_sig(x):
        p16, p50, p84 = np.nanpercentile(x, [16, 50, 84])
        return p50, 0.5 * (p84 - p16)

    for i in range(len(out)):
        vals = out.loc[i, [oii, oii_e, o3_5007, o3_5007_e, hb, hb_e, ha, ha_e, nii, nii_e]].astype(float).to_numpy()
        if not np.all(np.isfinite(vals)):
            continue
        f_oii, e_oii, f_5007, e_5007, f_hb, e_hb, f_ha, e_ha, f_nii, e_nii = vals
        if min(f_oii, f_5007, f_hb, f_ha, f_nii) <= 0:
            continue
        if min(e_oii, e_5007, e_hb, e_ha, e_nii) < 0:
            continue

        d_oii = np.clip(safe_normal(rng, f_oii, e_oii, n_mc), 1e-30, None)
        d_5007 = np.clip(safe_normal(rng, f_5007, e_5007, n_mc), 1e-30, None)
        d_hb = np.clip(safe_normal(rng, f_hb, e_hb, n_mc), 1e-30, None)
        d_ha = np.clip(safe_normal(rng, f_ha, e_ha, n_mc), 1e-30, None)
        d_nii = np.clip(safe_normal(rng, f_nii, e_nii, n_mc), 1e-30, None)

        d_total_oiii = (1 + 1 / 2.98) * d_5007
        d_o32 = d_total_oiii / d_oii
        d_r23 = (d_oii + d_total_oiii) / d_hb
        d_logO32 = np.log10(d_o32)
        d_logR23 = np.log10(d_r23)
        d_logN2O2 = np.log10(d_nii / d_oii)
        d_o3n2 = np.log10((d_5007 / d_hb) * (d_ha / d_nii))
        d_branch = np.where(d_logN2O2 < branch_logN2O2_threshold, "lower", "upper")
        d_logq, d_Z, d_converged, d_iterations = kk04_iterative_logq_and_metallicity(
            d_logO32,
            d_logR23,
            d_branch,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
        )
        d_logU = d_logq - np.log10(C_CM_S)
        mask_logu = d_converged & np.isfinite(d_logq) & np.isfinite(d_logU) & np.isfinite(d_Z)
        if mask_logu.sum() < 50:
            continue
        d_logq = d_logq[mask_logu]
        d_logU = d_logU[mask_logu]
        d_Z = d_Z[mask_logu]

        O32_med[i], O32_sig[i] = med_sig(d_o32)
        logO32_med[i], logO32_sig[i] = med_sig(d_logO32)
        R23_med[i], R23_sig[i] = med_sig(d_r23)
        logR23_med[i], logR23_sig[i] = med_sig(d_logR23)
        logN2O2_med[i], logN2O2_sig[i] = med_sig(d_logN2O2)
        O3N2_med[i], O3N2_sig[i] = med_sig(d_o3n2)
        Z_med[i], Z_sig[i] = med_sig(d_Z)
        logq_med[i], logq_sig[i] = med_sig(d_logq)
        logU_med[i], logU_sig[i] = med_sig(d_logU)
        branch_result[i] = "lower" if np.mean(d_branch[mask_logu] == "lower") >= 0.5 else "upper"
        converged_fraction[i] = np.mean(d_converged)
        iteration_med[i] = np.median(d_iterations[mask_logu])

    out["O32"] = O32_med
    out["O32_e"] = O32_sig
    out["logO32"] = logO32_med
    out["logO32_e"] = logO32_sig
    out["R23"] = R23_med
    out["R23_e"] = R23_sig
    out["logR23"] = logR23_med
    out["logR23_e"] = logR23_sig
    out["logN2O2"] = logN2O2_med
    out["logN2O2_e"] = logN2O2_sig
    out["O3N2"] = O3N2_med
    out["O3N2_e"] = O3N2_sig
    out["Z_12logOH"] = Z_med
    out["Z_12logOH_e"] = Z_sig
    out["logq_KK04"] = logq_med
    out["logq_KK04_e"] = logq_sig
    out["logU_KK04"] = logU_med
    out["logU_KK04_e"] = logU_sig
    out["logU_flag"] = np.where(np.isfinite(out["logU_KK04"]), "ok", "invalid")
    out["logU_KK04_branch"] = branch_result
    out["logU_KK04_converged"] = np.isfinite(logU_med)
    out["logU_KK04_converged_fraction"] = converged_fraction
    out["logU_KK04_iterations"] = iteration_med
    out["logU_meta_cal"] = "KK04_iterative_R23_O32"
    out["logU_meta_branch_threshold_logN2O2"] = branch_logN2O2_threshold
    return out


def add_electron_density(
    df: pd.DataFrame,
    te_default: float = 1.0e4,
    use_dereddened: bool = True,
    n_mc: int = 50,
    min_mc_physical_fraction: float = 0.5,
    min_line_snr: float = 10.0,
    upper_limit_sigma: float = 1.0,
) -> pd.DataFrame:
    import pyneb as pn

    out = df.copy()
    c_6716_I, c_6731_I = "F_[SII]6716_sum_dered", "F_[SII]6731_sum_dered"
    c_6716_F, c_6731_F = "F_[SII]6716_sum", "F_[SII]6731_sum"
    c_6716_Ie, c_6731_Ie = "F_[SII]6716_e_sum_dered", "F_[SII]6731_e_sum_dered"
    c_6716_Fe, c_6731_Fe = "F_[SII]6716_e_sum", "F_[SII]6731_e_sum"

    if use_dereddened and c_6716_I in out.columns and c_6731_I in out.columns:
        c6716, c6731 = c_6716_I, c_6731_I
        c6716e = c_6716_Ie if c_6716_Ie in out.columns else None
        c6731e = c_6731_Ie if c_6731_Ie in out.columns else None
    else:
        c6716, c6731 = c_6716_F, c_6731_F
        c6716e = c_6716_Fe if c_6716_Fe in out.columns else None
        c6731e = c_6731_Fe if c_6731_Fe in out.columns else None

    f1 = pd.to_numeric(out[c6716], errors="coerce").to_numpy(dtype=float)
    f2 = pd.to_numeric(out[c6731], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = f1 / f2
    out["SII_ratio_6716_6731"] = ratio

    if c6716e is not None and c6731e is not None:
        # Dereddened line errors contain the same E(B-V) uncertainty in both
        # lines. Treating those terms as independent greatly overstates the
        # uncertainty of this ratio because the lines are only 15 Angstrom
        # apart. Use the raw flux fractional errors when they are available.
        error_f1_col, error_f2_col = c6716, c6731
        error_e1_col, error_e2_col = c6716e, c6731e
        if use_dereddened and all(col in out.columns for col in (c_6716_F, c_6731_F, c_6716_Fe, c_6731_Fe)):
            error_f1_col, error_f2_col = c_6716_F, c_6731_F
            error_e1_col, error_e2_col = c_6716_Fe, c_6731_Fe
        error_f1 = pd.to_numeric(out[error_f1_col], errors="coerce").to_numpy(dtype=float)
        error_f2 = pd.to_numeric(out[error_f2_col], errors="coerce").to_numpy(dtype=float)
        e1 = pd.to_numeric(out[error_e1_col], errors="coerce").to_numpy(dtype=float)
        e2 = pd.to_numeric(out[error_e2_col], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["SII_ratio_6716_6731_e"] = np.abs(ratio) * np.sqrt((e1 / error_f1) ** 2 + (e2 / error_f2) ** 2)

    atom = pn.Atom("S", 2)
    density_floor_cm3 = 1.0
    density_ceiling_cm3 = 1.0e6
    low_density_ratio_limit = float(
        atom.getEmissivity(te_default, density_floor_cm3, wave=6716)
        / atom.getEmissivity(te_default, density_floor_cm3, wave=6731)
    )
    high_density_ratio_limit = float(
        atom.getEmissivity(te_default, density_ceiling_cm3, wave=6716)
        / atom.getEmissivity(te_default, density_ceiling_cm3, wave=6731)
    )
    positive_flux = np.isfinite(f1) & np.isfinite(f2) & (f1 > 0) & (f2 > 0)
    physical_ratio = (ratio > high_density_ratio_limit) & (ratio < low_density_ratio_limit)
    valid = positive_flux & np.isfinite(ratio) & physical_ratio
    ne = np.full(len(out), np.nan, dtype=float)
    ne[valid] = atom.getTemDen(ratio[valid], tem=te_default, wave1=6716, wave2=6731)
    ne[~np.isfinite(ne) | (ne < density_floor_cm3) | (ne > density_ceiling_cm3)] = np.nan
    out["ne_SII_cm3"] = ne

    flags = np.full(len(out), "ok", dtype=object)
    flags[~positive_flux | ~np.isfinite(ratio)] = "invalid_flux_or_ratio"
    flags[positive_flux & np.isfinite(ratio) & (ratio >= low_density_ratio_limit)] = "low_density_limit"
    flags[positive_flux & np.isfinite(ratio) & (ratio <= high_density_ratio_limit)] = "high_density_limit"
    flags[valid & ~np.isfinite(ne)] = "solver_invalid"
    out["ne_SII_flag"] = flags
    out["ne_SII_ratio_low_density_limit"] = low_density_ratio_limit
    out["ne_SII_ratio_high_density_limit"] = high_density_ratio_limit

    ratio_error = np.asarray(
        pd.to_numeric(out.get("SII_ratio_6716_6731_e", np.nan), errors="coerce"),
        dtype=float,
    )
    ratio_well_constrained = (
        (flags == "ok")
        & np.isfinite(ratio_error)
        & ((ratio + ratio_error) < low_density_ratio_limit)
        & ((ratio - ratio_error) > high_density_ratio_limit)
    )
    out["ne_SII_ratio_well_constrained"] = ratio_well_constrained
    reliable = flags == "ok"
    snr_columns = ("SNR_[SII]6716_sum", "SNR_[SII]6731_sum")
    if all(col in out.columns for col in snr_columns):
        snr_6716 = pd.to_numeric(out[snr_columns[0]], errors="coerce").to_numpy(dtype=float)
        snr_6731 = pd.to_numeric(out[snr_columns[1]], errors="coerce").to_numpy(dtype=float)
        min_snr = np.minimum(snr_6716, snr_6731)
        out["ne_SII_min_line_snr"] = min_snr
        reliable &= np.isfinite(min_snr) & (min_snr >= min_line_snr)
    out["ne_SII_reliable"] = reliable

    # On the low-density side, a larger ratio means a lower density. For
    # measurements consistent with the low-density limit, convert the lower
    # one-sided ratio bound into an upper density limit.
    ratio_lower_bound = ratio - upper_limit_sigma * ratio_error
    low_density_side_unreliable = (
        positive_flux
        & np.isfinite(ratio)
        & np.isfinite(ratio_error)
        & (ratio_error > 0)
        & ((ratio + upper_limit_sigma * ratio_error) >= low_density_ratio_limit)
    )
    upper_limit_constrained = (
        low_density_side_unreliable
        & (ratio_lower_bound > high_density_ratio_limit)
        & (ratio_lower_bound < low_density_ratio_limit)
    )
    density_upper_limit = np.full(len(out), np.nan, dtype=float)
    density_upper_limit[upper_limit_constrained] = atom.getTemDen(
        ratio_lower_bound[upper_limit_constrained],
        tem=te_default,
        wave1=6716,
        wave2=6731,
    )
    density_upper_limit[
        ~np.isfinite(density_upper_limit)
        | (density_upper_limit < density_floor_cm3)
        | (density_upper_limit > density_ceiling_cm3)
    ] = np.nan
    out["ne_SII_cm3_upper_limit"] = density_upper_limit
    out["ne_SII_upper_limit_sigma"] = upper_limit_sigma
    out["ne_SII_is_upper_limit"] = np.isfinite(density_upper_limit)

    def monte_carlo_ne_from_ratio(r, rerr, central_is_valid, seed=0):
        if n_mc <= 0 or not central_is_valid:
            return np.nan, np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        draws = rng.normal(loc=r, scale=rerr, size=n_mc)
        physical_draws = draws[
            np.isfinite(draws)
            & (draws > high_density_ratio_limit)
            & (draws < low_density_ratio_limit)
        ]
        physical_fraction = physical_draws.size / n_mc
        if physical_draws.size == 0 or physical_fraction < min_mc_physical_fraction:
            return np.nan, np.nan, np.nan, physical_fraction
        ne_draws = atom.getTemDen(physical_draws, tem=te_default, wave1=6716, wave2=6731)
        ne_draws = np.atleast_1d(ne_draws)
        ne_draws = ne_draws[
            np.isfinite(ne_draws)
            & (ne_draws >= density_floor_cm3)
            & (ne_draws <= density_ceiling_cm3)
        ]
        if ne_draws.size == 0:
            return np.nan, np.nan, np.nan, physical_fraction
        p16, p50, p84 = np.percentile(ne_draws, [16, 50, 84])
        return p50, p50 - p16, p84 - p50, physical_fraction

    if "SII_ratio_6716_6731_e" in out.columns:
        mc = np.array(
            [
                monte_carlo_ne_from_ratio(r, e, flags[i] == "ok", seed=i)
                if np.isfinite(r) and np.isfinite(e) and e > 0
                else (np.nan, np.nan, np.nan, np.nan)
                for i, (r, e) in enumerate(zip(ratio, out["SII_ratio_6716_6731_e"]))
            ],
            dtype=float,
        )
        out["ne_SII_cm3_mc"] = mc[:, 0]
        out["ne_SII_cm3_mc_minus"] = mc[:, 1]
        out["ne_SII_cm3_mc_plus"] = mc[:, 2]
        out["ne_SII_mc_physical_fraction"] = mc[:, 3]
    return out

def add_thermal_pressure(
    df: pd.DataFrame,
    n_e: str | np.ndarray | pd.Series | float = "ne_SII_cm3",
    T_e: str | np.ndarray | pd.Series | float = 1.0e4,
    particle_factor: float = 2.0,
) -> pd.DataFrame:
    """Add ionized-gas pressure estimates based on the [S II] density.

    ``P_e / k_B = n_e T_e`` is the electron pressure. ``P_thermal / k_B =
    particle_factor n_e T_e`` approximates the total thermal pressure; the
    default particle factor of two assumes fully ionized hydrogen.
    """
    out = df.copy()

    def values(value, label: str) -> np.ndarray:
        if isinstance(value, str):
            if value not in out.columns:
                raise ValueError(f"Missing {label} column: {value}")
            return pd.to_numeric(out[value], errors="coerce").to_numpy(dtype=float)
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return np.full(len(out), float(array), dtype=float)
        if len(array) != len(out):
            raise ValueError(f"{label} must be scalar or have one value per catalog row.")
        return array

    ne = values(n_e, "electron density")
    te = values(T_e, "electron temperature")
    valid = np.isfinite(ne) & (ne > 0) & np.isfinite(te) & (te > 0)
    electron_pressure_over_k = np.where(valid, ne * te, np.nan)
    thermal_pressure_over_k = particle_factor * electron_pressure_over_k
    k_b_cgs = c.k_B.to_value(u.erg / u.K)

    out["P_e_SII_over_k_K_cm3"] = electron_pressure_over_k
    out["P_thermal_SII_over_k_K_cm3"] = thermal_pressure_over_k
    out["P_thermal_SII_dyn_cm2"] = thermal_pressure_over_k * k_b_cgs
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_P_thermal_SII_over_k"] = np.log10(thermal_pressure_over_k)
    out["P_thermal_SII_Te_K"] = te
    out["P_thermal_SII_particle_factor"] = particle_factor

    # Preserve the partially implemented legacy name as an electron-pressure
    # alias, while using explicit names for all new analysis.
    out["P_SII_K_cm3"] = electron_pressure_over_k

    density_scale_columns = {
        "ne_SII_cm3_mc": "P_thermal_SII_over_k_K_cm3_mc",
        "ne_SII_cm3_mc_minus": "P_thermal_SII_over_k_K_cm3_mc_minus",
        "ne_SII_cm3_mc_plus": "P_thermal_SII_over_k_K_cm3_mc_plus",
        "ne_SII_cm3_upper_limit": "P_thermal_SII_over_k_K_cm3_upper_limit",
    }
    scale = particle_factor * te
    for density_col, pressure_col in density_scale_columns.items():
        if density_col in out.columns:
            density_values = pd.to_numeric(out[density_col], errors="coerce").to_numpy(dtype=float)
            out[pressure_col] = density_values * scale

    pressure_upper_limit_col = "P_thermal_SII_over_k_K_cm3_upper_limit"
    if pressure_upper_limit_col in out.columns:
        pressure_upper_limit = pd.to_numeric(out[pressure_upper_limit_col], errors="coerce").to_numpy(dtype=float)
        out["P_thermal_SII_dyn_cm2_upper_limit"] = pressure_upper_limit * k_b_cgs
        with np.errstate(divide="ignore", invalid="ignore"):
            out["log_P_thermal_SII_over_k_upper_limit"] = np.log10(pressure_upper_limit)

    if "ne_SII_reliable" in out.columns:
        out["P_thermal_SII_reliable"] = out["ne_SII_reliable"].fillna(False).astype(bool)
    if "ne_SII_is_upper_limit" in out.columns:
        out["P_thermal_SII_is_upper_limit"] = out["ne_SII_is_upper_limit"].fillna(False).astype(bool)
    if "ne_SII_flag" in out.columns:
        out["P_thermal_SII_flag"] = out["ne_SII_flag"]
    return out


def add_peak_region_properties(
    df: pd.DataFrame,
    distance_mpc: float = 0.84,
    ebv_col: str = "sum_E_BV",
    ebv_error_col: str = "sum_E_BV_err",
    te_default: float = 1.0e4,
    n_mc: int = 50,
) -> pd.DataFrame:
    """Add peak-pixel H-alpha luminosity and [S II] electron density."""
    out = df.copy()
    required_peak_columns = [
        "F_Halpha_peak",
        "F_Halpha_e_peak",
        "F_[SII]6716_peak",
        "F_[SII]6716_e_peak",
        "F_[SII]6731_peak",
        "F_[SII]6731_e_peak",
    ]
    missing = [col for col in required_peak_columns if col not in out.columns]
    if missing:
        raise ValueError(f"Missing peak-pixel flux columns: {', '.join(missing)}")

    ha = pd.to_numeric(out["F_Halpha_peak"], errors="coerce").to_numpy(dtype=float)
    ha_error = pd.to_numeric(out["F_Halpha_e_peak"], errors="coerce").to_numpy(dtype=float)
    ebv = (
        pd.to_numeric(out[ebv_col], errors="coerce").to_numpy(dtype=float)
        if ebv_col in out.columns
        else np.zeros(len(out), dtype=float)
    )
    ebv_error = (
        pd.to_numeric(out[ebv_error_col], errors="coerce").to_numpy(dtype=float)
        if ebv_error_col in out.columns
        else np.zeros(len(out), dtype=float)
    )
    k_halpha = 2.535
    extinction_factor = 10.0 ** (0.4 * ebv * k_halpha)
    ha_dered = ha * extinction_factor
    derivative_ebv = ha_dered * (0.4 * np.log(10.0) * k_halpha)
    ha_dered_error = np.sqrt((extinction_factor * ha_error) ** 2 + (derivative_ebv * ebv_error) ** 2)
    distance_cm = (float(distance_mpc) * u.Mpc).to_value(u.cm)
    luminosity_factor = 4.0 * np.pi * distance_cm**2

    out["F_Halpha_peak_dered"] = ha_dered
    out["F_Halpha_e_peak_dered"] = ha_dered_error
    out["L_Ha_peak"] = luminosity_factor * ha
    out["L_Ha_peak_dered"] = luminosity_factor * ha_dered
    out["L_Ha_e_peak_dered"] = luminosity_factor * ha_dered_error
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_L_Ha_peak"] = np.log10(out["L_Ha_peak"])
        out["log_L_Ha_peak_dered"] = np.log10(out["L_Ha_peak_dered"])
    out["L_Ha_peak_distance_mpc"] = float(distance_mpc)

    peak_density_input = pd.DataFrame(
        {
            "F_[SII]6716_sum": pd.to_numeric(out["F_[SII]6716_peak"], errors="coerce"),
            "F_[SII]6716_e_sum": pd.to_numeric(out["F_[SII]6716_e_peak"], errors="coerce"),
            "F_[SII]6731_sum": pd.to_numeric(out["F_[SII]6731_peak"], errors="coerce"),
            "F_[SII]6731_e_sum": pd.to_numeric(out["F_[SII]6731_e_peak"], errors="coerce"),
        }
    )
    for line_name in ("[SII]6716", "[SII]6731"):
        peak_snr_col = f"SNR_{line_name}_peak"
        if peak_snr_col in out.columns:
            peak_density_input[f"SNR_{line_name}_sum"] = pd.to_numeric(out[peak_snr_col], errors="coerce")
    peak_density = add_electron_density(
        peak_density_input,
        te_default=te_default,
        use_dereddened=False,
        n_mc=n_mc,
    )
    peak_density_names = {
        "SII_ratio_6716_6731": "SII_ratio_6716_6731_peak",
        "SII_ratio_6716_6731_e": "SII_ratio_6716_6731_e_peak",
        "ne_SII_cm3": "ne_SII_peak_cm3",
        "ne_SII_flag": "ne_SII_peak_flag",
        "ne_SII_ratio_low_density_limit": "ne_SII_peak_ratio_low_density_limit",
        "ne_SII_ratio_high_density_limit": "ne_SII_peak_ratio_high_density_limit",
        "ne_SII_ratio_well_constrained": "ne_SII_peak_ratio_well_constrained",
        "ne_SII_min_line_snr": "ne_SII_peak_min_line_snr",
        "ne_SII_reliable": "ne_SII_peak_reliable",
        "ne_SII_cm3_upper_limit": "ne_SII_peak_cm3_upper_limit",
        "ne_SII_upper_limit_sigma": "ne_SII_peak_upper_limit_sigma",
        "ne_SII_is_upper_limit": "ne_SII_peak_is_upper_limit",
        "ne_SII_cm3_mc": "ne_SII_peak_cm3_mc",
        "ne_SII_cm3_mc_minus": "ne_SII_peak_cm3_mc_minus",
        "ne_SII_cm3_mc_plus": "ne_SII_peak_cm3_mc_plus",
        "ne_SII_mc_physical_fraction": "ne_SII_peak_mc_physical_fraction",
    }
    for source_col, output_col in peak_density_names.items():
        if source_col in peak_density.columns:
            out[output_col] = peak_density[source_col].to_numpy()
    return out


def classify_symmetry(row, threshold: float = 0.2) -> str:
    r16 = row["radius_p16_pc"]
    r50 = row["radius_p50_pc"]
    r84 = row["radius_p84_pc"]
    if not np.isfinite(r16) or not np.isfinite(r50) or not np.isfinite(r84):
        return "unknown"
    asymmetry = abs(r84 - r16) / (r84 + r16) if r84 != r16 else 0.0
    return "symmetric" if asymmetry < threshold else "asymmetric"


def add_symmetry_class(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    out = df.copy()
    out["symmetry_class"] = out.apply(classify_symmetry, axis=1, threshold=threshold)
    return out


def add_primary_overlap_flags(
    df: pd.DataFrame,
    ra_col: str = "RA_deg",
    dec_col: str = "Dec_deg",
    match_radius_arcsec: float = 1.0,
    match_radius_px: float | None = None,
    pixel_scale_arcsec: float = 0.32,
    field_col: str = "field",
    snr_cols: list[str] | None = None,
    boundary_overlap: bool = False,
    boundary_max_zoi_pc: int = 100,
    boundary_label_col: str = "zoi_center_label",
    boundary_min_overlap_pixels: int = 1,
    boundary_grid_arcsec: float | None = None,
    boundary_valid_bounds: tuple[int, int, int, int] | None = (50, 2000, 50, 2000),
) -> pd.DataFrame:
    """Flag duplicate regions in overlapping fields and keep the highest-S/N row.

    Duplicate groups are normally seeded by nearly coincident peak coordinates.
    When ``boundary_overlap`` is true, regions are also grouped if their boundary
    footprints share any sky-pixel samples across different fields.
    """
    out = df.copy()
    out["duplicate_group_id"] = pd.Series([pd.NA] * len(out), dtype="object")
    out["duplicate_group_size"] = 1
    out["is_duplicate_overlap"] = False
    out["duplicate_peak_overlap"] = False
    out["duplicate_boundary_overlap"] = False
    out["duplicated"] = False
    out["primary"] = True
    out["primary_rank"] = 1
    out["primary_region_id"] = out["region_id"] if "region_id" in out.columns else pd.Series([pd.NA] * len(out), dtype="object")
    out["duplicate_score_sum_snr"] = np.nan
    out["duplicate_score_min_snr"] = np.nan
    out["duplicate_score_median_snr"] = np.nan
    out["duplicate_score_finite_nlines"] = 0

    if ra_col not in out.columns or dec_col not in out.columns:
        raise ValueError(f"Missing coordinate columns for overlap matching: {ra_col}, {dec_col}")

    if snr_cols is None:
        snr_cols = [
            col for col in out.columns
            if col.startswith("SNR_") and col.endswith("_sum")
        ]
    if not snr_cols:
        raise ValueError("No SNR columns found for primary overlap selection.")

    pair_reasons: dict[tuple[int, int], set[str]] = {}

    ra = pd.to_numeric(out[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(out[dec_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    if valid.sum() > 0:
        # Small-angle approximation in arcsec for local duplicate grouping.
        # When a pixel radius is requested, convert it to an on-sky radius
        # instead of comparing local field x/y pixels, which are not on a common
        # mosaic grid.
        x_arcsec = ra[valid] * np.cos(np.deg2rad(dec[valid])) * 3600.0
        y_arcsec = dec[valid] * 3600.0
        coords = np.column_stack([x_arcsec, y_arcsec])
        match_radius = (
            float(match_radius_px) * float(pixel_scale_arcsec)
            if match_radius_px is not None
            else float(match_radius_arcsec)
        )
        tree = cKDTree(coords)
        valid_idx = np.flatnonzero(valid)
        peak_pairs = tree.query_pairs(r=match_radius)
        if field_col in out.columns:
            fields = out[field_col].astype(str).to_numpy()
            peak_pairs = {
                (i, j) for i, j in peak_pairs
                if fields[valid_idx[i]] != fields[valid_idx[j]]
            }
        for i, j in peak_pairs:
            pair = tuple(sorted((int(valid_idx[i]), int(valid_idx[j]))))
            pair_reasons.setdefault(pair, set()).add("peak")

    if boundary_overlap:
        boundary_pairs = _find_boundary_overlap_pairs(
            out,
            field_col=field_col,
            boundary_max_zoi_pc=boundary_max_zoi_pc,
            boundary_label_col=boundary_label_col,
            boundary_min_overlap_pixels=boundary_min_overlap_pixels,
            boundary_grid_arcsec=boundary_grid_arcsec or pixel_scale_arcsec,
            boundary_valid_bounds=boundary_valid_bounds,
        )
        for pair in boundary_pairs:
            pair_reasons.setdefault(tuple(sorted(pair)), set()).add("boundary")

    if not pair_reasons:
        return out

    parent = np.arange(len(out), dtype=int)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in pair_reasons:
        union(i, j)

    groups: dict[int, list[int]] = {}
    pair_rows = sorted({idx for pair in pair_reasons for idx in pair})
    for row_idx in pair_rows:
        groups.setdefault(find(row_idx), []).append(row_idx)

    dup_groups = [members for members in groups.values() if len(members) > 1]
    if not dup_groups:
        return out

    snr_frame = out[snr_cols].apply(pd.to_numeric, errors="coerce")
    halpha_snr_col = next((col for col in ["SNR_Halpha_sum", "SNR_Halpha_int"] if col in out.columns), None)
    halpha_snr = (
        pd.to_numeric(out[halpha_snr_col], errors="coerce").to_numpy(dtype=float)
        if halpha_snr_col is not None
        else np.full(len(out), np.nan, dtype=float)
    )
    score_sum = snr_frame.sum(axis=1, min_count=1).to_numpy(dtype=float)
    score_min = snr_frame.min(axis=1, skipna=True).to_numpy(dtype=float)
    score_median = snr_frame.median(axis=1, skipna=True).to_numpy(dtype=float)
    score_count = snr_frame.notna().sum(axis=1).to_numpy(dtype=int)
    out["duplicate_score_sum_snr"] = score_sum
    out["duplicate_score_min_snr"] = score_min
    out["duplicate_score_median_snr"] = score_median
    out["duplicate_score_finite_nlines"] = score_count

    for group_num, members in enumerate(dup_groups, start=1):
        rows = np.asarray(members, dtype=int)
        row_set = set(rows)
        group_id = f"dup_{group_num:04d}"
        out.loc[rows, "duplicate_group_id"] = group_id
        out.loc[rows, "duplicate_group_size"] = int(len(rows))
        out.loc[rows, "is_duplicate_overlap"] = True
        out.loc[rows, "duplicated"] = True
        out.loc[rows, "primary"] = False
        group_pair_reasons = [
            reasons
            for (i, j), reasons in pair_reasons.items()
            if i in row_set and j in row_set
        ]
        if any("peak" in reasons for reasons in group_pair_reasons):
            out.loc[rows, "duplicate_peak_overlap"] = True
        if any("boundary" in reasons for reasons in group_pair_reasons):
            out.loc[rows, "duplicate_boundary_overlap"] = True

        def score_tuple(row_idx: int):
            return (
                float(halpha_snr[row_idx]) if np.isfinite(halpha_snr[row_idx]) else -np.inf,
                int(score_count[row_idx]),
                float(score_min[row_idx]) if np.isfinite(score_min[row_idx]) else -np.inf,
                float(score_median[row_idx]) if np.isfinite(score_median[row_idx]) else -np.inf,
                float(score_sum[row_idx]) if np.isfinite(score_sum[row_idx]) else -np.inf,
            )

        primary_row = max(rows, key=score_tuple)
        ordered_rows = sorted(rows, key=score_tuple, reverse=True)
        out.loc[primary_row, "primary"] = True
        primary_region_id = out.loc[primary_row, "region_id"] if "region_id" in out.columns else primary_row
        out.loc[rows, "primary_region_id"] = primary_region_id
        for rank, row_idx in enumerate(ordered_rows, start=1):
            out.loc[row_idx, "primary_rank"] = rank

    return out


def _field_wcs_candidates_for_overlap(field: str) -> list[Path]:
    return [
        paths.calibrated_field_map_dir(field) / f"M33{field}-Haflux.fits",
        paths.field_map_dir(field) / f"M33{field}-Haflux.fits",
        paths.field_map_dir(field) / f"M33-{field}_SN3.LineMaps.map.Ha+OIII.1x1.amplitude.fits",
        paths.field_map_dir(field) / f"M33{field}-SN3Continuum.fits",
        paths.repo_root().parent / "M33-Maps" / f"M33-{field}" / f"M33{field}-Haflux.fits",
        paths.repo_root().parent / "M33-Maps-Calibrated" / f"M33-{field}" / f"M33{field}-Haflux.fits",
    ]


def _catalog_boundary_label(row: pd.Series, field: str, boundary_label_col: str) -> int | None:
    if boundary_label_col in row.index and pd.notna(row[boundary_label_col]):
        try:
            return int(round(float(row[boundary_label_col])))
        except (TypeError, ValueError):
            pass
    if "region_id" in row.index:
        normalized = _normalize_region_id(field, row["region_id"])
        if normalized is not None:
            try:
                return int(normalized.split("_")[-1])
            except (TypeError, ValueError):
                pass
    for fallback in ("region_number", "label", "id"):
        if fallback in row.index and pd.notna(row[fallback]):
            try:
                return int(round(float(row[fallback])))
            except (TypeError, ValueError):
                continue
    return None


def _find_boundary_overlap_pairs(
    df: pd.DataFrame,
    field_col: str = "field",
    boundary_max_zoi_pc: int = 100,
    boundary_label_col: str = "zoi_center_label",
    boundary_min_overlap_pixels: int = 1,
    boundary_grid_arcsec: float = 0.32,
    boundary_valid_bounds: tuple[int, int, int, int] | None = (50, 2000, 50, 2000),
) -> set[tuple[int, int]]:
    """Return row-index pairs whose region footprints overlap on the sky."""
    from astropy.io import fits
    from astropy.wcs import WCS

    if field_col not in df.columns:
        return set()

    sky_cell_owner: dict[tuple[int, int], tuple[int, str]] = {}
    overlap_counts: dict[tuple[int, int], int] = {}
    fields = df[field_col].dropna().astype(str).unique()
    grid = float(boundary_grid_arcsec)
    if not np.isfinite(grid) or grid <= 0:
        grid = 0.32

    for field in sorted(fields):
        field_idx = df.index[df[field_col].astype(str) == field]
        if len(field_idx) == 0:
            continue

        label_to_rows: dict[int, list[int]] = {}
        for row_idx in field_idx:
            label = _catalog_boundary_label(df.loc[row_idx], field, boundary_label_col)
            if label is not None:
                label_to_rows.setdefault(label, []).append(int(row_idx))
        if not label_to_rows:
            continue

        boundary_path = paths.boundary_fits(field, boundary_max_zoi_pc)
        wcs_path = next((p for p in _field_wcs_candidates_for_overlap(field) if p.exists()), None)
        if not boundary_path.exists() or wcs_path is None:
            continue

        labels = np.asarray(fits.getdata(boundary_path))
        finite_labels = np.isfinite(labels)
        label_int = np.zeros(labels.shape, dtype=np.int64)
        label_int[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64)
        valid = finite_labels & (label_int > 0)
        if boundary_valid_bounds is not None:
            x0, x1, y0, y1 = boundary_valid_bounds
            bounds_mask = np.zeros(label_int.shape, dtype=bool)
            bounds_mask[max(0, y0):min(label_int.shape[0], y1), max(0, x0):min(label_int.shape[1], x1)] = True
            valid &= bounds_mask
        present_labels = np.array([label for label in label_to_rows if label > 0], dtype=np.int64)
        if present_labels.size == 0:
            continue
        valid &= np.isin(label_int, present_labels)
        if not np.any(valid):
            continue

        wcs = WCS(fits.getheader(wcs_path))
        y_pix_all, x_pix_all = np.nonzero(valid)
        chunk_size = 250_000
        for start in range(0, len(x_pix_all), chunk_size):
            stop = start + chunk_size
            x_pix = x_pix_all[start:stop]
            y_pix = y_pix_all[start:stop]
            pix_labels = label_int[y_pix, x_pix]
            sky = wcs.pixel_to_world(x_pix, y_pix)
            ra_deg = np.asarray(sky.ra.deg, dtype=float)
            dec_deg = np.asarray(sky.dec.deg, dtype=float)
            ok = np.isfinite(ra_deg) & np.isfinite(dec_deg)
            if not np.any(ok):
                continue
            ra_deg = ra_deg[ok]
            dec_deg = dec_deg[ok]
            pix_labels = pix_labels[ok]
            x_key = np.rint(ra_deg * np.cos(np.deg2rad(dec_deg)) * 3600.0 / grid).astype(np.int64)
            y_key = np.rint(dec_deg * 3600.0 / grid).astype(np.int64)

            for label, x_cell, y_cell in zip(pix_labels, x_key, y_key):
                rows = label_to_rows.get(int(label))
                if not rows:
                    continue
                row_idx = rows[0]
                key = (int(x_cell), int(y_cell))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        prior = sky_cell_owner.get((key[0] + dx, key[1] + dy))
                        if prior is None:
                            continue
                        prior_idx, prior_field = prior
                        if prior_field == field or prior_idx == row_idx:
                            continue
                        pair = tuple(sorted((int(prior_idx), int(row_idx))))
                        overlap_counts[pair] = overlap_counts.get(pair, 0) + 1
                        break
                    else:
                        continue
                    break
                sky_cell_owner.setdefault(key, (row_idx, field))

    min_pixels = max(1, int(boundary_min_overlap_pixels))
    return {pair for pair, count in overlap_counts.items() if count >= min_pixels}


def _normalize_region_id(field: str, region_id) -> str | None:
    if pd.isna(region_id):
        return None
    rid = str(region_id).strip()
    if "_" in rid:
        return rid
    try:
        label = int(round(float(rid)))
    except ValueError:
        return None
    return f"{field}_{label:04d}"


def _parse_wr_name_to_coord(name: str):
    from astropy.coordinates import SkyCoord

    name = str(name).strip()
    if not name.startswith("J") or len(name) < 16:
        raise ValueError(f"Unrecognized WR source name format: {name}")
    ra_h = name[1:3]
    ra_m = name[3:5]
    ra_s = name[5:10]
    dec_sign = name[10]
    dec_d = name[11:13]
    dec_m = name[13:15]
    dec_s = name[15:]
    ra_str = f"{ra_h}h{ra_m}m{ra_s}s"
    dec_str = f"{dec_sign}{dec_d}d{dec_m}m{dec_s}s"
    return SkyCoord(ra_str, dec_str, frame="icrs")


def add_boundary_source_flags(
    df: pd.DataFrame,
    max_zoi_pc: int = 100,
    wr_catalog: pd.DataFrame | None = None,
    snr_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u

    from .reporting import load_snr_catalog, load_wr_catalog

    out = df.copy()
    out["has_wr_in_boundary"] = False
    out["n_wr_in_boundary"] = 0
    out["wr_names_in_boundary"] = ""
    out["has_snr_in_boundary"] = False
    out["n_snr_in_boundary"] = 0

    wr_catalog = load_wr_catalog() if wr_catalog is None else wr_catalog.copy()
    snr_catalog = load_snr_catalog() if snr_catalog is None else snr_catalog.copy()

    wr_coords = []
    wr_names = []
    if not wr_catalog.empty and "Name (star)" in wr_catalog.columns:
        for name in wr_catalog["Name (star)"]:
            try:
                wr_coords.append(_parse_wr_name_to_coord(name))
                wr_names.append(str(name).strip())
            except Exception:
                continue
    wr_sky = SkyCoord(wr_coords) if wr_coords else None

    snr_sky = None
    if not snr_catalog.empty and {"RA", "Dec"}.issubset(snr_catalog.columns):
        try:
            snr_sky = SkyCoord(
                snr_catalog["RA"].astype(str).to_numpy(),
                snr_catalog["Dec"].astype(str).to_numpy(),
                unit=(u.hourangle, u.deg),
            )
        except Exception:
            snr_sky = None

    for field in sorted(out["field"].dropna().astype(str).unique()):
        field_mask = out["field"].astype(str) == field
        field_idx = out.index[field_mask]
        if len(field_idx) == 0:
            continue

        boundary_path = paths.boundary_fits(field, max_zoi_pc)
        wcs_candidates = [
            paths.field_map_dir(field) / f"M33{field}-Haflux.fits",
            paths.field_map_dir(field) / f"M33-{field}_SN3.LineMaps.map.Ha+OIII.1x1.amplitude.fits",
            paths.field_map_dir(field) / f"M33{field}-SN3Continuum.fits",
        ]
        wcs_path = next((p for p in wcs_candidates if p.exists()), None)
        if not boundary_path.exists() or wcs_path is None:
            continue

        boundary = fits.getdata(boundary_path)
        boundary = np.asarray(boundary, dtype=float)
        wcs = WCS(fits.getheader(wcs_path))

        label_to_rows: dict[int, list[int]] = {}
        for row_idx in field_idx:
            row = out.loc[row_idx]
            region_id = row["region_id"] if "region_id" in out.columns else np.nan
            normalized = _normalize_region_id(field, region_id)
            if normalized is None and "zoi_center_label" in out.columns and np.isfinite(row.get("zoi_center_label", np.nan)):
                label = int(round(float(row["zoi_center_label"])))
            else:
                try:
                    label = int(normalized.split("_")[-1])
                except Exception:
                    continue
            label_to_rows.setdefault(label, []).append(row_idx)

        def _assign_hits(skycoords, names, count_col, flag_col, names_col=None):
            if skycoords is None or len(skycoords) == 0:
                return
            xs, ys = wcs.world_to_pixel(skycoords)
            for src_i, (x, y) in enumerate(zip(xs, ys)):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xi = int(round(float(x)))
                yi = int(round(float(y)))
                if yi < 0 or yi >= boundary.shape[0] or xi < 0 or xi >= boundary.shape[1]:
                    continue
                label_val = boundary[yi, xi]
                if not np.isfinite(label_val) or label_val <= 0:
                    continue
                label = int(round(float(label_val)))
                rows = label_to_rows.get(label, [])
                for row_idx in rows:
                    out.at[row_idx, count_col] = int(out.at[row_idx, count_col]) + 1
                    out.at[row_idx, flag_col] = True
                    if names_col is not None:
                        current = str(out.at[row_idx, names_col]).strip()
                        new_name = str(names[src_i]).strip()
                        out.at[row_idx, names_col] = new_name if current == "" else f"{current}; {new_name}"

        _assign_hits(wr_sky, wr_names, "n_wr_in_boundary", "has_wr_in_boundary", "wr_names_in_boundary")
        _assign_hits(snr_sky, [f"SNR_{i+1}" for i in range(len(snr_sky))] if snr_sky is not None else [], "n_snr_in_boundary", "has_snr_in_boundary")

    return out


def poly_eval(coeffs, z):
    return np.polyval(coeffs[::-1], z)


def invert_logR_to_Z(coeffs, logR_obs, z_min=-2.5, z_max=0.8, ngrid=2000, branch="upper"):
    if not np.isfinite(logR_obs):
        return np.nan
    zgrid = np.linspace(z_min, z_max, ngrid)
    fgrid = poly_eval(coeffs, zgrid) - logR_obs
    if np.all(~np.isfinite(fgrid)):
        return np.nan
    finite = np.isfinite(fgrid)
    zgrid = zgrid[finite]
    fgrid = fgrid[finite]
    if zgrid.size < 2:
        return np.nan
    signs = np.sign(fgrid)
    idx = np.where(signs[:-1] * signs[1:] < 0)[0]
    # If there is no bracketed root, the observed ratio is outside the usable
    # range of this calibration (or lands on an ambiguous branch). Returning the
    # nearest grid edge creates artificial "flat line" metallicities in plots, so
    # treat these cases as invalid.
    if idx.size == 0:
        best_i = int(np.argmin(np.abs(fgrid)))
        best_abs = float(np.abs(fgrid[best_i]))
        at_edge = best_i == 0 or best_i == (zgrid.size - 1)
        if (not at_edge) and best_abs < 1e-3:
            return zgrid[best_i]
        return np.nan
    roots = []
    for i in idx:
        a, b = zgrid[i], zgrid[i + 1]
        try:
            roots.append(brentq(lambda z: poly_eval(coeffs, z) - logR_obs, a, b, maxiter=200))
        except ValueError:
            continue
    if not roots:
        return np.nan
    if branch == "upper":
        return max(roots)
    if branch == "lower":
        return min(roots)
    return roots[int(np.argmin([abs(poly_eval(coeffs, root) - logR_obs) for root in roots]))]


def odr_refine_z(coeffs, logR_obs, z0, sx=1.0, sy=1.0, z_min=-2.5, z_max=0.8):
    if not (np.isfinite(z0) and np.isfinite(logR_obs)):
        return np.nan

    def f(beta, x):
        z = beta[0]
        return poly_eval(coeffs, z) + 0.0 * x

    model = odr.Model(f)
    data = odr.RealData(np.array([0.0]), np.array([logR_obs]), sx=np.array([sx]), sy=np.array([sy]))
    out = odr.ODR(data, model, beta0=[z0]).run()
    z_fit = out.beta[0]
    if not np.isfinite(z_fit):
        return np.nan
    if z_fit < z_min or z_fit > z_max:
        return np.nan
    return z_fit


@dataclass(frozen=True)
class Calibration:
    name: str
    coeffs: np.ndarray
    R_func: callable
    valid_oh_range: tuple[float, float]
    reference: str
    branch: str = "upper"


METALLICITY_VALID_RANGES = {
    "Z_N2S2Halpha_Brazzini2024": ((7.50, 8.80), "N2S2Halpha", "Brazzini et al. 2024"),
    "Z_N2_Brazzini2024": ((7.50, 8.80), "N2", "Brazzini et al. 2024"),
    "Z_O3N2_Brazzini2024": ((7.50, 8.80), "O3N2", "Brazzini et al. 2024"),
    "Z_R3_Brazzini2024": ((7.50, 8.80), "R3", "Brazzini et al. 2024"),
    "Z_R23_Maiolino2008": ((7.05, 9.20), "R23", "Maiolino et al. 2008"),
    "Z_N2_Maiolino2008": ((7.05, 9.20), "N2", "Maiolino et al. 2008"),
    "Z_R23_Curti2017": ((7.60, 8.85), "R23", "Curti et al. 2017"),
    "Z_R3_Curti2017": ((7.60, 8.85), "R3", "Curti et al. 2017"),
    "Z_N2_Curti2017": ((7.60, 8.85), "N2", "Curti et al. 2017"),
    "Z_O3N2_Curti2017": ((7.60, 8.85), "O3N2", "Curti et al. 2017"),
    "Z_R_Pilyugin2016_highN2": ((7.00, 8.80), "R high-N2", "Pilyugin & Grebel 2016"),
    "Z_R_Pilyugin2016_lowN2": ((7.00, 8.80), "R low-N2", "Pilyugin & Grebel 2016"),
    "Z_S_Pilyugin2016_highN2": ((7.00, 8.80), "S high-N2", "Pilyugin & Grebel 2016"),
    "Z_S_Pilyugin2016_lowN2": ((7.00, 8.80), "S low-N2", "Pilyugin & Grebel 2016"),
    "Z_R23_KK2004": ((8.40, 9.40), "R23 upper branch", "Kobulnicky & Kewley 2004"),
    "Z_NII_KD2002": ((8.40, 9.40), "N2O2", "Kewley & Dopita 2002"),
    "Z_D2016": ((7.40, 9.40), "N2S2Halpha", "Dopita et al. 2016"),
    "Z_O3N2_M2013": ((7.60, 8.80), "O3N2", "Marino et al. 2013"),
    "Z_N2_M2013": ((7.60, 8.80), "N2", "Marino et al. 2013"),
    "Z_C2001": ((7.10, 9.10), "CL01", "Charlot & Longhetti 2001"),
    "Z_N2_PP2004": ((8.12, 9.05), "N2", "Pettini & Pagel 2004"),
    "Z_O3N2_PP2004": ((8.12, 9.05), "O3N2", "Pettini & Pagel 2004"),
    "Z_N2_Brown2016": ((7.60, 9.30), "N2", "Brown et al. 2016"),
    "Z_O3N2_Brown2016": ((7.60, 9.30), "O3N2", "Brown et al. 2016"),
    "Z_N2O2_Brown2016": ((7.60, 9.30), "N2O2", "Brown et al. 2016"),
    "Z_N2O2_KD2002": ((8.40, 9.40), "N2O2", "Kewley & Dopita 2002"),
}


def metallicity_valid_range_summary() -> pd.DataFrame:
    rows = []
    for column, (valid_range, indicator, reference) in METALLICITY_VALID_RANGES.items():
        rows.append(
            {
                "column": column,
                "indicator": indicator,
                "reference": reference,
                "valid_12logOH_min": valid_range[0],
                "valid_12logOH_max": valid_range[1],
            }
        )
    return pd.DataFrame(rows)


def print_metallicity_valid_ranges() -> None:
    summary = metallicity_valid_range_summary()
    print("Adopted metallicity calibration validity ranges:")
    for row in summary.itertuples(index=False):
        print(
            f"{row.column}: {row.valid_12logOH_min:.2f} <= 12+log(O/H) <= "
            f"{row.valid_12logOH_max:.2f}; indicator={row.indicator}; reference={row.reference}"
        )


def _mask_to_valid_metallicity_range(values, column):
    if column not in METALLICITY_VALID_RANGES:
        return values
    lo, hi = METALLICITY_VALID_RANGES[column][0]
    arr = np.asarray(values, dtype=float).copy()
    arr[(arr < lo) | (arr > hi)] = np.nan
    return arr


def _apply_valid_metallicity_ranges(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    out = df.copy()
    use_columns = METALLICITY_VALID_RANGES if columns is None else columns
    for col in use_columns:
        if col in out.columns:
            out[col] = _mask_to_valid_metallicity_range(out[col], col)
    return out


def build_calibrations():
    return [
        Calibration("N2S2Halpha_Brazzini2024", np.array([0.24, 2.21, 0.76]), lambda N2, S2, R3, R2, R23: N2 / S2 * (N2**0.264), *METALLICITY_VALID_RANGES["Z_N2S2Halpha_Brazzini2024"][0::2], branch="upper"),
        Calibration("N2_Brazzini2024", np.array([-0.41, 0.57, -4.91, -5.81, -1.95]), lambda N2, S2, R3, R2, R23: N2, *METALLICITY_VALID_RANGES["Z_N2_Brazzini2024"][0::2], branch="upper"),
        Calibration("O3N2_Brazzini2024", np.array([-0.51, -7.74, -6.12, -1.60]), lambda N2, S2, R3, R2, R23: R3 / N2, *METALLICITY_VALID_RANGES["Z_O3N2_Brazzini2024"][0::2], branch="upper"),
        Calibration("R3_Brazzini2024", np.array([-0.84, -5.86, -6.27, -1.95]), lambda N2, S2, R3, R2, R23: R3, *METALLICITY_VALID_RANGES["Z_R3_Brazzini2024"][0::2], branch="upper"),
        Calibration("R23_Maiolino2008", np.array([0.7462, -0.7149, -0.9401, -0.6154, -0.2524]), lambda N2, S2, R3, R2, R23: R23, *METALLICITY_VALID_RANGES["Z_R23_Maiolino2008"][0::2], branch="upper"),
        Calibration("N2_Maiolino2008", np.array([-0.7732, 1.2357, -0.2811, -0.7201, -0.3330]), lambda N2, S2, R3, R2, R23: N2, *METALLICITY_VALID_RANGES["Z_N2_Maiolino2008"][0::2], branch="upper"),
        Calibration("R3_Curti2017", np.array([-0.277, -3.549, -3.593, -0.981]), lambda N2, S2, R3, R2, R23: R3, *METALLICITY_VALID_RANGES["Z_R3_Curti2017"][0::2], branch="upper"),
        Calibration("R23_Curti2017", np.array([0.527, -1.569, -1.652, -0.421]), lambda N2, S2, R3, R2, R23: R23, *METALLICITY_VALID_RANGES["Z_R23_Curti2017"][0::2], branch="upper"),
        Calibration("N2_Curti2017", np.array([-0.489, 1.513, -2.554, -5.293, -2.867]), lambda N2, S2, R3, R2, R23: N2, *METALLICITY_VALID_RANGES["Z_N2_Curti2017"][0::2], branch="upper"),
        Calibration("O3N2_Curti2017", np.array([-0.281, -4.765, -2.268]), lambda N2, S2, R3, R2, R23: R3 / N2, *METALLICITY_VALID_RANGES["Z_O3N2_Curti2017"][0::2], branch="upper"),
    ]


def compute_metallicities(full_catalog, z_min=None, z_max=None, use_odr=True, sx=1.0, sy=1.0):
    ha = np.asarray(full_catalog["F_Halpha_sum_dered"], float)
    hb = np.asarray(full_catalog["F_Hbeta_sum_dered"], float)
    nii = np.asarray(full_catalog["F_[NII]6583_sum_dered"], float)
    sii1 = np.asarray(full_catalog["F_[SII]6716_sum_dered"], float)
    sii2 = np.asarray(full_catalog["F_[SII]6731_sum_dered"], float)
    oiii = np.asarray(full_catalog["F_[OIII]5007_sum_dered"], float)
    oii = np.asarray(full_catalog["F_[OII]3727_sum_dered"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = nii / ha
        s2 = (sii1 + sii2) / ha
        r3 = oiii / hb
        r2 = oii / hb
        r23 = r2 + r3
    out = {}
    for cal in build_calibrations():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            R = cal.R_func(n2, s2, r3, r2, r23)
        logR = np.full_like(R, np.nan, dtype=float)
        goodR = np.isfinite(R) & (R > 0)
        logR[goodR] = np.log10(R[goodR])
        Z = np.full_like(logR, np.nan, dtype=float)
        for i, logR_i in enumerate(logR):
            if not np.isfinite(logR_i):
                continue
            cal_z_min = cal.valid_oh_range[0] - 8.69 if z_min is None else z_min
            cal_z_max = cal.valid_oh_range[1] - 8.69 if z_max is None else z_max
            z0 = invert_logR_to_Z(
                cal.coeffs,
                logR_i,
                z_min=cal_z_min,
                z_max=cal_z_max,
                branch=cal.branch,
            )
            Z[i] = (
                odr_refine_z(cal.coeffs, logR_i, z0, sx=sx, sy=sy, z_min=cal_z_min, z_max=cal_z_max)
                if use_odr and np.isfinite(z0)
                else z0
            )
        out[cal.name] = Z
    return out


def add_metallicity_columns(df: pd.DataFrame, use_odr: bool = True, print_valid_ranges: bool = False) -> pd.DataFrame:
    if print_valid_ranges:
        print_metallicity_valid_ranges()
    out = df.copy()
    z_dict = compute_metallicities(out, use_odr=use_odr)
    out["Z_N2_Brazzini2024"] = z_dict["N2_Brazzini2024"] + 8.69
    out["Z_O3N2_Brazzini2024"] = z_dict["O3N2_Brazzini2024"] + 8.69
    out["Z_N2S2Halpha_Brazzini2024"] = z_dict["N2S2Halpha_Brazzini2024"] + 8.69
    out["Z_R3_Brazzini2024"] = z_dict["R3_Brazzini2024"] + 8.69
    out["Z_R23_Maiolino2008"] = z_dict["R23_Maiolino2008"] + 8.69
    out["Z_N2_Maiolino2008"] = z_dict["N2_Maiolino2008"] + 8.69
    out["Z_R23_Curti2017"] = z_dict["R23_Curti2017"] + 8.69
    out["Z_R3_Curti2017"] = z_dict["R3_Curti2017"] + 8.69
    out["Z_N2_Curti2017"] = z_dict["N2_Curti2017"] + 8.69
    out["Z_O3N2_Curti2017"] = z_dict["O3N2_Curti2017"] + 8.69

    mask = np.ones(len(out), dtype=bool)
    ha = np.asarray(out[mask]["F_Halpha_sum_dered"], float)
    hb = np.asarray(out[mask]["F_Hbeta_sum_dered"], float)
    nii = np.asarray(out[mask]["F_[NII]6583_sum_dered"], float)
    sii1 = np.asarray(out[mask]["F_[SII]6716_sum_dered"], float)
    sii2 = np.asarray(out[mask]["F_[SII]6731_sum_dered"], float)
    sii = sii1 + sii2
    oiii = np.asarray(out[mask]["F_[OIII]5007_sum_dered"], float)
    oii = np.asarray(out[mask]["F_[OII]3727_sum_dered"], float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        n2 = nii / ha
        s2 = (sii1 + sii2) / ha
        r3 = (1 + 1 / 2.89) * oiii / hb
        r2 = oii / hb
        out["Z_R_Pilyugin2016_highN2"] = 8.589 + 0.022 * np.log10(r3 / r2) + 0.399 * np.log10(n2) + (-0.137 + 0.164 * np.log10(r3 / r2) + 0.589 * np.log10(n2)) * np.log10(r2)
        out["Z_R_Pilyugin2016_lowN2"] = 7.932 + 0.944 * np.log10(r3 / r2) + 0.695 * np.log10(n2) + (0.970 - 0.291 * np.log10(r3 / r2) - 0.019 * np.log10(n2)) * np.log10(r2)
        out["Z_S_Pilyugin2016_highN2"] = 8.424 + 0.030 * np.log10(r3 / s2) + 0.751 * np.log10(n2) + (-0.349 + 0.182 * np.log10(r3 / s2) + 0.508 * np.log10(n2)) * np.log10(s2)
        out["Z_S_Pilyugin2016_lowN2"] = 8.072 + 0.789 * np.log10(r3 / s2) + 0.726 * np.log10(n2) + (1.069 - 0.170 * np.log10(r3 / s2) + 0.022 * np.log10(n2)) * np.log10(s2)
        r23 = (oiii + oii) / hb
        o32 = oiii / oii
        x = np.log10(r23)
        y = np.log10(o32)
        out["Z_R23_KK2004"] = 9.11 - 0.218 * x - 0.0587 * x**2 - 0.330 * x**3 - 0.199 * x**4 - y * (0.00235 - 0.1105 * x - 0.051 * x**2 - 0.04085 * x**3 - 0.003585 * x**4)
        out["Z_NII_KD2002"] = np.log10(1.54020 + 1.26602 * nii / oii + 0.167977 * (nii / oii) ** 2) + 8.93
        y = np.log10(nii / sii) + 0.264 * np.log10(nii / ha)
        out["Z_D2016"] = 8.77 + y
        out["Z_O3N2_M2013"] = 8.533 - 0.214 * np.log10((oiii / hb) / n2)
        out["Z_N2_M2013"] = 8.743 + 0.462 * np.log10(n2)
        out["Z_C2001"] = np.log10(5.09e-4 * (oii / oiii) ** 0.17 * (nii / sii / 0.85) ** 1.17) + 12.0
        out["Z_N2_PP2004"] = 9.37 + 2.03 * np.log10(n2) + 1.26 * np.log10(n2) ** 2 + 0.32 * np.log10(n2) ** 3
        out["Z_O3N2_PP2004"] = 8.73 - 0.32 * np.log10((oiii / hb) / n2)
        out["Z_N2_Brown2016"] = 9.12 + 0.58 * np.log10(n2)
        out["Z_O3N2_Brown2016"] = 8.98 - 0.32 * np.log10((oiii / hb) / n2)
        out["Z_N2O2_Brown2016"] = 9.20 + 0.54 * np.log10(nii / oii)
        out["Z_N2O2_KD2002"] = np.log10(1.54020 + 1.26602 * nii / oii + 0.167977 * (nii / oii) ** 2) + 8.93
    return _apply_valid_metallicity_ranges(out)


def add_metallicity_error_columns(df: pd.DataFrame, n_mc: int = 50, seed: int = 123) -> pd.DataFrame:
    out = df.copy()
    required_cols = [
        "F_Halpha_sum_dered",
        "F_Halpha_e_sum_dered",
        "F_Hbeta_sum_dered",
        "F_Hbeta_e_sum_dered",
        "F_[NII]6583_sum_dered",
        "F_[NII]6583_e_sum_dered",
        "F_[SII]6716_sum_dered",
        "F_[SII]6716_e_sum_dered",
        "F_[SII]6731_sum_dered",
        "F_[SII]6731_e_sum_dered",
        "F_[OIII]5007_sum_dered",
        "F_[OIII]5007_e_sum_dered",
        "F_[OII]3727_sum_dered",
        "F_[OII]3727_e_sum_dered",
    ]
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for metallicity error propagation: {missing}")

    cal_names = [
        "Z_N2_Brazzini2024",
        "Z_O3N2_Brazzini2024",
        "Z_N2S2Halpha_Brazzini2024",
        "Z_R3_Brazzini2024",
        "Z_R23_Maiolino2008",
        "Z_N2_Maiolino2008",
        "Z_R23_Curti2017",
        "Z_R3_Curti2017",
        "Z_N2_Curti2017",
        "Z_O3N2_Curti2017",
        "Z_R_Pilyugin2016_highN2",
        "Z_R_Pilyugin2016_lowN2",
        "Z_S_Pilyugin2016_highN2",
        "Z_S_Pilyugin2016_lowN2",
        "Z_R23_KK2004",
        "Z_NII_KD2002",
        "Z_D2016",
        "Z_O3N2_M2013",
        "Z_N2_M2013",
        "Z_C2001",
        "Z_N2_PP2004",
        "Z_O3N2_PP2004",
        "Z_N2_Brown2016",
        "Z_O3N2_Brown2016",
        "Z_N2O2_Brown2016",
        "Z_N2O2_KD2002",
    ]
    for col in cal_names:
        out[f"{col}_e"] = np.nan

    base_calibrations = build_calibrations()
    rng = np.random.default_rng(seed)

    def mc_percentile_sigma(draws):
        draws = draws[np.isfinite(draws)]
        if draws.size < 10:
            return np.nan
        p16, _, p84 = np.percentile(draws, [16, 50, 84])
        return 0.5 * (p84 - p16)

    for i in range(len(out)):
        vals = out.loc[
            i,
            [
                "F_Halpha_sum_dered",
                "F_Halpha_e_sum_dered",
                "F_Hbeta_sum_dered",
                "F_Hbeta_e_sum_dered",
                "F_[NII]6583_sum_dered",
                "F_[NII]6583_e_sum_dered",
                "F_[SII]6716_sum_dered",
                "F_[SII]6716_e_sum_dered",
                "F_[SII]6731_sum_dered",
                "F_[SII]6731_e_sum_dered",
                "F_[OIII]5007_sum_dered",
                "F_[OIII]5007_e_sum_dered",
                "F_[OII]3727_sum_dered",
                "F_[OII]3727_e_sum_dered",
            ],
        ].astype(float).to_numpy()
        if not np.all(np.isfinite(vals)):
            continue
        (
            f_ha,
            e_ha,
            f_hb,
            e_hb,
            f_nii,
            e_nii,
            f_sii1,
            e_sii1,
            f_sii2,
            e_sii2,
            f_oiii,
            e_oiii,
            f_oii,
            e_oii,
        ) = vals
        if min(f_ha, f_hb, f_nii, f_sii1, f_sii2, f_oiii, f_oii) <= 0:
            continue
        if min(e_ha, e_hb, e_nii, e_sii1, e_sii2, e_oiii, e_oii) < 0:
            continue

        # d_ha = np.clip(safe_normal(rng, f_ha, e_ha, n_mc), 1e-30, None)
        # d_hb = np.clip(safe_normal(rng, f_hb, e_hb, n_mc), 1e-30, None)
        # d_nii = np.clip(safe_normal(rng, f_nii, e_nii, n_mc), 1e-30, None)
        # d_sii1 = np.clip(safe_normal(rng, f_sii1, e_sii1, n_mc), 1e-30, None)
        # d_sii2 = np.clip(safe_normal(rng, f_sii2, e_sii2, n_mc), 1e-30, None)
        # d_oiii = np.clip(safe_normal(rng, f_oiii, e_oiii, n_mc), 1e-30, None)
        # d_oii = np.clip(safe_normal(rng, f_oii, e_oii, n_mc), 1e-30, None)
        
        d_ha = safe_normal(rng, f_ha, e_ha, n_mc)
        d_hb = safe_normal(rng, f_hb, e_hb, n_mc)
        d_nii = safe_normal(rng, f_nii, e_nii, n_mc)
        d_sii1 = safe_normal(rng, f_sii1, e_sii1, n_mc)
        d_sii2 = safe_normal(rng, f_sii2, e_sii2, n_mc)
        d_oiii = safe_normal(rng, f_oiii, e_oiii, n_mc)
        d_oii = safe_normal(rng, f_oii, e_oii, n_mc)

        good = (
            (d_ha > 0) & (d_hb > 0) & (d_nii > 0) &
            (d_sii1 > 0) & (d_sii2 > 0) &
            (d_oiii > 0) & (d_oii > 0)
        )

        d_ha, d_hb, d_nii = d_ha[good], d_hb[good], d_nii[good]
        d_sii1, d_sii2 = d_sii1[good], d_sii2[good]
        d_oiii, d_oii = d_oiii[good], d_oii[good]

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            N2 = d_nii / d_ha
            S2 = (d_sii1 + d_sii2) / d_ha
            R3 = d_oiii / d_hb
            R2 = d_oii / d_hb
            R23 = R2 + R3

        z_draws = {}
        base_name_map = {
            "N2_Brazzini2024": "Z_N2_Brazzini2024",
            "O3N2_Brazzini2024": "Z_O3N2_Brazzini2024",
            "N2S2Halpha_Brazzini2024": "Z_N2S2Halpha_Brazzini2024",
            "R3_Brazzini2024": "Z_R3_Brazzini2024",
            "R23_Maiolino2008": "Z_R23_Maiolino2008",
            "N2_Maiolino2008": "Z_N2_Maiolino2008",
            "R23_Curti2017": "Z_R23_Curti2017",
            "R3_Curti2017": "Z_R3_Curti2017",
            "N2_Curti2017": "Z_N2_Curti2017",
            "O3N2_Curti2017": "Z_O3N2_Curti2017",
        }

        for cal in base_calibrations:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ratio = cal.R_func(N2, S2, R3, R2, R23)
            logR = np.full_like(ratio, np.nan, dtype=float)
            goodR = np.isfinite(ratio) & (ratio > 0)
            logR[goodR] = np.log10(ratio[goodR])
            z = np.full_like(logR, np.nan, dtype=float)
            cal_z_min = cal.valid_oh_range[0] - 8.69
            cal_z_max = cal.valid_oh_range[1] - 8.69
            for j, logR_j in enumerate(logR):
                if not np.isfinite(logR_j):
                    continue
                z0 = invert_logR_to_Z(
                    cal.coeffs,
                    logR_j,
                    z_min=cal_z_min,
                    z_max=cal_z_max,
                    branch=cal.branch,
                )
                z[j] = (
                    odr_refine_z(cal.coeffs, logR_j, z0, z_min=cal_z_min, z_max=cal_z_max)
                    if np.isfinite(z0)
                    else np.nan
                )
            col = base_name_map[cal.name]
            z_draws[col] = _mask_to_valid_metallicity_range(z + 8.69, col)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            SII = d_sii1 + d_sii2
            R3_pg = (1 + 1 / 2.89) * d_oiii / d_hb
            N2_pg = d_nii / d_ha
            S2_pg = SII / d_ha
            R2_pg = d_oii / d_hb
            R23_pg = (d_oiii + d_oii) / d_hb
            O32_pg = d_oiii / d_oii

            z_draws["Z_R_Pilyugin2016_highN2"] = 8.589 + 0.022 * np.log10(R3_pg / R2_pg) + 0.399 * np.log10(N2_pg) + (-0.137 + 0.164 * np.log10(R3_pg / R2_pg) + 0.589 * np.log10(N2_pg)) * np.log10(R2_pg)
            z_draws["Z_R_Pilyugin2016_lowN2"] = 7.932 + 0.944 * np.log10(R3_pg / R2_pg) + 0.695 * np.log10(N2_pg) + (0.970 - 0.291 * np.log10(R3_pg / R2_pg) - 0.019 * np.log10(N2_pg)) * np.log10(R2_pg)
            z_draws["Z_S_Pilyugin2016_highN2"] = 8.424 + 0.030 * np.log10(R3_pg / S2_pg) + 0.751 * np.log10(N2_pg) + (-0.349 + 0.182 * np.log10(R3_pg / S2_pg) + 0.508 * np.log10(N2_pg)) * np.log10(S2_pg)
            z_draws["Z_S_Pilyugin2016_lowN2"] = 8.072 + 0.789 * np.log10(R3_pg / S2_pg) + 0.726 * np.log10(N2_pg) + (1.069 - 0.170 * np.log10(R3_pg / S2_pg) + 0.022 * np.log10(N2_pg)) * np.log10(S2_pg)
            x = np.log10(R23_pg)
            y = np.log10(O32_pg)
            z_draws["Z_R23_KK2004"] = 9.11 - 0.218 * x - 0.0587 * x**2 - 0.330 * x**3 - 0.199 * x**4 - y * (0.00235 - 0.1105 * x - 0.051 * x**2 - 0.04085 * x**3 - 0.003585 * x**4)
            z_draws["Z_NII_KD2002"] = np.log10(1.54020 + 1.26602 * d_nii / d_oii + 0.167977 * (d_nii / d_oii) ** 2) + 8.93
            y = np.log10(d_nii / SII) + 0.264 * np.log10(d_nii / d_ha)
            z_draws["Z_D2016"] = 8.77 + y
            z_draws["Z_O3N2_M2013"] = 8.533 - 0.214 * np.log10(R3 / N2)
            z_draws["Z_N2_M2013"] = 8.743 + 0.462 * np.log10(N2)
            z_draws["Z_C2001"] = np.log10(5.09e-4 * (d_oii / d_oiii) ** 0.17 * (d_nii / SII / 0.85) ** 1.17) + 12.0
            z_draws["Z_N2_PP2004"] = 9.37 + 2.03 * np.log10(N2) + 1.26 * np.log10(N2) ** 2 + 0.32 * np.log10(N2) ** 3
            z_draws["Z_O3N2_PP2004"] = 8.73 - 0.32 * np.log10(R3 / N2)
            z_draws["Z_N2_Brown2016"] = 9.12 + 0.58 * np.log10(N2)
            z_draws["Z_O3N2_Brown2016"] = 8.98 - 0.32 * np.log10(R3 / N2)
            z_draws["Z_N2O2_Brown2016"] = 9.20 + 0.54 * np.log10(d_nii / d_oii)
            z_draws["Z_N2O2_KD2002"] = np.log10(1.54020 + 1.26602 * d_nii / d_oii + 0.167977 * (d_nii / d_oii) ** 2) + 8.93

        for col in cal_names:
            z_draws[col] = _mask_to_valid_metallicity_range(z_draws[col], col)
            out.at[i, f"{col}_e"] = mc_percentile_sigma(np.asarray(z_draws[col], dtype=float))

    return out


def deproject_pixels_to_disk_pc(x, y, x0, y0, pa_deg=23.0, inc_deg=56.0, pc_per_pixel=1.0):
    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    theta = np.deg2rad(pa_deg)
    x_major = dx * np.cos(theta) + dy * np.sin(theta)
    y_minor = -dx * np.sin(theta) + dy * np.cos(theta)
    y_minor_deproj = y_minor / np.cos(np.deg2rad(inc_deg))
    return x_major * pc_per_pixel, y_minor_deproj * pc_per_pixel


def clark_evans_R(coords):
    n = len(coords)
    if n < 2:
        return np.nan, np.nan, np.nan
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    r_obs = np.mean(d[:, 1])
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    area = (xmax - xmin) * (ymax - ymin)
    lam = n / area
    r_exp = 1.0 / (2.0 * np.sqrt(lam))
    return r_obs / r_exp, r_obs, r_exp


def ripley_K_rect(coords, radii_pc):
    n = len(coords)
    if n < 2:
        return np.full_like(radii_pc, np.nan, dtype=float)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    area = (xmax - xmin) * (ymax - ymin)
    D = squareform(pdist(coords))
    K = np.zeros_like(radii_pc, dtype=float)
    for i, r in enumerate(radii_pc):
        count = np.sum((D <= r) & (D > 0))
        K[i] = area * count / (n * (n - 1))
    return K


def pair_correlation_rect(coords, r_edges):
    n = len(coords)
    if n < 2:
        return np.full(len(r_edges) - 1, np.nan), 0.0
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    area = (xmax - xmin) * (ymax - ymin)
    lam = n / area
    d = pdist(coords)
    counts, _ = np.histogram(d, bins=r_edges)
    shell_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    expected = 0.5 * n * lam * shell_areas
    return counts / expected, lam


def mst_mean_edge(coords):
    n = len(coords)
    if n < 2:
        return np.nan
    D = squareform(pdist(coords))
    return np.mean(minimum_spanning_tree(D).data)


def correlation_dimension(coords, r_min=50.0, r_max=500.0, n_bins=20):
    n = len(coords)
    if n < 3:
        return np.nan, None, None
    d = pdist(coords)
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)
    C = np.array([(d < r).sum() for r in r_vals], dtype=float)
    C /= (n * (n - 1) / 2.0)
    good = C > 0
    if good.sum() < 5:
        return np.nan, r_vals, C
    coeff = np.polyfit(np.log10(r_vals[good]), np.log10(C[good]), 1)
    return coeff[0], r_vals, C


def add_clustering_metrics(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    x0: float = 1024.0,
    y0: float = 1024.0,
    pixel_scale_arcsec: float = 0.32,
    distance_m33_pc: float = 883000.0,
    inclination_deg: float = 56.0,
    pa_deg: float = 23.0,
):
    out = df.copy()
    pc_per_arcsec = distance_m33_pc / 206265.0
    pc_per_pixel = pixel_scale_arcsec * pc_per_arcsec
    out["x_deproj_pc"], out["y_deproj_pc"] = deproject_pixels_to_disk_pc(
        out[x_col].values, out[y_col].values, x0=x0, y0=y0, pa_deg=pa_deg, inc_deg=inclination_deg, pc_per_pixel=pc_per_pixel
    )
    coords_pc = out[["x_deproj_pc", "y_deproj_pc"]].to_numpy()
    n_regions = len(coords_pc)
    tree = cKDTree(coords_pc)
    k_needed = min(6, n_regions)
    distances, _ = tree.query(coords_pc, k=k_needed)
    if k_needed == 1:
        distances = distances[:, np.newaxis]
    nearest_neighbor_pc = np.full(n_regions, np.nan)
    fifth_neighbor_pc = np.full(n_regions, np.nan)
    if n_regions >= 2:
        nearest_neighbor_pc = distances[:, 1]
    if n_regions >= 6:
        fifth_neighbor_pc = distances[:, 5]
    out["nearest_neighbor_pc_deproj"] = nearest_neighbor_pc
    out["n_regions_within_100pc_deproj"] = np.array([len(tree.query_ball_point(point, r=100.0)) - 1 for point in coords_pc], dtype=int)
    out["distance_5th_closest_pc_deproj"] = fifth_neighbor_pc
    out["sigma5_per_pc2_deproj"] = np.where(np.isfinite(out["distance_5th_closest_pc_deproj"]), 5.0 / (np.pi * out["distance_5th_closest_pc_deproj"] ** 2), np.nan)

    R_ce, nn_obs_mean, nn_exp_mean = clark_evans_R(coords_pc)
    radii_pc = np.arange(25, 1001, 25)
    K_r = ripley_K_rect(coords_pc, radii_pc)
    L_r = np.sqrt(K_r / np.pi) - radii_pc
    r_edges = np.arange(0, 1001, 25)
    g_r, _ = pair_correlation_rect(coords_pc, r_edges)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    mst_mean_pc = mst_mean_edge(coords_pc)
    D2, _, _ = correlation_dimension(coords_pc)

    global_stats = pd.DataFrame(
        {
            "statistic": [
                "N_regions",
                "pc_per_pixel",
                "inclination_deg",
                "position_angle_deg",
                "Clark_Evans_R",
                "mean_observed_nearest_neighbor_pc",
                "mean_CSR_nearest_neighbor_pc",
                "mean_MST_edge_pc",
                "correlation_dimension_D2",
            ],
            "value": [n_regions, pc_per_pixel, inclination_deg, pa_deg, R_ce, nn_obs_mean, nn_exp_mean, mst_mean_pc, D2],
        }
    )
    ripley_df = pd.DataFrame({"r_pc": radii_pc, "Ripley_K": K_r, "Besag_L_minus_r_pc": L_r})
    pcf_df = pd.DataFrame({"r_center_pc": r_centers, "pair_correlation_g_r": g_r})
    return out, global_stats, ripley_df, pcf_df


def write_clustering_outputs(
    df: pd.DataFrame,
    global_stats: pd.DataFrame,
    ripley_df: pd.DataFrame,
    pcf_df: pd.DataFrame,
    method: str = "summed_map",
    dig_mode: str = "no_dig",
) -> dict[str, Path]:
    out_catalog = paths.derived_catalog_csv("clustering", method=method, dig_mode=dig_mode)
    out_global = paths.derived_catalog_dir(method, dig_mode=dig_mode) / "clustering_global_statistics.csv"
    out_ripley = paths.derived_catalog_dir(method, dig_mode=dig_mode) / "clustering_ripley_profile.csv"
    out_pcf = paths.derived_catalog_dir(method, dig_mode=dig_mode) / "clustering_pair_correlation_profile.csv"
    write_catalog(df, out_catalog)
    write_catalog(global_stats, out_global)
    write_catalog(ripley_df, out_ripley)
    write_catalog(pcf_df, out_pcf)
    return {"catalog": out_catalog, "global": out_global, "ripley": out_ripley, "pcf": out_pcf}


def build_total_catalog(config: DerivedConfig | None = None, method: str = "summed_map", dig_mode: str = "no_dig"):
    config = config or DerivedConfig()
    all_catalog = merge_field_flux_catalogs(method=method, dig_mode=dig_mode)
    cat = add_logU_KK04(all_catalog.copy(), n_mc=config.logu_n_mc)
    density = add_electron_density(cat.copy(), n_mc=config.density_n_mc)
    pressure = add_thermal_pressure(
        density.copy(),
        T_e=config.electron_temperature_K,
        particle_factor=config.ionized_gas_particle_factor,
    )
    peak_properties = add_peak_region_properties(
        pressure.copy(),
        distance_mpc=config.m33_distance_mpc,
        te_default=config.electron_temperature_K,
        n_mc=config.density_n_mc,
    )
    symmetry = add_symmetry_class(peak_properties.copy())
    metallicity = add_metallicity_columns(symmetry.copy())
    metallicity = add_metallicity_error_columns(metallicity.copy(), n_mc=config.metallicity_n_mc)
    clustered, global_stats, ripley_df, pcf_df = add_clustering_metrics(metallicity.copy())
    return {
        "all_catalog": all_catalog,
        "with_logu": cat,
        "with_density": density,
        "with_pressure": pressure,
        "with_peak_properties": peak_properties,
        "with_symmetry": symmetry,
        "with_metallicity": metallicity,
        "with_clustering": clustered,
        "global_stats": global_stats,
        "ripley_df": ripley_df,
        "pcf_df": pcf_df,
    }
