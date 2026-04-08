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


def merge_field_flux_catalogs(catalog_dir: Path | None = None) -> pd.DataFrame:
    catalog_dir = catalog_dir or paths.flux_catalog_dir()
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


def write_total_flux_catalog(df: pd.DataFrame) -> Path:
    out_path = paths.total_flux_catalog_csv()
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


def add_logU_KK04(
    df: pd.DataFrame,
    n_mc: int = 2000,
    seed: int = 123,
    metallicity_cal: str = "M13_O3N2",
    Z_intrinsic_sigma_dex: float = 0.18,
    apply_Z_intrinsic_scatter: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    prefix = "F"
    oii, oii_e = _col(prefix, "[OII]3727", "sum_dered")
    o3_5007, o3_5007_e = _col(prefix, "[OIII]5007", "sum_dered")
    hb, hb_e = _col(prefix, "Hbeta", "sum_dered")
    ha, ha_e = _col(prefix, "Halpha", "sum_dered")
    nii, nii_e = _col(prefix, "[NII]6583", "sum_dered")

    rng = np.random.default_rng(seed)
    O32_med = np.full(len(out), np.nan)
    O32_sig = np.full(len(out), np.nan)
    logO32_med = np.full(len(out), np.nan)
    logO32_sig = np.full(len(out), np.nan)
    O3N2_med = np.full(len(out), np.nan)
    O3N2_sig = np.full(len(out), np.nan)
    Z_med = np.full(len(out), np.nan)
    Z_sig = np.full(len(out), np.nan)
    logq_med = np.full(len(out), np.nan)
    logq_sig = np.full(len(out), np.nan)
    logU_med = np.full(len(out), np.nan)
    logU_sig = np.full(len(out), np.nan)

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

        d_o32 = ((1 + 1 / 2.98) * d_5007) / d_oii
        mask_o32 = np.isfinite(d_o32) & (d_o32 > 0)
        if mask_o32.sum() < 50:
            continue
        d_o32 = d_o32[mask_o32]
        d_logO32 = np.log10(d_o32)

        d_o3n2 = np.log10((d_5007 / d_hb) * (d_ha / d_nii))
        mask_o3n2 = np.isfinite(d_o3n2)
        if mask_o3n2.sum() < 50:
            continue
        d_o3n2 = d_o3n2[mask_o3n2]

        if metallicity_cal == "M13_O3N2":
            d_Z = 8.533 - 0.214 * d_o3n2
        elif metallicity_cal == "PP04_O3N2":
            d_Z = 8.73 - 0.32 * d_o3n2
        else:
            raise ValueError("metallicity_cal must be 'M13_O3N2' or 'PP04_O3N2'.")

        if apply_Z_intrinsic_scatter and Z_intrinsic_sigma_dex and Z_intrinsic_sigma_dex > 0:
            d_Z = d_Z + rng.normal(0.0, Z_intrinsic_sigma_dex, size=d_Z.size)

        n = min(d_logO32.size, d_Z.size)
        d_logq = kk04_logq_from_logO32_and_Z(d_logO32[:n], d_Z[:n])
        d_logU = d_logq - np.log10(C_CM_S)
        mask_logu = np.isfinite(d_logq) & np.isfinite(d_logU)
        if mask_logu.sum() < 50:
            continue
        d_logq = d_logq[mask_logu]
        d_logU = d_logU[mask_logu]

        O32_med[i], O32_sig[i] = med_sig(d_o32)
        logO32_med[i], logO32_sig[i] = med_sig(d_logO32)
        O3N2_med[i], O3N2_sig[i] = med_sig(d_o3n2)
        Z_med[i], Z_sig[i] = med_sig(d_Z[:n])
        logq_med[i], logq_sig[i] = med_sig(d_logq)
        logU_med[i], logU_sig[i] = med_sig(d_logU)

    out["O32"] = O32_med
    out["O32_e"] = O32_sig
    out["logO32"] = logO32_med
    out["logO32_e"] = logO32_sig
    out["O3N2"] = O3N2_med
    out["O3N2_e"] = O3N2_sig
    out["Z_12logOH"] = Z_med
    out["Z_12logOH_e"] = Z_sig
    out["logq_KK04"] = logq_med
    out["logq_KK04_e"] = logq_sig
    out["logU_KK04"] = logU_med
    out["logU_KK04_e"] = logU_sig
    out["logU_flag"] = np.where(np.isfinite(out["logU_KK04"]), "ok", "invalid")
    out["logU_meta_cal"] = metallicity_cal
    out["logU_meta_Zscatter_dex"] = Z_intrinsic_sigma_dex if apply_Z_intrinsic_scatter else 0.0
    return out


def add_electron_density(
    df: pd.DataFrame,
    te_default: float = 1.0e4,
    use_dereddened: bool = True,
    n_mc: int = 200,
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

    out["SII_ratio_6716_6731"] = out[c6716] / out[c6731]
    if c6716e is not None and c6731e is not None:
        ratio = out["SII_ratio_6716_6731"].to_numpy(dtype=float)
        f1 = out[c6716].to_numpy(dtype=float)
        f2 = out[c6731].to_numpy(dtype=float)
        e1 = out[c6716e].to_numpy(dtype=float)
        e2 = out[c6731e].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["SII_ratio_6716_6731_e"] = np.abs(ratio) * np.sqrt((e1 / f1) ** 2 + (e2 / f2) ** 2)

    atom = pn.Atom("S", 2)
    valid = np.isfinite(out["SII_ratio_6716_6731"]) & (out[c6716] > 0) & (out[c6731] > 0)
    ne = np.full(len(out), np.nan, dtype=float)
    ne[valid.to_numpy()] = atom.getTemDen(out.loc[valid, "SII_ratio_6716_6731"].to_numpy(dtype=float), tem=te_default, wave1=6716, wave2=6731)
    out["ne_SII_cm3"] = ne

    def monte_carlo_ne_from_ratio(r, rerr, seed=0):
        rng = np.random.default_rng(seed)
        draws = rng.normal(loc=r, scale=rerr, size=n_mc)
        draws = draws[np.isfinite(draws) & (draws > 0)]
        if draws.size == 0:
            return np.nan, np.nan, np.nan
        ne_draws = atom.getTemDen(draws, tem=te_default, wave1=6716, wave2=6731)
        ne_draws = np.atleast_1d(ne_draws)
        ne_draws = ne_draws[np.isfinite(ne_draws) & (ne_draws > 0)]
        if ne_draws.size == 0:
            return np.nan, np.nan, np.nan
        p16, p50, p84 = np.percentile(ne_draws, [16, 50, 84])
        return p50, p50 - p16, p84 - p50

    if "SII_ratio_6716_6731_e" in out.columns:
        mc = np.array(
            [
                monte_carlo_ne_from_ratio(r, e, seed=i) if np.isfinite(r) and np.isfinite(e) and e > 0 else (np.nan, np.nan, np.nan)
                for i, (r, e) in enumerate(zip(out["SII_ratio_6716_6731"], out["SII_ratio_6716_6731_e"]))
            ],
            dtype=float,
        )
        out["ne_SII_cm3_mc"] = mc[:, 0]
        out["ne_SII_cm3_mc_minus"] = mc[:, 1]
        out["ne_SII_cm3_mc_plus"] = mc[:, 2]
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


def poly_eval(coeffs, z):
    return np.polyval(coeffs[::-1], z)


def invert_logR_to_Z(coeffs, logR_obs, z_min=6.5, z_max=9.5, ngrid=2000):
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
    if idx.size == 0:
        return zgrid[np.argmin(np.abs(fgrid))]
    best_i = None
    best_val = np.inf
    for i in idx:
        midz = 0.5 * (zgrid[i] + zgrid[i + 1])
        val = abs(poly_eval(coeffs, midz) - logR_obs)
        if val < best_val:
            best_val = val
            best_i = i
    a, b = zgrid[best_i], zgrid[best_i + 1]
    try:
        return brentq(lambda z: poly_eval(coeffs, z) - logR_obs, a, b, maxiter=200)
    except ValueError:
        return zgrid[np.argmin(np.abs(fgrid))]


def odr_refine_z(coeffs, logR_obs, z0, sx=1.0, sy=1.0):
    if not (np.isfinite(z0) and np.isfinite(logR_obs)):
        return np.nan

    def f(beta, x):
        z = beta[0]
        return poly_eval(coeffs, z) + 0.0 * x

    model = odr.Model(f)
    data = odr.RealData(np.array([0.0]), np.array([logR_obs]), sx=np.array([sx]), sy=np.array([sy]))
    out = odr.ODR(data, model, beta0=[z0]).run()
    return out.beta[0]


@dataclass(frozen=True)
class Calibration:
    name: str
    coeffs: np.ndarray
    R_func: callable


def build_calibrations():
    return [
        Calibration("N2S2Halpha_Brazzini2024", np.array([0.24, 2.21, 0.76]), lambda N2, S2, R3, R2, R23: N2 / S2 * (N2**0.264)),
        Calibration("N2_Brazzini2024", np.array([-0.41, 0.57, -4.91, -5.81, -1.95]), lambda N2, S2, R3, R2, R23: N2),
        Calibration("O3N2_Brazzini2024", np.array([-0.51, -7.74, -6.12, -1.60]), lambda N2, S2, R3, R2, R23: R3 / N2),
        Calibration("R3_Brazzini2024", np.array([-0.84, -5.86, -6.27, -1.95]), lambda N2, S2, R3, R2, R23: R3),
        Calibration("R23_Maiolino2008", np.array([0.7462, -0.7149, -0.9401, -0.6154, -0.2524]), lambda N2, S2, R3, R2, R23: R23),
        Calibration("N2_Maiolino2008", np.array([-0.7732, 1.2357, -0.2811, -0.7201, -0.3330]), lambda N2, S2, R3, R2, R23: N2),
        Calibration("R3_Curti2017", np.array([-0.277, -3.549, -3.593, -0.981]), lambda N2, S2, R3, R2, R23: R3),
        Calibration("R23_Curti2017", np.array([0.527, -1.569, -1.652, -0.421]), lambda N2, S2, R3, R2, R23: R23),
        Calibration("N2_Curti2017", np.array([-0.489, 1.513, -2.554, -5.293, -2.867]), lambda N2, S2, R3, R2, R23: N2),
        Calibration("O3N2_Curti2017", np.array([-0.281, -4.765, -2.268]), lambda N2, S2, R3, R2, R23: R3 / N2),
    ]


def compute_metallicities(full_catalog, z_min=6.5, z_max=9.5, use_odr=True, sx=1.0, sy=1.0):
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
            z0 = invert_logR_to_Z(cal.coeffs, logR_i, z_min=z_min, z_max=z_max)
            Z[i] = odr_refine_z(cal.coeffs, logR_i, z0, sx=sx, sy=sy) if use_odr and np.isfinite(z0) else z0
        out[cal.name] = Z
    return out


def add_metallicity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_dict = compute_metallicities(out, use_odr=True)
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
    return out


def add_metallicity_error_columns(df: pd.DataFrame, n_mc: int = 500, seed: int = 123) -> pd.DataFrame:
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

        d_ha = np.clip(safe_normal(rng, f_ha, e_ha, n_mc), 1e-30, None)
        d_hb = np.clip(safe_normal(rng, f_hb, e_hb, n_mc), 1e-30, None)
        d_nii = np.clip(safe_normal(rng, f_nii, e_nii, n_mc), 1e-30, None)
        d_sii1 = np.clip(safe_normal(rng, f_sii1, e_sii1, n_mc), 1e-30, None)
        d_sii2 = np.clip(safe_normal(rng, f_sii2, e_sii2, n_mc), 1e-30, None)
        d_oiii = np.clip(safe_normal(rng, f_oiii, e_oiii, n_mc), 1e-30, None)
        d_oii = np.clip(safe_normal(rng, f_oii, e_oii, n_mc), 1e-30, None)

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
            for j, logR_j in enumerate(logR):
                if not np.isfinite(logR_j):
                    continue
                z0 = invert_logR_to_Z(cal.coeffs, logR_j)
                z[j] = odr_refine_z(cal.coeffs, logR_j, z0) if np.isfinite(z0) else np.nan
            z_draws[base_name_map[cal.name]] = z + 8.69

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


def write_clustering_outputs(df: pd.DataFrame, global_stats: pd.DataFrame, ripley_df: pd.DataFrame, pcf_df: pd.DataFrame) -> dict[str, Path]:
    out_catalog = paths.flux_catalog_dir() / "total_flux_catalog_with_deprojected_clustering_metrics.csv"
    out_global = paths.flux_catalog_dir() / "clustering_global_statistics.csv"
    out_ripley = paths.flux_catalog_dir() / "clustering_ripley_profile.csv"
    out_pcf = paths.flux_catalog_dir() / "clustering_pair_correlation_profile.csv"
    write_catalog(df, out_catalog)
    write_catalog(global_stats, out_global)
    write_catalog(ripley_df, out_ripley)
    write_catalog(pcf_df, out_pcf)
    return {"catalog": out_catalog, "global": out_global, "ripley": out_ripley, "pcf": out_pcf}


def build_total_catalog(config: DerivedConfig | None = None):
    config = config or DerivedConfig()
    all_catalog = merge_field_flux_catalogs()
    cat = add_logU_KK04(all_catalog.copy(), n_mc=config.logu_n_mc)
    density = add_electron_density(cat.copy(), n_mc=config.density_n_mc)
    symmetry = add_symmetry_class(density.copy())
    metallicity = add_metallicity_columns(symmetry.copy())
    metallicity = add_metallicity_error_columns(metallicity.copy(), n_mc=config.metallicity_n_mc)
    clustered, global_stats, ripley_df, pcf_df = add_clustering_metrics(metallicity.copy())
    return {
        "all_catalog": all_catalog,
        "with_logu": cat,
        "with_density": density,
        "with_symmetry": symmetry,
        "with_metallicity": metallicity,
        "with_clustering": clustered,
        "global_stats": global_stats,
        "ripley_df": ripley_df,
        "pcf_df": pcf_df,
    }
