import unittest

import numpy as np
import pandas as pd

from m33_pipeline.derived import (
    add_electron_density,
    add_logU_KK04,
    add_metallicity_columns,
    add_metallicity_error_columns,
    add_peak_region_properties,
    add_symmetry_class,
    add_thermal_pressure,
    metallicity_valid_range_summary,
    merge_field_flux_catalogs,
)


class DerivedTest(unittest.TestCase):
    def test_logu_kk04_iteratively_solves_r23_and_o32_on_upper_branch(self):
        df = pd.DataFrame(
            {
                "F_[OII]3727_sum_dered": [1.0],
                "F_[OIII]5007_sum_dered": [0.8],
                "F_Hbeta_sum_dered": [0.5],
                "F_Halpha_sum_dered": [1.43],
                "F_[NII]6583_sum_dered": [0.2],
            }
        )

        out = add_logU_KK04(df, n_mc=0)

        self.assertEqual(out.loc[0, "logU_meta_cal"], "KK04_iterative_R23_O32")
        self.assertEqual(out.loc[0, "logU_KK04_branch"], "upper")
        self.assertTrue(out.loc[0, "logU_KK04_converged"])
        self.assertGreater(out.loc[0, "logU_KK04_iterations"], 1)
        self.assertTrue(np.isfinite(out.loc[0, "Z_12logOH"]))
        self.assertTrue(np.isfinite(out.loc[0, "logU_KK04"]))

    def test_logu_kk04_uses_n2o2_to_select_lower_branch(self):
        df = pd.DataFrame(
            {
                "F_[OII]3727_sum_dered": [1.0],
                "F_[OIII]5007_sum_dered": [1.5],
                "F_Hbeta_sum_dered": [0.5],
                "F_Halpha_sum_dered": [1.43],
                "F_[NII]6583_sum_dered": [0.02],
            }
        )

        out = add_logU_KK04(df, n_mc=0)

        self.assertEqual(out.loc[0, "logU_KK04_branch"], "lower")
        self.assertTrue(out.loc[0, "logU_KK04_converged"])
        self.assertLess(out.loc[0, "Z_12logOH"], 8.5)

    def test_logu_kk04_monte_carlo_runs_iterative_solution_for_flux_draws(self):
        df = pd.DataFrame(
            {
                "F_[OII]3727_sum_dered": [1.0],
                "F_[OII]3727_e_sum_dered": [0.02],
                "F_[OIII]5007_sum_dered": [0.8],
                "F_[OIII]5007_e_sum_dered": [0.02],
                "F_Hbeta_sum_dered": [0.5],
                "F_Hbeta_e_sum_dered": [0.01],
                "F_Halpha_sum_dered": [1.43],
                "F_Halpha_e_sum_dered": [0.02],
                "F_[NII]6583_sum_dered": [0.2],
                "F_[NII]6583_e_sum_dered": [0.01],
            }
        )

        out = add_logU_KK04(df, n_mc=100, seed=1)

        self.assertEqual(out.loc[0, "logU_KK04_branch"], "upper")
        self.assertGreater(out.loc[0, "logU_KK04_converged_fraction"], 0.9)
        self.assertGreater(out.loc[0, "logU_KK04_e"], 0.0)

    def test_add_symmetry_class(self):
        df = pd.DataFrame(
            {
                "radius_p16_pc": [5.0, 5.0],
                "radius_p50_pc": [6.0, 6.0],
                "radius_p84_pc": [5.2, 15.0],
            }
        )
        out = add_symmetry_class(df)
        self.assertEqual(out.loc[0, "symmetry_class"], "symmetric")
        self.assertEqual(out.loc[1, "symmetry_class"], "asymmetric")

    def test_merge_field_flux_catalogs_has_rows(self):
        df = merge_field_flux_catalogs()
        self.assertGreater(len(df), 0)
        self.assertIn("field", df.columns)

    def test_add_metallicity_error_columns_adds_error_fields(self):
        df = pd.DataFrame(
            {
                "F_Halpha_sum_dered": [1.0],
                "F_Halpha_e_sum_dered": [0.05],
                "F_Hbeta_sum_dered": [0.3],
                "F_Hbeta_e_sum_dered": [0.02],
                "F_[NII]6583_sum_dered": [0.2],
                "F_[NII]6583_e_sum_dered": [0.01],
                "F_[SII]6716_sum_dered": [0.15],
                "F_[SII]6716_e_sum_dered": [0.01],
                "F_[SII]6731_sum_dered": [0.12],
                "F_[SII]6731_e_sum_dered": [0.01],
                "F_[OIII]5007_sum_dered": [0.4],
                "F_[OIII]5007_e_sum_dered": [0.02],
                "F_[OII]3727_sum_dered": [0.5],
                "F_[OII]3727_e_sum_dered": [0.03],
            }
        )
        out = add_metallicity_error_columns(df, n_mc=50, seed=1)
        self.assertIn("Z_N2_Brazzini2024_e", out.columns)
        self.assertIn("Z_O3N2_M2013_e", out.columns)

    def test_add_metallicity_columns_populates_inversion_calibrations(self):
        df = pd.DataFrame(
            {
                "F_Halpha_sum_dered": [1.0],
                "F_Hbeta_sum_dered": [0.3],
                "F_[NII]6583_sum_dered": [0.2],
                "F_[SII]6716_sum_dered": [0.15],
                "F_[SII]6731_sum_dered": [0.12],
                "F_[OIII]5007_sum_dered": [0.4],
                "F_[OII]3727_sum_dered": [0.5],
            }
        )
        out = add_metallicity_columns(df, use_odr=False)
        self.assertTrue(pd.notna(out.loc[0, "Z_R3_Brazzini2024"]))
        self.assertTrue(pd.notna(out.loc[0, "Z_R23_Maiolino2008"]))
        self.assertTrue(pd.notna(out.loc[0, "Z_R23_Curti2017"]))
        self.assertTrue(pd.notna(out.loc[0, "Z_R3_Curti2017"]))

    def test_brazzini_n2s2halpha_uses_upper_branch(self):
        df = pd.DataFrame(
            {
                "F_Halpha_sum_dered": [1.0],
                "F_Hbeta_sum_dered": [0.35],
                "F_[NII]6583_sum_dered": [0.328],
                "F_[SII]6716_sum_dered": [0.45],
                "F_[SII]6731_sum_dered": [0.3885],
                "F_[OIII]5007_sum_dered": [0.4],
                "F_[OII]3727_sum_dered": [0.5],
            }
        )

        out = add_metallicity_columns(df, use_odr=False)

        self.assertGreater(out.loc[0, "Z_N2S2Halpha_Brazzini2024"], 8.0)
        self.assertLess(out.loc[0, "Z_N2S2Halpha_Brazzini2024"], 8.5)

    def test_metallicity_values_outside_valid_range_are_rejected(self):
        df = pd.DataFrame(
            {
                "F_Halpha_sum_dered": [1.0],
                "F_Hbeta_sum_dered": [0.3],
                "F_[NII]6583_sum_dered": [0.003],
                "F_[SII]6716_sum_dered": [0.08],
                "F_[SII]6731_sum_dered": [0.07],
                "F_[OIII]5007_sum_dered": [0.4],
                "F_[OII]3727_sum_dered": [0.5],
            }
        )

        out = add_metallicity_columns(df, use_odr=False)

        self.assertTrue(np.isnan(out.loc[0, "Z_N2S2Halpha_Brazzini2024"]))

    def test_metallicity_valid_range_summary_lists_references(self):
        summary = metallicity_valid_range_summary()
        row = summary.loc[summary["column"] == "Z_N2S2Halpha_Brazzini2024"].iloc[0]
        self.assertEqual(row["reference"], "Brazzini et al. 2024")
        self.assertEqual(row["indicator"], "N2S2Halpha")
        self.assertEqual(row["valid_12logOH_min"], 7.50)

    def test_electron_density_flags_low_density_limit_and_does_not_report_biased_mc_value(self):
        df = pd.DataFrame(
            {
                "F_[SII]6716_sum": [1.46],
                "F_[SII]6716_e_sum": [0.02],
                "F_[SII]6731_sum": [1.0],
                "F_[SII]6731_e_sum": [0.02],
            }
        )

        out = add_electron_density(df, use_dereddened=False, n_mc=200)

        self.assertEqual(out.loc[0, "ne_SII_flag"], "low_density_limit")
        self.assertTrue(np.isnan(out.loc[0, "ne_SII_cm3"]))
        self.assertTrue(np.isnan(out.loc[0, "ne_SII_cm3_mc"]))
        self.assertTrue(out.loc[0, "ne_SII_is_upper_limit"])
        self.assertGreater(out.loc[0, "ne_SII_cm3_upper_limit"], 1.0)

    def test_electron_density_uses_raw_fractional_errors_for_dereddened_ratio(self):
        df = pd.DataFrame(
            {
                "F_[SII]6716_sum": [1.3],
                "F_[SII]6716_e_sum": [0.013],
                "F_[SII]6731_sum": [1.0],
                "F_[SII]6731_e_sum": [0.01],
                "F_[SII]6716_sum_dered": [2.6],
                "F_[SII]6716_e_sum_dered": [1.0],
                "F_[SII]6731_sum_dered": [2.0],
                "F_[SII]6731_e_sum_dered": [1.0],
            }
        )

        out = add_electron_density(df, n_mc=0)

        expected = 1.3 * np.sqrt(0.01**2 + 0.01**2)
        self.assertAlmostEqual(out.loc[0, "SII_ratio_6716_6731_e"], expected)
        self.assertEqual(out.loc[0, "ne_SII_flag"], "ok")
        self.assertTrue(out.loc[0, "ne_SII_reliable"])

    def test_electron_density_reliability_requires_both_lines_above_snr_threshold(self):
        df = pd.DataFrame(
            {
                "F_[SII]6716_sum": [1.3, 1.3],
                "F_[SII]6716_e_sum": [0.013, 0.013],
                "F_[SII]6731_sum": [1.0, 1.0],
                "F_[SII]6731_e_sum": [0.01, 0.01],
                "SNR_[SII]6716_sum": [20.0, 20.0],
                "SNR_[SII]6731_sum": [20.0, 5.0],
            }
        )

        out = add_electron_density(df, use_dereddened=False, n_mc=0)

        self.assertTrue(out.loc[0, "ne_SII_reliable"])
        self.assertFalse(out.loc[1, "ne_SII_reliable"])
        self.assertEqual(out.loc[1, "ne_SII_min_line_snr"], 5.0)

    def test_electron_density_does_not_claim_upper_limit_if_ratio_error_never_reaches_physical_range(self):
        df = pd.DataFrame(
            {
                "F_[SII]6716_sum": [1.8],
                "F_[SII]6716_e_sum": [0.01],
                "F_[SII]6731_sum": [1.0],
                "F_[SII]6731_e_sum": [0.01],
            }
        )

        out = add_electron_density(df, use_dereddened=False, n_mc=0)

        self.assertEqual(out.loc[0, "ne_SII_flag"], "low_density_limit")
        self.assertFalse(out.loc[0, "ne_SII_is_upper_limit"])
        self.assertTrue(np.isnan(out.loc[0, "ne_SII_cm3_upper_limit"]))

    def test_thermal_pressure_adds_total_electron_and_physical_pressure(self):
        df = pd.DataFrame(
            {
                "ne_SII_cm3": [100.0],
                "ne_SII_reliable": [True],
                "ne_SII_flag": ["ok"],
            }
        )

        out = add_thermal_pressure(df, T_e=1.0e4, particle_factor=2.0)

        self.assertEqual(out.loc[0, "P_e_SII_over_k_K_cm3"], 1.0e6)
        self.assertEqual(out.loc[0, "P_SII_K_cm3"], 1.0e6)
        self.assertEqual(out.loc[0, "P_thermal_SII_over_k_K_cm3"], 2.0e6)
        self.assertAlmostEqual(out.loc[0, "log_P_thermal_SII_over_k"], np.log10(2.0e6))
        self.assertAlmostEqual(out.loc[0, "P_thermal_SII_dyn_cm2"], 2.0e6 * 1.380649e-16)
        self.assertTrue(out.loc[0, "P_thermal_SII_reliable"])
        self.assertEqual(out.loc[0, "P_thermal_SII_flag"], "ok")

    def test_thermal_pressure_propagates_density_mc_and_upper_limit_columns(self):
        df = pd.DataFrame(
            {
                "ne_SII_cm3": [np.nan],
                "ne_SII_cm3_mc": [80.0],
                "ne_SII_cm3_mc_minus": [20.0],
                "ne_SII_cm3_mc_plus": [40.0],
                "ne_SII_cm3_upper_limit": [150.0],
                "ne_SII_is_upper_limit": [True],
            }
        )

        out = add_thermal_pressure(df, T_e=8000.0, particle_factor=2.0)

        self.assertEqual(out.loc[0, "P_thermal_SII_over_k_K_cm3_mc"], 1.28e6)
        self.assertEqual(out.loc[0, "P_thermal_SII_over_k_K_cm3_mc_minus"], 3.2e5)
        self.assertEqual(out.loc[0, "P_thermal_SII_over_k_K_cm3_mc_plus"], 6.4e5)
        self.assertEqual(out.loc[0, "P_thermal_SII_over_k_K_cm3_upper_limit"], 2.4e6)
        self.assertAlmostEqual(out.loc[0, "P_thermal_SII_dyn_cm2_upper_limit"], 2.4e6 * 1.380649e-16)
        self.assertAlmostEqual(out.loc[0, "log_P_thermal_SII_over_k_upper_limit"], np.log10(2.4e6))
        self.assertTrue(out.loc[0, "P_thermal_SII_is_upper_limit"])

    def test_peak_region_properties_adds_halpha_luminosity_and_sii_density(self):
        df = pd.DataFrame(
            {
                "F_Halpha_peak": [2.0e-15],
                "F_Halpha_e_peak": [0.1e-15],
                "F_[SII]6716_peak": [1.3],
                "F_[SII]6716_e_peak": [0.013],
                "F_[SII]6731_peak": [1.0],
                "F_[SII]6731_e_peak": [0.01],
                "SNR_[SII]6716_peak": [100.0],
                "SNR_[SII]6731_peak": [100.0],
                "sum_E_BV": [0.0],
                "sum_E_BV_err": [0.0],
            }
        )

        out = add_peak_region_properties(df, distance_mpc=0.84, n_mc=0)

        self.assertEqual(out.loc[0, "F_Halpha_peak_dered"], 2.0e-15)
        self.assertGreater(out.loc[0, "L_Ha_peak"], 0.0)
        self.assertEqual(out.loc[0, "L_Ha_peak"], out.loc[0, "L_Ha_peak_dered"])
        self.assertAlmostEqual(out.loc[0, "SII_ratio_6716_6731_peak"], 1.3)
        self.assertGreater(out.loc[0, "ne_SII_peak_cm3"], 0.0)
        self.assertTrue(out.loc[0, "ne_SII_peak_reliable"])


if __name__ == "__main__":
    unittest.main()
