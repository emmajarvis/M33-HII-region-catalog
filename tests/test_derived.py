import unittest

import pandas as pd

from m33_pipeline.derived import add_metallicity_error_columns, add_symmetry_class, merge_field_flux_catalogs


class DerivedTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
