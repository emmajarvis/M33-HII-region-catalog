import tempfile
import unittest
from pathlib import Path

import pandas as pd

from m33_pipeline.reporting import build_catalog_number_values, write_latex_commands


class ReportingTest(unittest.TestCase):
    def test_build_catalog_number_values(self):
        cat = pd.DataFrame(
            {
                "BPT_class_sum_dered": ["Star-forming", "Composite", "AGN/Shock"],
                "SNR_Halpha_sum": [4.0, 4.0, 4.0],
                "SNR_[OIII]5007_sum": [4.0, 4.0, 4.0],
                "SNR_[SII]6716_sum": [4.0, 4.0, 4.0],
                "SNR_[NII]6583_sum": [4.0, 4.0, 4.0],
                "SNR_Hbeta_sum": [4.0, 4.0, 4.0],
                "ne_SII_cm3": [100.0, 200.0, 300.0],
                "log_L_Ha_sum_dered": [36.0, 37.0, 38.0],
                "radius_p50_pc": [10.0, 11.0, 12.0],
                "radius_p16_pc": [8.0, 9.0, 10.0],
                "radius_p84_pc": [12.0, 13.0, 14.0],
                "radius_areaeq_pc": [9.0, 10.0, 11.0],
                "logU_KK04": [-3.1, -3.0, -2.9],
                "Z_N2S2Halpha_Brazzini2024": [8.3, 8.4, 8.5],
                "sum_A_V": [0.1, 0.2, 0.3],
                "DIG_fraction": [0.2, 0.3, 0.4],
                "distance_5th_closest_pc_deproj": [50.0, 60.0, 70.0],
                "L_Ha_sum_dered": [1e36, 2e36, 3e36],
                "sum_E_BV": [0.05, 0.06, 0.07],
                "nearest_neighbor_pc_deproj": [20.0, 25.0, 30.0],
                "sigma5_per_pc2_deproj": [1e-3, 2e-3, 3e-3],
            }
        )
        values, formats = build_catalog_number_values(cat)
        self.assertEqual(values["nregions"], 3)
        self.assertEqual(values["nstarforming"], 1)
        self.assertIn("medianlogLHa", values)
        self.assertIn("medianlogLHa", formats)

    def test_write_latex_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "numbers.tex"
            out = write_latex_commands(path, {"nregions": 3, "medianlogLHa": 37.0}, {"medianlogLHa": ".1f"})
            text = out.read_text()
            self.assertIn(r"\newcommand{\nregions}{3}", text)
            self.assertIn(r"\newcommand{\medianlogLHa}{37.0}", text)


if __name__ == "__main__":
    unittest.main()
