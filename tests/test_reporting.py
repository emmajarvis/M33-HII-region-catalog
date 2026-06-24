import tempfile
import unittest
from pathlib import Path

import pandas as pd

from m33_pipeline.reporting import (
    add_fit_values,
    add_region_removal_values,
    build_catalog_number_values,
    format_value,
    latex_command_name,
    write_latex_commands,
)


class ReportingTest(unittest.TestCase):
    def test_build_catalog_number_values(self):
        cat = pd.DataFrame(
            {
                "BPT_class_sum_dered": ["Star-forming", "Composite", "AGN/Shock", "Unclassified"],
                "SNR_Halpha_sum": [4.0, 4.0, 2.0, 4.0],
                "SNR_[OIII]5007_sum": [4.0, 4.0, 4.0, 4.0],
                "SNR_[SII]6716_sum": [4.0, 4.0, 4.0, 4.0],
                "SNR_[NII]6583_sum": [4.0, 4.0, 4.0, 4.0],
                "SNR_Hbeta_sum": [4.0, 4.0, 4.0, 4.0],
                "ne_SII_cm3": [100.0, 200.0, 300.0, 400.0],
                "log_L_Ha_sum_dered": [36.0, 37.0, 38.0, 39.0],
                "radius_p50_pc": [10.0, 11.0, 12.0, 13.0],
                "radius_p16_pc": [8.0, 9.0, 10.0, 11.0],
                "radius_p84_pc": [12.0, 13.0, 14.0, 15.0],
                "radius_areaeq_pc": [9.0, 10.0, 11.0, 12.0],
                "logU_KK04": [-3.1, -3.0, -2.9, -2.8],
                "Z_N2S2Halpha_Brazzini2024": [8.3, 8.4, 8.5, 8.6],
                "sum_A_V": [0.1, 0.2, 0.3, 0.4],
                "DIG_fraction": [0.2, 0.3, 0.4, 0.5],
                "distance_5th_closest_pc_deproj": [50.0, 60.0, 70.0, 80.0],
                "L_Ha_sum_dered": [1e36, 2e36, 3e36, 4e36],
                "sum_E_BV": [0.05, 0.06, 0.07, 0.08],
                "nearest_neighbor_pc_deproj": [20.0, 25.0, 30.0, 35.0],
                "sigma5_per_pc2_deproj": [1e-3, 2e-3, 3e-3, 4e-3],
                "has_snr_in_boundary": [True, False, True, False],
                "has_wr_in_boundary": [False, True, True, False],
                "has_pn_in_boundary": [True, False, True, False],
            }
        )
        values, formats = build_catalog_number_values(cat)
        self.assertEqual(values["nregions"], 4)
        self.assertEqual(values["nstarforming"], 1)
        self.assertEqual(values["nUnclassified"], 1)
        self.assertEqual(values["nregionssnrthree"], 3)
        self.assertEqual(values["nStarformingSNRThree"], 1)
        self.assertEqual(values["nCompositeSNRThree"], 1)
        self.assertEqual(values["nAgnSNRThree"], 0)
        self.assertIn("medianlogLHa", values)
        self.assertIn("medianlogLHa", formats)
        self.assertEqual(values["nRegionsContainingSNR"], 2)
        self.assertEqual(values["nRegionsContainingWR"], 2)
        self.assertEqual(values["nRegionsContainingSNRAndWR"], 1)
        self.assertEqual(values["nRegionsContainingSNROrWR"], 3)
        self.assertEqual(values["nRegionsContainingPN"], 2)

    def test_write_latex_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "numbers.tex"
            out = write_latex_commands(path, {"nregions": 3, "medianlogLHa": 37.0}, {"medianlogLHa": ".1f"})
            text = out.read_text()
            self.assertIn(r"\newcommand{\nregions}{3}", text)
            self.assertIn(r"\newcommand{\medianlogLHa}{37.0}", text)

    def test_latex_commands_spell_out_digits_and_format_scientific_notation(self):
        self.assertEqual(latex_command_name("resolutionD0p84"), "resolutionDZeropEightFour")
        self.assertEqual(format_value(1.2e36, ".2e"), r"$1.20\times10^{36}$")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "numbers.tex"
            write_latex_commands(path, {"value2": 1.2e36}, {"value2": ".2e"})
            text = path.read_text()
            self.assertIn(r"\newcommand{\valueTwo}{$1.20\times10^{36}$}", text)

    def test_add_fit_values_adds_slope_intercept_and_errors(self):
        values, formats = {}, {}
        add_fit_values(
            values,
            formats,
            {
                "slope": -0.1,
                "intercept": 8.5,
                "slope_stderr": 0.02,
                "intercept_stderr": 0.1,
                "n_points": 50,
            },
            "radialMetallicity",
        )
        self.assertEqual(values["radialMetallicitySlope"], -0.1)
        self.assertEqual(values["radialMetallicityInterceptErr"], 0.1)
        self.assertEqual(values["radialMetallicityN"], 50)

    def test_add_region_removal_values_aggregates_available_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for field, input_peaks, saddle_removed, small_removed, final_regions in [
                ("NW", 100, 10, 5, 80),
                ("SE", 50, 2, 3, 40),
            ]:
                Path(tmpdir, f"{field}_region_summary.json").write_text(
                    (
                        "{"
                        f'"input_peaks": {input_peaks}, '
                        f'"saddle_removed": {saddle_removed}, '
                        '"edge_removed": 0, '
                        f'"candidate_peaks": {input_peaks - saddle_removed}, '
                        f'"zoi_small_removed": {small_removed}, '
                        '"zoi_duplicate_removed": 1, '
                        f'"final_peaks_after_zoi": {input_peaks - saddle_removed - small_removed - 1}, '
                        '"boundary_small_removed": 2, '
                        f'"final_regions_after_boundary": {final_regions}'
                        "}"
                    ),
                    encoding="utf-8",
                )
            values = {}
            add_region_removal_values(values, tmpdir)
            self.assertEqual(values["nRegionSummaryFields"], 2)
            self.assertEqual(values["nPeaksInput"], 150)
            self.assertEqual(values["nPeaksRemovedSmallZoI"], 8)
            self.assertEqual(values["nRegionsAfterSmallBoundary"], 120)


if __name__ == "__main__":
    unittest.main()
