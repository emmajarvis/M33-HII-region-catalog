import unittest

import numpy as np
import pandas as pd

from m33_pipeline.config import PhotometryConfig
from m33_pipeline.photometry import (
    _calibrate_integrated_flux_df,
    _dig_background_for_line,
    _mask_negative_fluxes,
    add_peak_pixel_fluxes,
    integrated_flux_and_snr,
    robust_dig_background,
    region_edge_ring,
)


class PhotometryTest(unittest.TestCase):
    def test_region_edge_ring_single_pixel(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        ring = region_edge_ring(mask, iterations=1)
        self.assertEqual(int(ring.sum()), 4)
        self.assertFalse(bool(ring[1, 1]))

    def test_integrated_flux_and_snr_background(self):
        flux = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 5.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        err = np.ones_like(flux)
        region = np.zeros_like(flux, dtype=bool)
        region[1, 1] = True
        stats = integrated_flux_and_snr(flux, err, region, background_per_pixel=1.0)
        self.assertEqual(stats["F_raw"], 5.0)
        self.assertEqual(stats["sigma_F"], 1.0)
        self.assertEqual(stats["background_per_pixel"], 1.0)
        self.assertEqual(stats["F_bgsub"], 4.0)
        self.assertEqual(stats["SNR_bgsub"], 4.0)

    def test_calibrate_integrated_flux_df_scales_flux_and_error_by_channel(self):
        df = pd.DataFrame(
            {
                "id": [0],
                "F_[OII]3727": [6.845],
                "F_[OII]3727_e": [0.6845],
                "SNR_[OII]3727": [10.0],
                "F_Hbeta": [10.958],
                "F_Hbeta_e": [1.0958],
                "SNR_Hbeta": [10.0],
                "F_[OIII]5007": [21.916],
                "F_[OIII]5007_e": [2.1916],
                "SNR_[OIII]5007": [10.0],
                "F_Halpha": [10.966],
                "F_Halpha_e": [1.0966],
                "SNR_Halpha": [10.0],
                "F_[SII]6716": [5.483],
                "F_[SII]6716_e": [0.5483],
                "SNR_[SII]6716": [10.0],
            }
        )

        calibrated = _calibrate_integrated_flux_df(df, "NW")

        self.assertAlmostEqual(calibrated.loc[0, "F_[OII]3727"], 10.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_[OII]3727_e"], 1.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_Hbeta"], 10.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_Hbeta_e"], 1.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_[OIII]5007"], 20.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_[OIII]5007_e"], 2.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_Halpha"], 10.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_Halpha_e"], 1.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_[SII]6716"], 5.0)
        self.assertAlmostEqual(calibrated.loc[0, "F_[SII]6716_e"], 0.5)
        self.assertAlmostEqual(calibrated.loc[0, "SNR_Halpha"], 10.0)

    def test_robust_dig_background_uses_conservative_percentile(self):
        flux = np.array([[1.0, 2.0, 3.0, 100.0]])
        annulus_mask = np.array([[True, True, True, True]])

        dig = robust_dig_background(
            flux,
            annulus_mask,
            clip_sigma=3.0,
            clip_iterations=1,
            background_percentile=25.0,
        )

        self.assertAlmostEqual(dig["dig_median"], 1.5)
        self.assertEqual(dig["n_annulus"], 4)

    def test_robust_dig_background_never_returns_negative_threshold(self):
        flux = np.array([[-5.0, -2.0, -1.0, 0.0]])
        annulus_mask = np.array([[True, True, True, True]])

        dig = robust_dig_background(
            flux,
            annulus_mask,
            clip_sigma=3.0,
            clip_iterations=1,
            background_percentile=10.0,
        )

        self.assertEqual(dig["dig_median"], 0.0)

    def test_dig_background_for_line_scales_from_halpha_anchor(self):
        config = PhotometryConfig()
        self.assertAlmostEqual(_dig_background_for_line(10.0, "Halpha", config), 10.0)
        self.assertAlmostEqual(_dig_background_for_line(10.0, "Hbeta", config), 10.0 / 3.1)
        self.assertAlmostEqual(_dig_background_for_line(10.0, "[NII]6583", config), 2.1)

    def test_mask_negative_fluxes_sets_flux_error_and_snr_to_nan(self):
        df = pd.DataFrame(
            {
                "F_Hbeta_sum": [-1.0, 2.0],
                "F_Hbeta_e_sum": [0.1, 0.2],
                "SNR_Hbeta_sum": [-10.0, 10.0],
                "F_Hbeta_sum_nodig": [1.0, -2.0],
                "F_Hbeta_e_sum_nodig": [0.1, 0.2],
                "SNR_Hbeta_sum_nodig": [10.0, -10.0],
            }
        )

        masked = _mask_negative_fluxes(df, ["Hbeta"], prefixes=["sum", "sum_nodig"])

        self.assertTrue(np.isnan(masked.loc[0, "F_Hbeta_sum"]))
        self.assertTrue(np.isnan(masked.loc[0, "F_Hbeta_e_sum"]))
        self.assertTrue(np.isnan(masked.loc[0, "SNR_Hbeta_sum"]))
        self.assertTrue(np.isnan(masked.loc[1, "F_Hbeta_sum_nodig"]))
        self.assertTrue(np.isnan(masked.loc[1, "F_Hbeta_e_sum_nodig"]))
        self.assertTrue(np.isnan(masked.loc[1, "SNR_Hbeta_sum_nodig"]))

    def test_add_peak_pixel_fluxes_samples_xy_and_applies_active_dig_mode(self):
        df = pd.DataFrame(
            {
                "x": [2, 0],
                "y": [1, 2],
                "Halpha_dig_median": [1.5, 0.5],
            }
        )
        flux = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 10.0], [7.0, 8.0, 9.0]])
        error = np.ones_like(flux) * 0.5

        out = add_peak_pixel_fluxes(df, {"Halpha": (flux, error)}, dig_mode="dig_subtracted")

        self.assertEqual(out.loc[0, "F_Halpha_peak_nodig"], 10.0)
        self.assertEqual(out.loc[0, "F_Halpha_peak_digsub"], 8.5)
        self.assertEqual(out.loc[0, "F_Halpha_peak"], 8.5)
        self.assertEqual(out.loc[0, "F_Halpha_e_peak"], 0.5)
        self.assertEqual(out.loc[0, "SNR_Halpha_peak"], 17.0)
        self.assertTrue(out.loc[0, "peak_pixel_valid"])


if __name__ == "__main__":
    unittest.main()
