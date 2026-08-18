import unittest

import numpy as np
import pandas as pd

from m33_pipeline.config import PhotometryConfig
from m33_pipeline.photometry import (
    _calibrate_integrated_flux_df,
    _apply_active_flux_columns,
    _compute_dig_background_catalog,
    _select_capped_dig_background,
    _select_nonnegative_dig_background,
    _scaled_local_background_mask,
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

    def test_scaled_local_background_mask_excludes_all_boundaries(self):
        boundary = np.zeros((7, 7), dtype=float)
        boundary[3, 3] = 1
        boundary[3, 4] = 2
        zoi = np.ones_like(boundary)

        mask, info = _scaled_local_background_mask(
            zoi,
            boundary,
            1,
            PhotometryConfig(dig_annulus_inner_px=0, dig_annulus_min_width_px=1, dig_annulus_max_width_px=1),
        )

        self.assertEqual(info["dig_annulus_source"], "local_zoi_annulus")
        self.assertFalse(bool(mask[3, 3]))
        self.assertFalse(bool(mask[3, 4]))
        self.assertGreater(int(mask.sum()), 0)

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

    def test_dig_subtracted_active_snr_uses_observed_line_snr(self):
        df = pd.DataFrame(
            {
                "F_Halpha_sum_nodig": [12.0],
                "F_Halpha_e_sum_nodig": [2.0],
                "SNR_Halpha_sum_nodig": [6.0],
                "F_Halpha_sum_digsub": [4.0],
            }
        )

        out = _apply_active_flux_columns(df, ["Halpha"], dig_mode="dig_subtracted")

        self.assertEqual(out.loc[0, "F_Halpha_sum"], 4.0)
        self.assertEqual(out.loc[0, "F_Halpha_e_sum"], 2.0)
        self.assertEqual(out.loc[0, "SNR_Halpha_sum"], 6.0)

    def test_dig_background_falls_back_to_local_then_no_subtraction(self):
        bg, flux, method = _select_nonnegative_dig_background(
            raw_flux=10.0,
            npix_region=5,
            local_background=1.0,
        )
        self.assertEqual(bg, 1.0)
        self.assertEqual(flux, 5.0)
        self.assertEqual(method, "local_line")

        bg, flux, method = _select_nonnegative_dig_background(
            raw_flux=10.0,
            npix_region=5,
            local_background=2.5,
        )
        self.assertEqual(bg, 0.0)
        self.assertEqual(flux, 10.0)
        self.assertEqual(method, "none_negative_guard")

    def test_capped_dig_background_limits_subtracted_fraction(self):
        bg, flux, fraction, method = _select_capped_dig_background(
            raw_flux=10.0,
            npix_region=5,
            model_background_flux=8.0,
            max_subtraction_fraction=0.3,
        )

        self.assertEqual(bg, 0.6)
        self.assertEqual(flux, 7.0)
        self.assertEqual(fraction, 0.3)
        self.assertEqual(method, "smoothed_map_capped")

    def test_dig_catalog_does_not_write_digsub_error_or_snr_columns(self):
        boundary = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float)
        zoi = np.ones_like(boundary)
        flux = np.array([[2.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        err = np.ones_like(flux) * 0.5

        out = _compute_dig_background_catalog(
            boundary,
            zoi,
            {"Halpha": (flux, err)},
            PhotometryConfig(dig_background_percentile=50.0),
        )

        self.assertIn("F_Halpha_sum_digsub", out.columns)
        self.assertNotIn("F_Halpha_e_sum_digsub", out.columns)
        self.assertNotIn("SNR_Halpha_sum_digsub", out.columns)
        self.assertGreaterEqual(out.loc[0, "F_Halpha_sum_digsub"], 0.0)
        self.assertLessEqual(out.loc[0, "Halpha_dig_fraction_raw"], 0.5)

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
        self.assertNotIn("F_Halpha_e_peak_digsub", out.columns)
        self.assertEqual(out.loc[0, "SNR_Halpha_peak"], 20.0)
        self.assertTrue(out.loc[0, "peak_pixel_valid"])


if __name__ == "__main__":
    unittest.main()
