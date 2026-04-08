import unittest

import numpy as np

from m33_pipeline.photometry import integrated_flux_and_snr, region_edge_ring


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
        ring = region_edge_ring(region, iterations=1)
        stats = integrated_flux_and_snr(flux, err, region, ring_mask=ring)
        self.assertEqual(stats["F_raw"], 5.0)
        self.assertEqual(stats["sigma_F"], 1.0)
        self.assertEqual(stats["b_edge"], 1.0)
        self.assertEqual(stats["F_bgsub"], 4.0)
        self.assertEqual(stats["SNR_bgsub"], 4.0)


if __name__ == "__main__":
    unittest.main()
