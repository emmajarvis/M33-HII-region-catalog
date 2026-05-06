import unittest

from m33_pipeline.resolution_degradation import (
    build_resolution_paths,
    effective_zoi_radius_for_map,
    field_distance_tag,
    zoi_max_radius_pc_for_distance,
)


class ResolutionDegradationTest(unittest.TestCase):
    def test_zoi_radius_rule(self):
        self.assertEqual(zoi_max_radius_pc_for_distance(50.0), 100.0)
        self.assertEqual(zoi_max_radius_pc_for_distance(100.0), 100.0)

    def test_effective_zoi_radius_has_pixel_floor(self):
        zoi_pc, zoi_px = effective_zoi_radius_for_map(100.0, pix_pc=155.0)
        self.assertEqual(zoi_px, 4)
        self.assertGreaterEqual(zoi_pc, 4 * 155.0)

    def test_field_distance_tag(self):
        self.assertEqual(field_distance_tag("NW", 20.0), "NW_20Mpc")
        self.assertEqual(field_distance_tag("NW", 0.84), "NW_0.84Mpc")

    def test_resolution_paths_root(self):
        stage_paths = build_resolution_paths()
        self.assertEqual(stage_paths.root.name, "04_resolution_degradation")
        self.assertEqual(stage_paths.catalogs_dir.parent, stage_paths.root)


if __name__ == "__main__":
    unittest.main()
