import unittest

from m33_pipeline import paths


class PathsTest(unittest.TestCase):
    def test_flux_catalog_csv_path(self):
        path = paths.flux_catalog_csv("F7")
        self.assertEqual(path.name, "flux_catalog_F7.csv")
        self.assertEqual(path.parent.name, "flux_catalogs")

    def test_boundary_metrics_path(self):
        path = paths.boundary_metrics_csv("NE", 100)
        self.assertEqual(path.name, "Boundary_metrics_NE.csv")
        self.assertEqual(path.parent.name, "Boundary_map_100pc")


if __name__ == "__main__":
    unittest.main()
