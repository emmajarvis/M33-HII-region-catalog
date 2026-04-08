import unittest

import pandas as pd

from m33_pipeline.validate import validate_field_flux_catalog


class ValidateTest(unittest.TestCase):
    def test_validate_field_flux_catalog_missing_columns(self):
        df = pd.DataFrame({"region_id": [1], "field": ["F7"]})
        warnings = validate_field_flux_catalog(df, "F7")
        self.assertTrue(any("missing columns" in warning for warning in warnings))

    def test_validate_field_flux_catalog_duplicate_region_id(self):
        df = pd.DataFrame(
            {
                "region_id": [1, 1],
                "x": [0, 1],
                "y": [0, 1],
                "field": ["F7", "F7"],
                "F_Halpha_sum": [1.0, 2.0],
                "F_Halpha_e_sum": [0.1, 0.1],
                "SNR_Halpha_sum": [10.0, 20.0],
            }
        )
        warnings = validate_field_flux_catalog(df, "F7")
        self.assertTrue(any("duplicate values" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
