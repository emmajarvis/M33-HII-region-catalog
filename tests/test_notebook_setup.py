import os
import tempfile
import unittest
from pathlib import Path

from m33_pipeline.notebook_setup import prepare_notebook, repo_root_from_notebook
from m33_pipeline.paths import repo_root


class NotebookSetupTest(unittest.TestCase):
    def test_repo_root_from_nested_stage_dir(self):
        nested = repo_root() / "01_region_identification"
        self.assertEqual(repo_root_from_notebook(nested), repo_root())

    def test_prepare_notebook_changes_cwd_to_repo_root(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(dir=repo_root()) as tmpdir:
            os.chdir(Path(tmpdir))
            try:
                detected = prepare_notebook()
                self.assertEqual(detected, repo_root())
                self.assertEqual(Path.cwd(), repo_root())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
