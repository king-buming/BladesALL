"""Regression: a checkout must resolve its own modules without an installed backend."""

import importlib
import importlib.abc
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RejectExternalBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "fedlib":
            raise AssertionError(f"External backend import attempted: {fullname}")
        return None


class UnifiedImportTests(unittest.TestCase):
    def test_all_module_families_resolve_inside_checkout(self):
        # A missing migration or residual backend dependency must fail here.
        self.assertTrue((ROOT / "blades" / "__init__.py").is_file())
        guard = RejectExternalBackend()
        sys.meta_path.insert(0, guard)
        try:
            for name in ("adversaries", "aggregators", "datasets", "models", "trainers", "clients"):
                with self.subTest(module=name):
                    module = importlib.import_module(f"blades.{name}")
                    self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT))
        finally:
            sys.meta_path.remove(guard)


if __name__ == "__main__":
    unittest.main()
