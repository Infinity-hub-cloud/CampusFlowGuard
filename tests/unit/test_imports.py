from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


class ImportTests(unittest.TestCase):
    def test_project_package_imports(self) -> None:
        package = importlib.import_module("campus_flow_guard")
        self.assertEqual(package.__version__, "0.1.0")

    def test_core_data_stack_imports(self) -> None:
        for module_name in ("numpy", "pandas", "sklearn", "pyarrow", "matplotlib", "yaml"):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    @unittest.skipUnless(
        _available("tensorflow") and _available("keras"),
        "TensorFlow/Keras are not installed; verify after creating the CPU environment.",
    )
    def test_ml_stack_imports(self) -> None:
        importlib.import_module("tensorflow")
        importlib.import_module("keras")

    @unittest.skipUnless(
        _available("fastapi") and _available("uvicorn"),
        "FastAPI/Uvicorn are not installed; verify after creating the CPU environment.",
    )
    def test_api_stack_imports(self) -> None:
        importlib.import_module("fastapi")
        importlib.import_module("uvicorn")

    @unittest.skipUnless(
        _available("pytest"),
        "pytest is not installed; unittest remains available for the current smoke test.",
    )
    def test_pytest_imports(self) -> None:
        importlib.import_module("pytest")


if __name__ == "__main__":
    unittest.main()
