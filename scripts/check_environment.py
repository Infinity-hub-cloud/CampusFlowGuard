from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_MODULES = {
    "tensorflow": "tensorflow",
    "keras": "keras",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "PyYAML": "yaml",
}

ORIGINAL_MODULES = (
    "framework",
    "framework.dataset_specification",
    "framework.flow_transformer_parameters",
    "implementations.pre_processings",
    "framework.flow_transformer",
    "implementations.input_encodings",
    "implementations.classification_heads",
    "implementations.transformers.basic_transformers",
    "implementations.transformers.named_transformers",
    "framework.sequential_input_encoding",
)


def _run_command(executable: str | Path, *args: str) -> dict[str, Any]:
    command = [str(executable), *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _windows_version() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "release": platform.release(),
        "version": platform.version(),
    }
    if os.name != "nt":
        return result

    try:
        import winreg

        key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        names = ("ProductName", "DisplayVersion", "CurrentBuild", "UBR", "EditionID")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            result["registry"] = {name: winreg.QueryValueEx(key, name)[0] for name in names}
    except (OSError, ImportError) as exc:
        result["registry_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _package_status(distribution: str, module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "distribution": distribution,
        "module": module_name,
        "distribution_version": None,
        "import_ok": False,
        "module_version": None,
        "error": None,
    }
    try:
        result["distribution_version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        result["metadata_error"] = f"{type(exc).__name__}: {exc}"

    try:
        module = importlib.import_module(module_name)
        result["import_ok"] = True
        result["module_version"] = getattr(module, "__version__", None)
    except Exception as exc:  # Diagnostic script must preserve third-party import errors.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _find_nvidia_smi() -> Path | None:
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return Path(discovered)
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
        candidate = system_root / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            return candidate
    return None


def _gpu_status() -> dict[str, Any]:
    nvidia_smi = _find_nvidia_smi()
    nvcc = shutil.which("nvcc")
    result: dict[str, Any] = {
        "nvidia_smi": str(nvidia_smi) if nvidia_smi else None,
        "nvcc": nvcc,
    }
    if nvidia_smi:
        result["query"] = _run_command(
            nvidia_smi,
            "--query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id,compute_cap",
            "--format=csv,noheader,nounits",
        )
    if nvcc:
        result["nvcc_version"] = _run_command(nvcc, "--version")
    return result


def _tensorflow_devices(package_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not package_results["tensorflow"]["import_ok"]:
        return {"available": False, "reason": package_results["tensorflow"]["error"]}
    try:
        tensorflow = importlib.import_module("tensorflow")
        return {
            "available": True,
            "version": tensorflow.__version__,
            "built_with_cuda": tensorflow.test.is_built_with_cuda(),
            "built_with_gpu_support": tensorflow.test.is_built_with_gpu_support(),
            "physical_devices": [
                {"name": device.name, "type": device.device_type}
                for device in tensorflow.config.list_physical_devices()
            ],
            "gpu_devices": [
                device.name for device in tensorflow.config.list_physical_devices("GPU")
            ],
            "build_info": tensorflow.sysconfig.get_build_info(),
        }
    except Exception as exc:  # Keep the exact TensorFlow diagnostic error.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _original_imports(original_repo: Path | None) -> dict[str, Any]:
    if original_repo is None:
        return {"checked": False, "reason": "No original repository path was provided."}
    if not original_repo.is_dir():
        return {"checked": False, "error": f"Directory not found: {original_repo}"}

    results: dict[str, Any] = {}
    repo_text = str(original_repo.resolve())
    sys.path.insert(0, repo_text)
    try:
        for module_name in ORIGINAL_MODULES:
            try:
                importlib.import_module(module_name)
                results[module_name] = {"import_ok": True, "error": None}
            except Exception as exc:  # Preserve upstream errors without failing diagnosis.
                results[module_name] = {
                    "import_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        try:
            framework = importlib.import_module("framework")
            exports = (
                "NamedDatasetSpecifications",
                "EvaluationDatasetSampling",
                "FlowTransformer",
                "FlowTransformerParameters",
            )
            missing = [name for name in exports if not hasattr(framework, name)]
            results["framework_demo_exports"] = {
                "import_ok": not missing,
                "missing": missing,
                "error": None if not missing else f"Missing exports: {', '.join(missing)}",
            }
        except Exception as exc:
            results["framework_demo_exports"] = {
                "import_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    finally:
        if sys.path and sys.path[0] == repo_text:
            sys.path.pop(0)

    return {
        "checked": True,
        "path": repo_text,
        "main_py_checked": False,
        "main_py_reason": "Import has dataset-loading and training side effects.",
        "modules": results,
    }


def collect_diagnostics(original_repo: Path | None) -> dict[str, Any]:
    package_results = {
        distribution: _package_status(distribution, module_name)
        for distribution, module_name in PACKAGE_MODULES.items()
    }
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "windows": _windows_version(),
        "tools": {
            "pip": _run_command(sys.executable, "-m", "pip", "--version"),
            "pip_check": _run_command(sys.executable, "-m", "pip", "check"),
            "conda": _run_command(shutil.which("conda") or "conda", "--version"),
            "git": _run_command(shutil.which("git") or "git", "--version"),
        },
        "gpu": _gpu_status(),
        "packages": package_results,
        "tensorflow_devices": _tensorflow_devices(package_results),
        "original_repository_imports": _original_imports(original_repo),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the CampusFlowGuard runtime environment.")
    parser.add_argument(
        "--original-repo",
        type=Path,
        default=Path(os.environ["FLOWTRANSFORMER_REPO"])
        if "FLOWTRANSFORMER_REPO" in os.environ
        else None,
        help="Optional read-only FlowTransformer repository path.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when a declared package cannot be imported.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = collect_diagnostics(args.original_repo)
    rendered = json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str)
    print(rendered)

    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    if args.strict:
        failed = [
            name for name, status in diagnostics["packages"].items() if not status["import_ok"]
        ]
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
