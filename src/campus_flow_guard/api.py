"""Minimal local FastAPI wrapper around the existing inference module."""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from campus_flow_guard.inference import run as run_inference


InferenceRunner = Callable[[Path, Path], dict[str, Any]]


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"API configuration root must be a mapping: {path}")
    return loaded


def _api_response(inference_document: dict[str, Any], sample_limit: int) -> dict[str, Any]:
    summary = inference_document["detection_summary"]
    return {
        "total_flows": int(summary["input_row_count"]),
        "evaluated_windows": int(summary["window_count"]),
        "attack_count": int(summary["detected_attack_count"]),
        "risk_level_counts": dict(summary["risk_level_counts"]),
        "sample_predictions": list(inference_document["detections"][:sample_limit]),
    }


def create_app(
    api_config_path: Path,
    *,
    inference_runner: InferenceRunner = run_inference,
) -> FastAPI:
    api_config_path = api_config_path.expanduser().resolve()
    project_root = api_config_path.parents[2]
    config = _read_yaml(api_config_path)
    inference_config_path = (
        project_root / str(config["inference"]["config_path"])
    ).resolve()
    maximum_bytes = int(float(config["upload"]["maximum_size_mib"]) * 1024 * 1024)
    sample_limit = int(config["upload"]["sample_prediction_limit"])
    allowed_suffixes = {str(value).lower() for value in config["upload"]["allowed_suffixes"]}
    if maximum_bytes <= 0 or sample_limit <= 0:
        raise ValueError("Upload size and sample prediction limit must be positive")

    app = FastAPI(
        title="CampusFlowGuard Local Demo",
        version="0.1.0",
        description="Local defensive CSV inference only; no external API calls.",
    )
    html_path = Path(__file__).resolve().parent / "static" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return html_path.read_text(encoding="utf-8")

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = file.filename or ""
        if Path(filename).suffix.lower() not in allowed_suffixes:
            raise HTTPException(status_code=415, detail="Only configured CSV file suffixes are accepted")
        content = await file.read(maximum_bytes + 1)
        await file.close()
        if len(content) > maximum_bytes:
            raise HTTPException(status_code=413, detail="Uploaded CSV exceeds the configured size limit")
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary_file:
                temporary_file.write(content)
                temporary_path = Path(temporary_file.name)
            inference_document = await run_in_threadpool(
                inference_runner,
                inference_config_path,
                temporary_path,
            )
            return _api_response(inference_document, sample_limit)
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local CampusFlowGuard FastAPI demo")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    try:
        import uvicorn

        config_path = args.config.expanduser().resolve()
        config = _read_yaml(config_path)
        host = args.host or str(config["run"]["host"])
        port = args.port or int(config["run"]["port"])
        uvicorn.run(create_app(config_path), host=host, port=port, log_level="info")
    except Exception as exc:
        print(f"本地展示接口启动失败: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
