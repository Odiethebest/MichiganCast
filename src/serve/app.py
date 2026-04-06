from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.serve.monitoring import InferenceMonitor, MonitoringConfig

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]

    class BaseModel:  # type: ignore[no-redef]
        pass

    _FASTAPI_AVAILABLE = False


class PredictRequest(BaseModel):
    image_sequence: list
    meteo_sequence: list
    request_id: Optional[str] = None


class PredictResponse(BaseModel):
    request_id: str
    model_path: str
    latency_ms: float
    rain_probability: list[float]
    batch_size: int


@dataclass
class RuntimeState:
    model_path: str
    device: torch.device
    model: torch.jit.ScriptModule
    monitor: InferenceMonitor
    expected_image_steps: int | None = None
    expected_meteo_steps: int | None = None
    expected_meteo_features: int | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS is unavailable.")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_inputs(
    req: PredictRequest,
    *,
    expected_image_steps: int | None = None,
    expected_meteo_steps: int | None = None,
    expected_meteo_features: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(req.image_sequence, dtype=np.float32)
    meteo = np.asarray(req.meteo_sequence, dtype=np.float32)

    if image.ndim == 4:
        image = np.expand_dims(image, axis=0)
    if meteo.ndim == 2:
        meteo = np.expand_dims(meteo, axis=0)

    if image.ndim != 5:
        raise ValueError("image_sequence must have shape [B,T,1,H,W] or [T,1,H,W].")
    if meteo.ndim != 3:
        raise ValueError("meteo_sequence must have shape [B,T,F] or [T,F].")
    if image.shape[0] != meteo.shape[0]:
        raise ValueError("image_sequence batch size must match meteo_sequence batch size.")
    if expected_image_steps is not None and image.shape[1] != expected_image_steps:
        raise ValueError(f"image_sequence steps must equal {expected_image_steps}, got {image.shape[1]}.")
    if expected_meteo_steps is not None and meteo.shape[1] != expected_meteo_steps:
        raise ValueError(f"meteo_sequence steps must equal {expected_meteo_steps}, got {meteo.shape[1]}.")
    if expected_meteo_features is not None and meteo.shape[2] != expected_meteo_features:
        raise ValueError(f"meteo feature count must equal {expected_meteo_features}, got {meteo.shape[2]}.")
    return image, meteo


def _load_runtime(model_path: str, device: str, monitor_log: str) -> RuntimeState:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"TorchScript model not found: {model_path}")
    target_device = _resolve_device(device)
    model = torch.jit.load(str(path), map_location=target_device)
    model.eval()
    monitor = InferenceMonitor(MonitoringConfig(log_jsonl_path=monitor_log))
    expected_image_steps = None
    expected_meteo_steps = None
    expected_meteo_features = None

    metadata_path = path.parent / f"{path.stem}_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape = payload.get("example_input_shape", {})
        image_shape = shape.get("image_sequence", [])
        meteo_shape = shape.get("meteo_sequence", [])
        if len(image_shape) >= 3:
            expected_image_steps = int(image_shape[1])
        if len(meteo_shape) >= 3:
            expected_meteo_steps = int(meteo_shape[1])
            expected_meteo_features = int(meteo_shape[2])

    return RuntimeState(
        model_path=str(path),
        device=target_device,
        model=model,
        monitor=monitor,
        expected_image_steps=expected_image_steps,
        expected_meteo_steps=expected_meteo_steps,
        expected_meteo_features=expected_meteo_features,
    )


def create_app(
    *,
    model_path: str = "artifacts/models/michigancast_multimodal.ts",
    device: str = "cpu",
    monitor_log: str = "artifacts/logs/inference_monitoring.jsonl",
) -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed. Install with: conda install -n pytorch_env fastapi uvicorn")

    app = FastAPI(title="MichiganCast Inference API", version="0.1.0")
    runtime = _load_runtime(model_path=model_path, device=device, monitor_log=monitor_log)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_path": runtime.model_path,
            "device": str(runtime.device),
        }

    @app.get("/metrics/summary")
    def metrics_summary() -> dict:
        return runtime.monitor.summary()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):  # type: ignore[valid-type]
        request_id = req.request_id or uuid.uuid4().hex
        try:
            image_np, meteo_np = _prepare_inputs(
                req,
                expected_image_steps=runtime.expected_image_steps,
                expected_meteo_steps=runtime.expected_meteo_steps,
                expected_meteo_features=runtime.expected_meteo_features,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            start = time.perf_counter()
            with torch.no_grad():
                image_tensor = torch.tensor(image_np, dtype=torch.float32, device=runtime.device)
                meteo_tensor = torch.tensor(meteo_np, dtype=torch.float32, device=runtime.device)
                logits = runtime.model(image_tensor, meteo_tensor)
                probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference_failed: {exc}")

        runtime.monitor.record(
            request_id=request_id,
            image_array=image_np,
            meteo_array=meteo_np,
            prediction_scores=probs,
            latency_ms=latency_ms,
            model_path=runtime.model_path,
        )

        return PredictResponse(
            request_id=request_id,
            model_path=runtime.model_path,
            latency_ms=latency_ms,
            rain_probability=probs.flatten().tolist(),
            batch_size=int(probs.shape[0] if probs.ndim > 0 else 1),
        )

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MichiganCast FastAPI inference service (T35/T36)")
    parser.add_argument("--model-path", default="artifacts/models/michigancast_multimodal.ts")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--monitor-log", default="artifacts/logs/inference_monitoring.jsonl")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed. Install with: conda install -n pytorch_env fastapi uvicorn")

    args = _build_arg_parser().parse_args()
    app = create_app(model_path=args.model_path, device=args.device, monitor_log=args.monitor_log)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
