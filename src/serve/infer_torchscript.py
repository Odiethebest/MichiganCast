from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_IMAGE_STEPS = 16
DEFAULT_IMAGE_SIZE = 64
DEFAULT_METEO_STEPS = 48
DEFAULT_METEO_FEATURE_COUNT = 13


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


def _build_dummy_inputs(
    *,
    batch_size: int,
    image_steps: int,
    image_size: int,
    meteo_steps: int,
    meteo_feature_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.standard_normal(
        size=(batch_size, image_steps, 1, image_size, image_size),
        dtype=np.float32,
    )
    meteo = rng.standard_normal(
        size=(batch_size, meteo_steps, meteo_feature_count),
        dtype=np.float32,
    )
    return image, meteo


def run_inference(
    *,
    model_path: str,
    output_json: str,
    device: str,
    input_npz: str | None,
    batch_size: int,
    image_steps: int,
    image_size: int,
    meteo_steps: int,
    meteo_feature_count: int,
    seed: int,
    metadata_path: str | None,
) -> dict:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"TorchScript model not found: {model_path}")

    target_device = _resolve_device(device)
    model = torch.jit.load(str(model_file), map_location=target_device)
    model.eval()

    resolved_image_steps = image_steps
    resolved_image_size = image_size
    resolved_meteo_steps = meteo_steps
    resolved_meteo_features = meteo_feature_count
    resolved_metadata_path = metadata_path

    if resolved_metadata_path is None:
        model_stem = model_file.stem
        candidate = model_file.parent / f"{model_stem}_metadata.json"
        if candidate.exists():
            resolved_metadata_path = str(candidate)

    if resolved_metadata_path is not None and Path(resolved_metadata_path).exists():
        meta = json.loads(Path(resolved_metadata_path).read_text(encoding="utf-8"))
        shape = meta.get("example_input_shape", {})
        image_shape = shape.get("image_sequence", [])
        meteo_shape = shape.get("meteo_sequence", [])
        if resolved_image_steps <= 0 and len(image_shape) >= 3:
            resolved_image_steps = int(image_shape[1])
        if resolved_image_size <= 0 and len(image_shape) >= 5:
            resolved_image_size = int(image_shape[3])
        if resolved_meteo_steps <= 0 and len(meteo_shape) >= 2:
            resolved_meteo_steps = int(meteo_shape[1])
        if resolved_meteo_features <= 0 and len(meteo_shape) >= 3:
            resolved_meteo_features = int(meteo_shape[2])

    if resolved_image_steps <= 0:
        resolved_image_steps = DEFAULT_IMAGE_STEPS
    if resolved_image_size <= 0:
        resolved_image_size = DEFAULT_IMAGE_SIZE
    if resolved_meteo_steps <= 0:
        resolved_meteo_steps = DEFAULT_METEO_STEPS
    if resolved_meteo_features <= 0:
        resolved_meteo_features = DEFAULT_METEO_FEATURE_COUNT

    if input_npz is not None:
        payload = np.load(input_npz)
        image_np = payload["image_sequence"].astype(np.float32)
        meteo_np = payload["meteo_sequence"].astype(np.float32)
    else:
        image_np, meteo_np = _build_dummy_inputs(
            batch_size=batch_size,
            image_steps=resolved_image_steps,
            image_size=resolved_image_size,
            meteo_steps=resolved_meteo_steps,
            meteo_feature_count=resolved_meteo_features,
            seed=seed,
        )

    image_tensor = torch.tensor(image_np, dtype=torch.float32, device=target_device)
    meteo_tensor = torch.tensor(meteo_np, dtype=torch.float32, device=target_device)

    with torch.no_grad():
        logits = model(image_tensor, meteo_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()

    result = {
        "model_path": str(model_file),
        "metadata_path": resolved_metadata_path,
        "device": str(target_device),
        "batch_size": int(image_tensor.shape[0]),
        "image_sequence_shape": list(image_tensor.shape),
        "meteo_sequence_shape": list(meteo_tensor.shape),
        "rain_probability": probs,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent TorchScript inference runner (T25)")
    parser.add_argument("--model-path", default="artifacts/models/michigancast_multimodal.ts")
    parser.add_argument("--output-json", default="artifacts/models/inference_output.json")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--input-npz", default=None, help="Optional npz with image_sequence and meteo_sequence")
    parser.add_argument("--metadata-path", default=None, help="Optional metadata json generated by export script")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-steps", type=int, default=0, help="<=0 means infer from metadata or default")
    parser.add_argument("--image-size", type=int, default=0, help="<=0 means infer from metadata or default")
    parser.add_argument("--meteo-steps", type=int, default=0, help="<=0 means infer from metadata or default")
    parser.add_argument(
        "--meteo-feature-count",
        type=int,
        default=0,
        help="<=0 means infer from metadata or default",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_inference(
        model_path=args.model_path,
        output_json=args.output_json,
        device=args.device,
        input_npz=args.input_npz,
        batch_size=args.batch_size,
        image_steps=args.image_steps,
        image_size=args.image_size,
        meteo_steps=args.meteo_steps,
        meteo_feature_count=args.meteo_feature_count,
        seed=args.seed,
        metadata_path=args.metadata_path,
    )
    print(f"[infer] output_json={args.output_json}")
    print(f"[infer] probabilities={result['rain_probability']}")


if __name__ == "__main__":
    main()
