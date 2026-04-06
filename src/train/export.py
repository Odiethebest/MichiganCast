from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.multimodal.model import MichiganCastMultimodalNet


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


def export_torchscript(
    *,
    checkpoint_path: str,
    output_path: str,
    metadata_path: str,
    image_steps: int,
    image_size: int,
    meteo_steps: int,
    meteo_feature_count: int,
    conv_hidden_dim: int,
    meteo_hidden_dim: int,
    fusion_hidden_dim: int,
    dropout: float,
    device: str,
) -> dict:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = _resolve_device(device)
    state_dict = torch.load(ckpt, map_location=target_device)
    inferred_conv_hidden = int(state_dict["convlstm.cell.conv.weight"].shape[0] // 4)
    inferred_meteo_hidden = int(state_dict["meteo_lstm.weight_ih_l0"].shape[0] // 4)
    inferred_meteo_features = int(state_dict["meteo_lstm.weight_ih_l0"].shape[1])
    inferred_fusion_hidden = int(state_dict["head.0.weight"].shape[0])

    conv_hidden_dim = inferred_conv_hidden if conv_hidden_dim <= 0 else conv_hidden_dim
    meteo_hidden_dim = inferred_meteo_hidden if meteo_hidden_dim <= 0 else meteo_hidden_dim
    meteo_feature_count = inferred_meteo_features if meteo_feature_count <= 0 else meteo_feature_count
    fusion_hidden_dim = inferred_fusion_hidden if fusion_hidden_dim <= 0 else fusion_hidden_dim

    model = MichiganCastMultimodalNet(
        image_channels=1,
        meteo_feature_count=meteo_feature_count,
        conv_hidden_dim=conv_hidden_dim,
        meteo_hidden_dim=meteo_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
    ).to(target_device)

    model.load_state_dict(state_dict)
    model.eval()

    example_image = torch.randn(1, image_steps, 1, image_size, image_size, device=target_device)
    example_meteo = torch.randn(1, meteo_steps, meteo_feature_count, device=target_device)

    with torch.no_grad():
        eager_out = model(example_image, example_meteo)
        scripted = torch.jit.trace(model, (example_image, example_meteo))
        scripted_out = scripted(example_image, example_meteo)
        max_abs_diff = float(torch.max(torch.abs(eager_out - scripted_out)).item())

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    meta = {
        "format": "torchscript",
        "checkpoint_path": str(ckpt),
        "torchscript_path": str(out_path),
        "device": str(target_device),
        "example_input_shape": {
            "image_sequence": [1, image_steps, 1, image_size, image_size],
            "meteo_sequence": [1, meteo_steps, meteo_feature_count],
        },
        "model_hparams": {
            "inferred_from_checkpoint": {
                "conv_hidden_dim": inferred_conv_hidden,
                "meteo_hidden_dim": inferred_meteo_hidden,
                "meteo_feature_count": inferred_meteo_features,
                "fusion_hidden_dim": inferred_fusion_hidden,
            },
            "conv_hidden_dim": conv_hidden_dim,
            "meteo_hidden_dim": meteo_hidden_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "dropout": dropout,
        },
        "trace_max_abs_diff": max_abs_diff,
    }
    meta_path = Path(metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MichiganCast multimodal model to TorchScript (T25)")
    parser.add_argument("--checkpoint-path", default="models/pytorch/michigancast_multimodal_best.pth")
    parser.add_argument("--output-path", default="artifacts/models/michigancast_multimodal.ts")
    parser.add_argument("--metadata-path", default="artifacts/models/michigancast_multimodal_metadata.json")
    parser.add_argument("--image-steps", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--meteo-steps", type=int, default=48)
    parser.add_argument("--meteo-feature-count", type=int, default=0, help="<=0 means infer from checkpoint")
    parser.add_argument("--conv-hidden-dim", type=int, default=0, help="<=0 means infer from checkpoint")
    parser.add_argument("--meteo-hidden-dim", type=int, default=0, help="<=0 means infer from checkpoint")
    parser.add_argument("--fusion-hidden-dim", type=int, default=0, help="<=0 means infer from checkpoint")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    meta = export_torchscript(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        image_steps=args.image_steps,
        image_size=args.image_size,
        meteo_steps=args.meteo_steps,
        meteo_feature_count=args.meteo_feature_count,
        conv_hidden_dim=args.conv_hidden_dim,
        meteo_hidden_dim=args.meteo_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        device=args.device,
    )
    print(f"[export] torchscript={meta['torchscript_path']}")
    print(f"[export] metadata={args.metadata_path}")
    print(f"[export] trace_max_abs_diff={meta['trace_max_abs_diff']:.8f}")


if __name__ == "__main__":
    main()
