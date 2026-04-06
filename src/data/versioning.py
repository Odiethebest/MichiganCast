from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


DEFAULT_CONFIG = "configs/data/versioning.yaml"


@dataclass(frozen=True)
class VersioningConfig:
    manifest_root: str
    allowed_layers: tuple[str, ...]
    hash_algorithm: str = "sha256"


def load_versioning_config(config_path: str = DEFAULT_CONFIG) -> VersioningConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Versioning config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload.get("versioning", {})
    manifest_root = section.get("manifest_root", "configs/data/versions")
    allowed_layers = tuple(section.get("allowed_layers", []))
    if not allowed_layers:
        raise ValueError("allowed_layers in versioning config cannot be empty.")
    hash_algorithm = section.get("hash_algorithm", "sha256")
    return VersioningConfig(
        manifest_root=manifest_root,
        allowed_layers=allowed_layers,
        hash_algorithm=hash_algorithm,
    )


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _collect_file_stats(path: Path) -> Dict[str, object]:
    stats = {
        "size_bytes": int(path.stat().st_size),
        "suffix": path.suffix.lower(),
    }
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=1000, low_memory=False)
        stats["sampled_rows"] = int(len(df))
        stats["sampled_columns"] = list(df.columns)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        stats["rows"] = int(len(df))
        stats["columns"] = list(df.columns)
    return stats


def _hash_directory(path: Path, algorithm: str = "sha256") -> Dict[str, object]:
    hasher = hashlib.new(algorithm)
    file_entries: List[Dict[str, object]] = []
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    for file in files:
        relative = str(file.relative_to(path))
        file_hash = _hash_file(file, algorithm=algorithm)
        hasher.update(relative.encode("utf-8"))
        hasher.update(file_hash.encode("utf-8"))
        file_entries.append(
            {
                "relative_path": relative,
                "size_bytes": int(file.stat().st_size),
                "hash": file_hash,
            }
        )
    return {
        "size_bytes": int(sum(x["size_bytes"] for x in file_entries)),
        "file_count": int(len(file_entries)),
        "directory_hash": hasher.hexdigest(),
        "files": file_entries,
    }


def create_dataset_manifest(
    *,
    dataset_id: str,
    layer: str,
    target_path: str,
    source_paths: List[str],
    build_command: str,
    notes: str,
    config: VersioningConfig,
) -> Dict[str, object]:
    if layer not in config.allowed_layers:
        raise ValueError(f"Layer '{layer}' is not in allowed layers: {list(config.allowed_layers)}")

    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Target path not found: {path}")

    if path.is_file():
        artifact_info = {
            "type": "file",
            "path": str(path),
            "hash": _hash_file(path, algorithm=config.hash_algorithm),
            "stats": _collect_file_stats(path),
        }
    else:
        artifact_info = {
            "type": "directory",
            "path": str(path),
            **_hash_directory(path, algorithm=config.hash_algorithm),
        }

    manifest = {
        "dataset_id": dataset_id,
        "layer": layer,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": config.hash_algorithm,
        "artifact": artifact_info,
        "lineage": {
            "source_paths": source_paths,
            "build_command": build_command,
            "notes": notes,
        },
    }
    return manifest


def save_manifest(manifest: Dict[str, object], manifest_root: str) -> Path:
    out_dir = Path(manifest_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['dataset_id']}.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset version manifest tool (DVC-equivalent)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Versioning config yaml")
    parser.add_argument("--dataset-id", required=True, help="Stable dataset identifier")
    parser.add_argument("--layer", required=True, help="Data layer name")
    parser.add_argument("--target-path", required=True, help="Artifact file or directory to snapshot")
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        help="Upstream source path (repeatable)",
    )
    parser.add_argument("--build-command", default="", help="Command that generated this dataset")
    parser.add_argument("--notes", default="", help="Optional free-form notes")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = load_versioning_config(args.config)
    manifest = create_dataset_manifest(
        dataset_id=args.dataset_id,
        layer=args.layer,
        target_path=args.target_path,
        source_paths=list(args.source_path),
        build_command=args.build_command,
        notes=args.notes,
        config=cfg,
    )
    out_path = save_manifest(manifest, cfg.manifest_root)
    print(f"[versioning] dataset_id={args.dataset_id} layer={args.layer}")
    print(f"[versioning] output={out_path}")


if __name__ == "__main__":
    main()
