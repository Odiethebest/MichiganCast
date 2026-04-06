from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.eval.metrics import evaluate_binary_predictions


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    threshold: float = 0.5
    precision_threshold: float = 0.7
    early_stopping_patience: int = 5
    checkpoint_path: str = "models/pytorch/michigancast_multimodal_best.pth"
    use_scheduler: bool = True


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {
        "image_sequence": batch["image_sequence"].to(device, non_blocking=non_blocking),
        "meteo_sequence": batch["meteo_sequence"].to(device, non_blocking=non_blocking),
        "target": batch["target"].to(device, non_blocking=non_blocking),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch["image_sequence"], batch["meteo_sequence"])
        loss = criterion(logits, batch["target"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(num_batches, 1)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    precision_threshold: float = 0.7,
) -> Tuple[float, Dict[str, object]]:
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch["image_sequence"], batch["meteo_sequence"])
            loss = criterion(logits, batch["target"])
            probs = torch.sigmoid(logits)

            running_loss += float(loss.item())
            num_batches += 1
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(batch["target"].detach().cpu().numpy())

    avg_loss = running_loss / max(num_batches, 1)
    if not all_targets:
        return avg_loss, {
            "threshold": threshold,
            "precision_threshold": precision_threshold,
            "pr_auc": 0.0,
            "f1": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "recall_at_precision": 0.0,
            "brier": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_prob = np.concatenate(all_probs).astype(float)
    y_true = np.concatenate(all_targets).astype(int)
    metrics = evaluate_binary_predictions(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        precision_threshold=precision_threshold,
    )
    return avg_loss, metrics


def fit_multimodal_model(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    pos_weight: float | None = None,
) -> Dict[str, object]:
    if pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        if config.use_scheduler
        else None
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_f1": [],
        "val_recall": [],
        "val_precision": [],
        "val_recall_at_precision": [],
        "val_brier": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint = Path(config.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            threshold=config.threshold,
            precision_threshold=config.precision_threshold,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall_at_precision"].append(val_metrics["recall_at_precision"])
        history["val_brier"].append(val_metrics["brier"])

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("[train_loop] early stopping triggered")
                break

    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint),
        "epochs_ran": len(history["train_loss"]),
    }
