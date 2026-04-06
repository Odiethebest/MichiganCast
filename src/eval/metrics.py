from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def recall_at_precision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    precision_threshold: float = 0.7,
) -> float:
    """
    Return the best achievable recall under a minimum precision constraint.

    If no operating point reaches the precision threshold, returns 0.0.
    """
    if not (0.0 <= precision_threshold <= 1.0):
        raise ValueError("precision_threshold must be within [0, 1].")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= precision_threshold
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    precision_threshold: float = 0.7,
) -> Dict[str, object]:
    """
    Unified binary metrics for MichiganCast.

    Includes PR-AUC, F1, Recall, Recall@Precision, Brier, and confusion matrix.
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision_threshold": float(precision_threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_precision": float(
            recall_at_precision_threshold(y_true, y_prob, precision_threshold=precision_threshold)
        ),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics
