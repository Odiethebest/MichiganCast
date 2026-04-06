"""Multimodal dataset/model/train modules for MichiganCast (T20)."""

from .dataset import MultimodalForecastDataset, build_multimodal_datasets
from .model import MichiganCastMultimodalNet
from .train_loop import TrainingConfig, evaluate_loader, fit_multimodal_model, train_one_epoch

__all__ = [
    "MultimodalForecastDataset",
    "build_multimodal_datasets",
    "MichiganCastMultimodalNet",
    "TrainingConfig",
    "train_one_epoch",
    "evaluate_loader",
    "fit_multimodal_model",
]
