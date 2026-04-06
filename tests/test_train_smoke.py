import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.multimodal.model import MichiganCastMultimodalNet
from src.models.multimodal.train_loop import TrainingConfig, fit_multimodal_model


class _TinyMultimodalDataset(Dataset):
    def __init__(self, length: int, meteo_features: int) -> None:
        self.length = length
        self.meteo_features = meteo_features

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image_sequence = torch.randn(4, 1, 8, 8)
        meteo_sequence = torch.randn(6, self.meteo_features)
        target = torch.tensor(float(idx % 2), dtype=torch.float32)
        return {
            "image_sequence": image_sequence,
            "meteo_sequence": meteo_sequence,
            "target": target,
        }


class TestTrainSmoke(unittest.TestCase):
    def test_fit_multimodal_model_smoke(self) -> None:
        meteo_features = 5
        model = MichiganCastMultimodalNet(
            image_channels=1,
            meteo_feature_count=meteo_features,
            conv_hidden_dim=8,
            meteo_hidden_dim=16,
            fusion_hidden_dim=8,
            dropout=0.1,
        )
        train_loader = DataLoader(_TinyMultimodalDataset(length=16, meteo_features=meteo_features), batch_size=4)
        val_loader = DataLoader(_TinyMultimodalDataset(length=8, meteo_features=meteo_features), batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "smoke_checkpoint.pth"
            cfg = TrainingConfig(
                epochs=1,
                learning_rate=1e-3,
                early_stopping_patience=2,
                checkpoint_path=str(ckpt),
                use_scheduler=True,
            )
            result = fit_multimodal_model(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=torch.device("cpu"),
                config=cfg,
            )
            self.assertGreaterEqual(result["epochs_ran"], 1)
            self.assertTrue(ckpt.exists())


if __name__ == "__main__":
    unittest.main()
