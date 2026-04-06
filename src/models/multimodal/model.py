from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        combined = torch.cat([x_t, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = image_size
        h_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        c_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        return h_state, c_state


class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], batch_first: bool = True) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = x.size()
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(batch_size=b, image_size=(h, w), device=x.device)

        h_t, c_t = hidden_state
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :, :, :], (h_t, c_t))
            outputs.append(h_t)
        output_seq = torch.stack(outputs, dim=1)
        return output_seq, (h_t, c_t)


class MichiganCastMultimodalNet(nn.Module):
    """ConvLSTM (image) + LSTM (meteo) fusion network for rain-event forecasting."""

    def __init__(
        self,
        *,
        image_channels: int = 1,
        meteo_feature_count: int,
        conv_hidden_dim: int = 64,
        meteo_hidden_dim: int = 128,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=image_channels,
            hidden_dim=conv_hidden_dim,
            kernel_size=(3, 3),
            batch_first=True,
        )
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_flatten = nn.Flatten()

        self.meteo_lstm = nn.LSTM(
            input_size=meteo_feature_count,
            hidden_size=meteo_hidden_dim,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(conv_hidden_dim + meteo_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, image_sequence: torch.Tensor, meteo_sequence: torch.Tensor) -> torch.Tensor:
        _, (h_img, _) = self.convlstm(image_sequence)
        img_feat = self.image_flatten(self.image_pool(h_img))

        _, (h_meteo, _) = self.meteo_lstm(meteo_sequence)
        meteo_feat = h_meteo.squeeze(0)

        combined = torch.cat([img_feat, meteo_feat], dim=1)
        logits = self.head(combined).squeeze(-1)
        return logits
