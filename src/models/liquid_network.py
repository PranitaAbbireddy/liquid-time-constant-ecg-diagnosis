import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidTimeConstantCell(nn.Module):
    """Input-dependent time constant liquid update."""

    def __init__(self, input_size: int, hidden_size: int, tau_min: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.recurrent_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_proj = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.input_proj(x_t) + self.recurrent_proj(h_prev))
        tau_input = torch.cat([x_t, h_prev], dim=-1)
        tau = self.tau_min + F.softplus(self.tau_proj(tau_input))
        return h_prev + (candidate - h_prev) / tau


class LiquidECGClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int = 187,
        input_size: int = 1,
        hidden_size: int = 128,
        output_size: int = 5,
        tau_min: float = 0.5,
        pooling: str = "last",
        temporal_stride: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.temporal_stride = max(1, temporal_stride)
        self.cell = LiquidTimeConstantCell(input_size, hidden_size, tau_min=tau_min)
        self.fc = nn.Linear(hidden_size, output_size)

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_sequence(x)
        if self.temporal_stride > 1:
            x = x[:, :: self.temporal_stride, :]
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        if self.pooling == "mean":
            acc = torch.zeros_like(h)

        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            if self.pooling == "mean":
                acc = acc + h

        if self.pooling == "mean":
            pooled = acc / seq_len
        else:
            pooled = h
        return self.fc(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
