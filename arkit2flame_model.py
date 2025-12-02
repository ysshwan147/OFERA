
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# -----------------------------
# Core Blocks
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h, inplace=True)
        return x + h


def _make_trunk(in_dim: int, hidden_dim: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        *[ResidualBlock(hidden_dim) for _ in range(num_blocks)],
    )


# -----------------------------
# Baseline: single MLP 51 -> 68
# -----------------------------
class ARKitToFLAMENet(nn.Module):
    def __init__(self, in_dim: int = 51, hidden_dim: int = 128, out_dim: int = 68, num_blocks: int = 4):
        super().__init__()
        self.trunk = _make_trunk(in_dim, hidden_dim, num_blocks)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return self.head(z)

