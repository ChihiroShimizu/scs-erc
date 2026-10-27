from __future__ import annotations

import math

import torch
from torch import nn


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


class SceneGate(nn.Module):
    """Causal soft reset gate for the dialogue-level story state."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        init_probability: float = 0.05,
        max_probability: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < init_probability < max_probability <= 1.0:
            raise ValueError(
                "SceneGate requires 0 < init_probability < max_probability <= 1"
            )
        self.max_probability = float(max_probability)
        feature_dim = hidden_dim * 3 + 3
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias,
            _logit(init_probability / self.max_probability),
        )

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        story: torch.Tensor,
        speaker_change: torch.Tensor,
        speaker_gap: torch.Tensor,
    ) -> torch.Tensor:
        semantic_distance = 1.0 - nn.functional.cosine_similarity(
            current.float(), previous.float(), dim=-1, eps=1e-8
        ).unsqueeze(-1)
        features = torch.cat(
            [
                current,
                previous,
                story,
                semantic_distance.to(current.dtype),
                speaker_change.to(current.dtype),
                speaker_gap.to(current.dtype),
            ],
            dim=-1,
        )
        return self.max_probability * torch.sigmoid(self.network(features))
