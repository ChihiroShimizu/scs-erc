from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ContextUtilityRouter(nn.Module):
    """Predict whether context helps and which history turns are useful.

    The module is deliberately causal.  Its target-level branch sees the
    current utterance, summaries of *past* same/other-speaker turns, cached
    current-only Anchor logits, and simple availability/gap features.  The
    pair branch scores one observed history turn against the current target.
    Gold emotion labels and counterfactual effects are training targets only.
    """

    NUM_UTILITY_CLASSES = 3
    HARMFUL = 0
    UNCERTAIN = 1
    HELPFUL = 2

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        if self.embedding_dim <= 0 or self.num_classes <= 1 or self.hidden_dim <= 0:
            raise ValueError("ContextUtilityRouter dimensions must be positive")

        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # anchor probability + normalized entropy + top1/top2 margin
        anchor_dim = self.num_classes + 2
        # same available, other available, normalized turn, normalized
        # same-speaker gap
        target_scalar_dim = 4
        target_input_dim = self.hidden_dim * 3 + anchor_dim + target_scalar_dim
        self.target_backbone = nn.Sequential(
            nn.LayerNorm(target_input_dim),
            nn.Linear(target_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.target_utility_head = nn.Linear(self.hidden_dim, 1)
        self.target_class_head = nn.Linear(
            self.hidden_dim, self.NUM_UTILITY_CLASSES
        )
        self.cue_slow_head = nn.Linear(self.hidden_dim, 1)

        # current/history/difference/product embeddings, two Anchor
        # distributions, same-speaker flag, and normalized gap.
        pair_input_dim = self.hidden_dim * 4 + self.num_classes * 2 + 2
        self.pair_backbone = nn.Sequential(
            nn.LayerNorm(pair_input_dim),
            nn.Linear(pair_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.pair_utility_head = nn.Linear(self.hidden_dim, 1)

    def _anchor_features(self, logits: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        safe = probability.clamp_min(1e-8)
        entropy = -(safe * safe.log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(
            probability.new_tensor(float(self.num_classes))
        )
        top2 = probability.topk(k=2, dim=-1).values
        margin = top2[..., :1] - top2[..., 1:2]
        return torch.cat([probability, entropy, margin], dim=-1)

    def forward_target(
        self,
        current: torch.Tensor,
        same_history: torch.Tensor,
        other_history: torch.Tensor,
        anchor_logits: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = current.shape[:-1]
        for name, value in (
            ("same_history", same_history),
            ("other_history", other_history),
        ):
            if value.shape != current.shape:
                raise ValueError(f"{name} must match current embeddings")
        if anchor_logits.shape != (*expected, self.num_classes):
            raise ValueError("anchor_logits has an incompatible shape")
        if scalar_features.shape != (*expected, 4):
            raise ValueError("scalar_features must end in four causal values")

        current_hidden = self.embedding_projection(current)
        same_hidden = self.embedding_projection(same_history)
        other_hidden = self.embedding_projection(other_history)
        hidden = self.target_backbone(
            torch.cat(
                [
                    current_hidden,
                    same_hidden,
                    other_hidden,
                    self._anchor_features(anchor_logits),
                    scalar_features.to(current.dtype),
                ],
                dim=-1,
            )
        )
        return {
            "utility": self.target_utility_head(hidden).squeeze(-1),
            "class_logits": self.target_class_head(hidden),
            "cue_slow_logit": self.cue_slow_head(hidden).squeeze(-1),
        }

    def forward_pair(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        current_anchor_logits: torch.Tensor,
        history_anchor_logits: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        if history.shape != current.shape:
            raise ValueError("history must match current embeddings")
        expected = current.shape[:-1]
        for name, value in (
            ("current_anchor_logits", current_anchor_logits),
            ("history_anchor_logits", history_anchor_logits),
        ):
            if value.shape != (*expected, self.num_classes):
                raise ValueError(f"{name} has an incompatible shape")
        if scalar_features.shape != (*expected, 2):
            raise ValueError(
                "pair scalar_features must contain same-speaker and gap"
            )

        current_hidden = self.embedding_projection(current)
        history_hidden = self.embedding_projection(history)
        pair_hidden = self.pair_backbone(
            torch.cat(
                [
                    current_hidden,
                    history_hidden,
                    (current_hidden - history_hidden).abs(),
                    current_hidden * history_hidden,
                    torch.softmax(current_anchor_logits.float(), dim=-1).to(
                        current.dtype
                    ),
                    torch.softmax(history_anchor_logits.float(), dim=-1).to(
                        current.dtype
                    ),
                    scalar_features.to(current.dtype),
                ],
                dim=-1,
            )
        )
        return self.pair_utility_head(pair_hidden).squeeze(-1)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        embedding_dim: int,
        num_classes: int,
    ) -> "ContextUtilityRouter":
        settings = config.get("experiment13", {})
        return cls(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=int(settings.get("router_hidden_dim", 128)),
            dropout=float(settings.get("router_dropout", 0.2)),
        )


__all__ = ["ContextUtilityRouter"]
