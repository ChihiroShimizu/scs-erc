from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


class BoundaryAwareResidualReranker(nn.Module):
    """Safely rerank anchor logits near their classification boundary.

    A frozen causal state tracker supplies contextual probabilities. The trainable
    module predicts a bounded, zero-initialized class-specific residual. Stable and
    transition labels are never model inputs and never select an inference branch.
    """

    VALID_VARIANTS = {
        "residual",
        "margin_gate",
        "margin_loss",
        "keep_loss",
        "topk_rerank",
    }

    def __init__(
        self,
        config: dict[str, Any],
        num_classes: int,
        context_tracker: nn.Module,
    ) -> None:
        super().__init__()
        cfg = config.get("boundary_model", {})
        self.num_classes = int(num_classes)
        self.variant = str(cfg.get("variant", "margin_gate"))
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"boundary_model.variant must be one of {sorted(self.VALID_VARIANTS)}, "
                f"got {self.variant!r}"
            )
        self.context_tracker = context_tracker
        self.context_tracker.requires_grad_(False)
        self.context_tracker.eval()

        hidden_dim = int(cfg.get("hidden_dim", 128))
        dropout = float(cfg.get("dropout", 0.15))
        feature_dim = 4 * self.num_classes + 4
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden_dim, self.num_classes)
        self.gate_head = nn.Linear(hidden_dim, 1)

        self.residual_max = float(cfg.get("residual_max", 2.0))
        self.gate_max = float(cfg.get("gate_max", 0.50))
        self.margin_threshold = float(cfg.get("margin_threshold", 1.0))
        self.margin_temperature = float(cfg.get("margin_temperature", 0.5))
        self.top_k = min(int(cfg.get("top_k", 3)), self.num_classes)
        if self.residual_max <= 0.0:
            raise ValueError("boundary_model.residual_max must be positive")
        if not 0.0 < self.gate_max <= 1.0:
            raise ValueError("boundary_model.gate_max must be in (0, 1]")
        if self.margin_temperature <= 0.0:
            raise ValueError("boundary_model.margin_temperature must be positive")
        if not 1 <= self.top_k <= self.num_classes:
            raise ValueError("boundary_model.top_k is outside the class range")

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        gate_init = float(cfg.get("gate_init", 0.20))
        if not 0.0 < gate_init < 1.0:
            raise ValueError("boundary_model.gate_init must be in (0, 1)")
        nn.init.constant_(self.gate_head.bias, _logit(gate_init))

    def train(self, mode: bool = True) -> BoundaryAwareResidualReranker:
        super().train(mode)
        self.context_tracker.eval()
        return self

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
        return -(
            probabilities.float()
            * torch.log(probabilities.float().clamp_min(1e-8))
        ).sum(dim=-1, keepdim=True)

    def forward(
        self,
        embeddings: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        current_logits: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.context_tracker.eval()
        with torch.no_grad():
            context_logits, state_aux = self.context_tracker(
                embeddings,
                speaker_ids,
                mask,
                current_logits=current_logits,
                return_aux=True,
            )
        anchor_logits = state_aux["anchor_logits"].float()
        context_logits = context_logits.float()
        prior = state_aux["prior_emotion_distribution"].float()
        anchor_probabilities = torch.softmax(anchor_logits, dim=-1)
        context_probabilities = torch.softmax(context_logits, dim=-1)
        context_delta = context_logits - anchor_logits

        top_two = anchor_logits.topk(min(2, self.num_classes), dim=-1).values
        if self.num_classes > 1:
            anchor_margin = top_two[..., 0] - top_two[..., 1]
        else:
            anchor_margin = torch.zeros_like(top_two[..., 0])
        anchor_entropy = self._entropy(anchor_probabilities)
        prior_entropy = self._entropy(prior)
        midpoint = 0.5 * (anchor_probabilities + prior)
        js_divergence = 0.5 * (
            (
                anchor_probabilities
                * (
                    torch.log(anchor_probabilities.clamp_min(1e-8))
                    - torch.log(midpoint.clamp_min(1e-8))
                )
            ).sum(dim=-1, keepdim=True)
            + (
                prior
                * (
                    torch.log(prior.clamp_min(1e-8))
                    - torch.log(midpoint.clamp_min(1e-8))
                )
            ).sum(dim=-1, keepdim=True)
        )
        known = state_aux["state_known_mask"].float().unsqueeze(-1)
        scalar_features = torch.cat(
            [
                anchor_margin.unsqueeze(-1),
                anchor_entropy,
                prior_entropy,
                js_divergence,
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                anchor_probabilities,
                context_probabilities,
                prior,
                context_delta,
                scalar_features,
            ],
            dim=-1,
        )
        hidden = self.feature_encoder(features)
        raw_residual = self.residual_head(hidden)
        raw_residual = raw_residual - raw_residual.mean(dim=-1, keepdim=True)
        residual = self.residual_max * torch.tanh(raw_residual)

        if self.variant == "topk_rerank":
            candidate_indices = anchor_logits.topk(self.top_k, dim=-1).indices
            candidate_mask = torch.zeros_like(residual, dtype=torch.bool)
            candidate_mask.scatter_(-1, candidate_indices, True)
            residual = residual * candidate_mask.to(residual.dtype)
        else:
            candidate_mask = torch.ones_like(residual, dtype=torch.bool)

        learned_gate = torch.sigmoid(self.gate_head(hidden))
        if self.variant == "residual":
            boundary_factor = torch.ones_like(learned_gate)
            gate = torch.full_like(learned_gate, self.gate_max)
        else:
            boundary_factor = torch.sigmoid(
                (self.margin_threshold - anchor_margin.unsqueeze(-1))
                / self.margin_temperature
            )
            gate = self.gate_max * learned_gate * boundary_factor
        gate = gate * mask.unsqueeze(-1).to(gate.dtype)
        logits = anchor_logits + gate * residual
        logits = torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))

        if not return_aux:
            return logits
        aux = {
            "anchor_logits": anchor_logits,
            "context_candidate_logits": context_logits,
            "prior_emotion_distribution": prior,
            "anchor_probabilities": anchor_probabilities,
            "context_probabilities": context_probabilities,
            "boundary_residual": residual,
            "boundary_gate": gate,
            "boundary_factor": boundary_factor,
            "anchor_margin": anchor_margin,
            "candidate_mask": candidate_mask,
            "state_known_mask": state_aux["state_known_mask"],
            "context_known_feature": known,
        }
        return logits, aux

