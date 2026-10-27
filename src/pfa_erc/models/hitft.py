from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .layers import (
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    causal_padding_attention_mask,
)


class HiTFTForecaster(nn.Module):
    """High-dimensional TFT variant for pseudo-future embedding prediction.

    Inputs are causal histories of RoBERTa CLS embeddings. The output has shape
    ``[batch, future_steps, embedding_dim]``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        self.embedding_dim = int(model["embedding_dim"])
        self.hidden_dim = int(model["hidden_dim"])
        self.future_steps = int(model["future_steps"])
        self.forecaster_mode = str(model.get("forecaster_mode", "absolute"))
        self.transition_residual_scale = float(
            model.get("transition_residual_scale", 0.25)
        )
        self.max_speakers = int(data.get("max_speakers", 16))
        speaker_dim = int(model.get("speaker_embedding_dim", 64))
        dropout = float(model.get("dropout", 0.3))
        heads = int(model.get("num_heads", 4))

        self.speaker_embedding = nn.Embedding(self.max_speakers, speaker_dim)
        self.static_grn = GatedResidualNetwork(speaker_dim, self.hidden_dim, dropout=dropout)
        self.variable_selection = VariableSelectionNetwork(
            [self.embedding_dim, speaker_dim, 2],
            self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.local_grn = GatedResidualNetwork(
            self.hidden_dim,
            self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.local_glu = GatedLinearUnit(self.hidden_dim, self.hidden_dim)
        self.local_norm = nn.LayerNorm(self.hidden_dim)
        self.context_grn = GatedResidualNetwork(
            self.hidden_dim,
            self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.temporal_attention = nn.MultiheadAttention(
            self.hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.attention_glu = GatedLinearUnit(self.hidden_dim, self.hidden_dim)
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.enrichment_glu = GatedLinearUnit(self.hidden_dim, self.hidden_dim)
        self.enrichment_norm = nn.LayerNorm(self.hidden_dim)

        self.horizon_embedding = nn.Parameter(torch.empty(self.future_steps, self.hidden_dim))
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.horizon_attention = nn.MultiheadAttention(
            self.hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.horizon_grn = GatedResidualNetwork(self.hidden_dim, self.hidden_dim, dropout=dropout)
        self.output: nn.Linear | None = None
        self.stable_delta_head: nn.Linear | None = None
        self.transition_delta_head: nn.Linear | None = None
        self.stable_residual_head: nn.Linear | None = None
        self.transition_residual_head: nn.Linear | None = None
        self.transition_logit_head: nn.Linear | None = None
        if self.forecaster_mode == "absolute":
            self.output = nn.Linear(self.hidden_dim, self.embedding_dim)
        elif self.forecaster_mode == "transition_aware":
            self.stable_delta_head = nn.Linear(self.hidden_dim, self.embedding_dim)
            self.transition_delta_head = nn.Linear(self.hidden_dim, self.embedding_dim)
            self.transition_logit_head = nn.Linear(self.hidden_dim, 1)
        elif self.forecaster_mode == "transition_residual":
            # Keep the original absolute head so an already-good HiTFT
            # checkpoint can initialize both the backbone and base forecast.
            self.output = nn.Linear(self.hidden_dim, self.embedding_dim)
            self.stable_residual_head = nn.Linear(self.hidden_dim, self.embedding_dim)
            self.transition_residual_head = nn.Linear(self.hidden_dim, self.embedding_dim)
            self.transition_logit_head = nn.Linear(self.hidden_dim, 1)
            # Both candidates start almost exactly at the warm-started base
            # prediction. Small asymmetric weights avoid an unstable cosine
            # gradient at a perfectly zero residual while preserving base MSE.
            nn.init.normal_(self.stable_residual_head.weight, std=1e-4)
            nn.init.zeros_(self.stable_residual_head.bias)
            nn.init.normal_(self.transition_residual_head.weight, std=1e-4)
            nn.init.zeros_(self.transition_residual_head.bias)
            transition_prior = float(model.get("transition_prior", 0.5))
            if not 0.0 < transition_prior < 1.0:
                raise ValueError("model.transition_prior must be strictly between 0 and 1")
            nn.init.zeros_(self.transition_logit_head.weight)
            nn.init.constant_(
                self.transition_logit_head.bias,
                math.log(transition_prior / (1.0 - transition_prior)),
            )
        else:
            raise ValueError(f"Unsupported model.forecaster_mode: {self.forecaster_mode}")

    @property
    def is_transition_forecaster(self) -> bool:
        return self.forecaster_mode in {"transition_aware", "transition_residual"}

    def transition_head_parameters(self) -> list[nn.Parameter]:
        modules = [
            self.stable_delta_head,
            self.transition_delta_head,
            self.stable_residual_head,
            self.transition_residual_head,
            self.transition_logit_head,
        ]
        return [
            parameter
            for module in modules
            if module is not None
            for parameter in module.parameters()
        ]

    def set_transition_backbone_trainable(self, trainable: bool) -> None:
        if not self.is_transition_forecaster:
            raise ValueError("Backbone staging is only available for transition forecasters")
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        # Transition heads remain trainable during the frozen-backbone stage.
        for parameter in self.transition_head_parameters():
            parameter.requires_grad_(True)

    def forward(
        self,
        history: torch.Tensor,
        speaker_ids: torch.Tensor,
        history_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if history.ndim != 3:
            raise ValueError(f"history must be [B,L,D], got {tuple(history.shape)}")
        batch_size, length, _ = history.shape
        if not torch.all(history_mask.any(dim=1)):
            raise ValueError("Every HiTFT sample must contain at least one observed step")

        speaker_ids = speaker_ids.clamp(min=0, max=self.max_speakers - 1)
        speaker = self.speaker_embedding(speaker_ids)
        mask_float = history_mask.unsqueeze(-1).to(speaker.dtype)
        static_speaker = (speaker * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        static_context = self.static_grn(static_speaker)

        positions = torch.arange(length, device=history.device, dtype=history.dtype)
        denom = max(length - 1, 1)
        position_features = torch.stack(
            [positions / denom, torch.log1p(positions) / torch.log1p(positions.new_tensor(denom))],
            dim=-1,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        selected, variable_weights = self.variable_selection(
            [history, speaker, position_features], static_context
        )
        selected = selected * mask_float
        local_sequence, _ = self.lstm(selected)
        local_enriched = self.local_grn(selected, static_context)
        local = self.local_norm(self.local_glu(local_enriched) + local_sequence)

        enriched = self.context_grn(local, static_context)
        temporal_mask = causal_padding_attention_mask(
            history_mask,
            self.temporal_attention.num_heads,
        )
        temporal, temporal_weights = self.temporal_attention(
            enriched,
            enriched,
            enriched,
            attn_mask=temporal_mask,
            need_weights=return_aux,
        )
        # The batch-specific temporal mask prevents all-masked query rows.
        # Keep nan_to_num as a defensive boundary for non-mask-related bad
        # inputs or backend-specific numerical failures.
        temporal = torch.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)
        temporal = torch.where(history_mask.unsqueeze(-1), temporal, torch.zeros_like(temporal))
        if temporal_weights is not None:
            temporal_weights = torch.nan_to_num(
                temporal_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            temporal_weights = torch.where(
                history_mask.unsqueeze(-1),
                temporal_weights,
                torch.zeros_like(temporal_weights),
            )
        attended = self.attention_norm(self.attention_glu(temporal) + enriched)
        attended = self.enrichment_norm(self.enrichment_glu(attended) + local)
        attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        attended = torch.where(history_mask.unsqueeze(-1), attended, torch.zeros_like(attended))

        indices = torch.arange(length, device=history.device).unsqueeze(0).expand(batch_size, -1)
        last_index = indices.masked_fill(~history_mask, -1).max(dim=1).values.clamp_min(0)
        last = attended[torch.arange(batch_size, device=history.device), last_index]
        horizon_query = last.unsqueeze(1) + self.horizon_embedding.unsqueeze(0)
        horizon_context, horizon_weights = self.horizon_attention(
            horizon_query,
            attended,
            attended,
            key_padding_mask=~history_mask,
            need_weights=return_aux,
        )
        horizon_features = self.horizon_grn(horizon_query + horizon_context)
        extra: dict[str, torch.Tensor] = {}
        if self.forecaster_mode == "absolute":
            assert self.output is not None
            prediction = self.output(horizon_features)
        elif self.forecaster_mode == "transition_aware":
            assert self.stable_delta_head is not None
            assert self.transition_delta_head is not None
            assert self.transition_logit_head is not None
            anchor = history[torch.arange(batch_size, device=history.device), last_index]
            stable_delta = self.stable_delta_head(horizon_features)
            transition_delta = self.transition_delta_head(horizon_features)
            transition_logits = self.transition_logit_head(horizon_features).squeeze(-1)
            transition_probability = torch.sigmoid(transition_logits)
            stable_prediction = anchor.unsqueeze(1) + stable_delta
            transition_prediction = anchor.unsqueeze(1) + transition_delta
            prediction = stable_prediction + transition_probability.unsqueeze(-1) * (
                transition_prediction - stable_prediction
            )
            extra = {
                "anchor": anchor,
                "stable_delta": stable_delta,
                "transition_delta": transition_delta,
                "stable_prediction": stable_prediction,
                "transition_prediction": transition_prediction,
                "transition_logits": transition_logits,
                "transition_probability": transition_probability,
            }
        else:
            assert self.output is not None
            assert self.stable_residual_head is not None
            assert self.transition_residual_head is not None
            assert self.transition_logit_head is not None
            anchor = history[torch.arange(batch_size, device=history.device), last_index]
            base_prediction = self.output(horizon_features)
            stable_residual = self.transition_residual_scale * torch.tanh(
                self.stable_residual_head(horizon_features)
            )
            transition_residual = self.transition_residual_scale * torch.tanh(
                self.transition_residual_head(horizon_features)
            )
            transition_logits = self.transition_logit_head(horizon_features).squeeze(-1)
            transition_probability = torch.sigmoid(transition_logits)
            stable_prediction = base_prediction + stable_residual
            transition_prediction = base_prediction + transition_residual
            prediction = stable_prediction + transition_probability.unsqueeze(-1) * (
                transition_prediction - stable_prediction
            )
            extra = {
                "anchor": anchor,
                "base_prediction": base_prediction,
                "direction_reference": base_prediction.detach(),
                "stable_delta": stable_residual,
                "transition_delta": transition_residual,
                "stable_residual": stable_residual,
                "transition_residual": transition_residual,
                "stable_prediction": stable_prediction,
                "transition_prediction": transition_prediction,
                "transition_logits": transition_logits,
                "transition_probability": transition_probability,
            }
        if not return_aux:
            return prediction
        return prediction, {
            "variable_weights": variable_weights,
            "temporal_weights": temporal_weights,
            "horizon_weights": horizon_weights,
            **extra,
        }
