from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .layers import masked_attention_pool


class ClassSpecificResidualCorrector(nn.Module):
    """Causal, class-specific correction on top of a frozen utterance anchor.

    The module never receives emotion labels.  It builds two explicit context
    signals from the current-utterance hidden states:

    * change: current state minus the previous state of the same speaker;
    * trigger: attention-pooled earlier states from other speakers.

    A signed residual is predicted for every emotion logit.  An optional vector
    gate controls each emotion independently, avoiding the single scalar gate
    used by Experiments 3C/4C.  The residual head is zero-initialized so the
    complete model starts exactly at the fixed 3A anchor.
    """

    def __init__(self, config: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        model = config["model"]
        self.hidden_dim = int(model["hidden_dim"])
        self.num_classes = int(num_classes)
        self.max_speakers = int(config["data"].get("max_speakers", 16))
        self.use_change = bool(model.get("residual_use_change", False))
        self.use_trigger = bool(model.get("residual_use_trigger", False))
        self.use_vector_gate = bool(model.get("residual_use_vector_gate", False))
        self.use_transition_feature = bool(
            model.get("residual_use_transition_feature", False)
        )
        self.residual_scale = float(model.get("residual_scale", 1.0))
        if self.residual_scale <= 0.0:
            raise ValueError("model.residual_scale must be positive")

        dropout = float(model.get("dropout", 0.3))
        transition_prior = float(model.get("transition_prior", 0.54))
        if not 0.0 < transition_prior < 1.0:
            raise ValueError("model.transition_prior must be strictly between 0 and 1")

        if self.use_trigger:
            self.trigger_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.trigger_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.trigger_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.trigger_query = None
            self.trigger_key = None
            self.trigger_value = None

        feature_dim = self.hidden_dim + self.num_classes
        if self.use_change:
            feature_dim += self.hidden_dim + 1
        if self.use_trigger:
            feature_dim += self.hidden_dim + 1

        self.transition_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.transition_head.weight)
        nn.init.constant_(
            self.transition_head.bias,
            math.log(transition_prior / (1.0 - transition_prior)),
        )

        residual_feature_dim = feature_dim + int(self.use_transition_feature)
        self.residual_backbone = nn.Sequential(
            nn.Linear(residual_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.residual_head = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.vector_gate: nn.Linear | None = None
        if self.use_vector_gate:
            gate_init = float(model.get("residual_gate_init", 0.5))
            if not 0.0 < gate_init < 1.0:
                raise ValueError("model.residual_gate_init must be in (0, 1)")
            self.vector_gate = nn.Linear(residual_feature_dim, self.num_classes)
            nn.init.zeros_(self.vector_gate.weight)
            nn.init.constant_(
                self.vector_gate.bias,
                math.log(gate_init / (1.0 - gate_init)),
            )

    def _same_speaker_change(
        self,
        hidden: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length, hidden_dim = hidden.shape
        invalid = mask & ((speaker_ids < 0) | (speaker_ids >= self.max_speakers))
        if invalid.any():
            bad = torch.unique(speaker_ids[invalid]).detach().cpu().tolist()
            raise ValueError(
                f"speaker_ids {bad} exceed data.max_speakers={self.max_speakers}"
            )

        last_hidden = hidden.new_zeros(batch_size, self.max_speakers, hidden_dim)
        last_seen = torch.zeros(
            batch_size,
            self.max_speakers,
            dtype=torch.bool,
            device=hidden.device,
        )
        previous_steps: list[torch.Tensor] = []
        delta_steps: list[torch.Tensor] = []
        known_steps: list[torch.Tensor] = []
        safe_speakers = speaker_ids.clamp(0, self.max_speakers - 1)
        for turn in range(length):
            speaker = safe_speakers[:, turn]
            selector = torch.nn.functional.one_hot(
                speaker, num_classes=self.max_speakers
            ).to(hidden.dtype)
            selector = selector * mask[:, turn : turn + 1].to(hidden.dtype)
            previous = (last_hidden * selector.unsqueeze(-1)).sum(dim=1)
            seen = last_seen.gather(1, speaker.unsqueeze(-1)).squeeze(-1)
            known = mask[:, turn] & seen
            delta = torch.where(
                known.unsqueeze(-1), hidden[:, turn] - previous, torch.zeros_like(previous)
            )
            previous_steps.append(previous)
            delta_steps.append(delta)
            known_steps.append(known)

            keep = 1.0 - selector.unsqueeze(-1)
            last_hidden = (
                last_hidden * keep
                + hidden[:, turn].unsqueeze(1) * selector.unsqueeze(-1)
            )
            last_seen = last_seen | selector.bool()

        previous_same = torch.stack(previous_steps, dim=1)
        same_delta = torch.stack(delta_steps, dim=1)
        previous_known = torch.stack(known_steps, dim=1)
        return previous_same, same_delta, previous_known

    def _other_speaker_trigger(
        self,
        hidden: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, hidden_dim = hidden.shape
        trigger = hidden.new_zeros(batch_size, length, hidden_dim)
        known = torch.zeros(batch_size, length, dtype=torch.bool, device=hidden.device)
        if not self.use_trigger:
            return trigger, known
        assert self.trigger_query is not None
        assert self.trigger_key is not None
        assert self.trigger_value is not None
        queries = self.trigger_query(hidden)
        keys = self.trigger_key(hidden)
        values = self.trigger_value(hidden)
        pooled_steps: list[torch.Tensor] = []
        known_steps: list[torch.Tensor] = []
        for turn in range(length):
            if turn == 0:
                pooled_steps.append(hidden.new_zeros(batch_size, hidden_dim))
                known_steps.append(torch.zeros(batch_size, dtype=torch.bool, device=hidden.device))
                continue
            history_mask = (
                mask[:, :turn]
                & mask[:, turn : turn + 1]
                & speaker_ids[:, :turn].ne(speaker_ids[:, turn : turn + 1])
            )
            pooled, _ = masked_attention_pool(
                queries[:, turn], keys[:, :turn], values[:, :turn], history_mask
            )
            has_history = history_mask.any(dim=1)
            pooled_steps.append(
                pooled * has_history.unsqueeze(-1).to(pooled.dtype)
            )
            known_steps.append(has_history)
        trigger = torch.stack(pooled_steps, dim=1)
        known = torch.stack(known_steps, dim=1)
        return trigger, known

    def forward(
        self,
        hidden: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        anchor_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3 or anchor_logits.ndim != 3:
            raise ValueError("hidden and anchor_logits must be [batch, turns, features]")
        if hidden.shape[:2] != mask.shape or speaker_ids.shape != mask.shape:
            raise ValueError("hidden, speaker_ids and mask must share batch/turn dimensions")
        if anchor_logits.shape != (*mask.shape, self.num_classes):
            raise ValueError(
                f"Expected anchor_logits shape {(*mask.shape, self.num_classes)}, "
                f"got {tuple(anchor_logits.shape)}"
            )

        previous_same, same_delta, previous_known = self._same_speaker_change(
            hidden, speaker_ids, mask
        )
        other_trigger, trigger_known = self._other_speaker_trigger(
            hidden, speaker_ids, mask
        )
        features = [hidden, torch.softmax(anchor_logits, dim=-1)]
        if self.use_change:
            features.extend(
                [same_delta, previous_known.unsqueeze(-1).to(hidden.dtype)]
            )
        if self.use_trigger:
            features.extend(
                [other_trigger, trigger_known.unsqueeze(-1).to(hidden.dtype)]
            )
        detector_features = torch.cat(features, dim=-1)
        transition_logits = self.transition_head(detector_features).squeeze(-1)
        transition_probability = torch.sigmoid(transition_logits)

        residual_features = detector_features
        if self.use_transition_feature:
            residual_features = torch.cat(
                [residual_features, transition_probability.unsqueeze(-1)], dim=-1
            )
        raw_residual = self.residual_head(self.residual_backbone(residual_features))
        if self.vector_gate is None:
            class_gate = torch.ones_like(raw_residual)
        else:
            class_gate = torch.sigmoid(self.vector_gate(residual_features))

        valid = mask.unsqueeze(-1).to(hidden.dtype)
        raw_residual = raw_residual * valid
        class_gate = class_gate * valid
        applied_residual = self.residual_scale * class_gate * raw_residual
        final_logits = anchor_logits + applied_residual
        residual_expert_logits = anchor_logits + self.residual_scale * raw_residual
        return {
            "logits": final_logits,
            "residual_expert_logits": residual_expert_logits,
            "raw_class_residual": raw_residual,
            "class_residual": applied_residual,
            "class_gate": class_gate,
            "previous_same": previous_same * valid,
            "same_speaker_delta": same_delta * valid,
            "previous_same_mask": previous_known & mask,
            "other_speaker_trigger": other_trigger * valid,
            "other_speaker_trigger_mask": trigger_known & mask,
            "predicted_transition_logits": transition_logits * mask.to(
                transition_logits.dtype
            ),
            "predicted_transition_probability": transition_probability * mask.to(
                transition_probability.dtype
            ),
        }
