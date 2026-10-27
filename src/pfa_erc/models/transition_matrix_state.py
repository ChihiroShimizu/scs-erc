from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


class EventTransitionMatrixTracker(nn.Module):
    """Causal ERC model with event-conditioned emotion transition matrices.

    Each event predicts a row-stochastic CxC matrix. Speaker beliefs are
    propagated through those matrices without making an intermediate argmax.
    Stable/transition is never supplied as an input or routing decision; its
    probability is derived from the composed transition operator.
    """

    VALID_VARIANTS = {
        "matrix_speaker",
        "matrix_listener_no_uncertainty",
        "matrix_listener",
    }

    def __init__(self, config: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        cfg = config.get("state_model", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        self.num_classes = int(num_classes)
        self.embedding_dim = int(
            cfg.get("embedding_dim", model_cfg.get("embedding_dim", 1024))
        )
        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.max_speakers = int(
            cfg.get("max_speakers", data_cfg.get("max_speakers", 16))
        )
        self.variant = str(cfg.get("variant", "matrix_listener"))
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"state_model.variant must be one of {sorted(self.VALID_VARIANTS)}, "
                f"got {self.variant!r}"
            )
        self.anchor_source = str(cfg.get("anchor_source", "cached_logits"))
        if self.anchor_source not in {"cached_logits", "learned"}:
            raise ValueError("state_model.anchor_source must be cached_logits or learned")

        self.use_listener = self.variant != "matrix_speaker"
        self.uncertainty_aware = self.variant != "matrix_listener_no_uncertainty"
        self.speaker_update_max = float(cfg.get("speaker_update_max", 0.50))
        self.listener_update_max = float(cfg.get("listener_update_max", 0.15))
        for name, value in (
            ("speaker_update_max", self.speaker_update_max),
            ("listener_update_max", self.listener_update_max),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"state_model.{name} must be in (0, 1]")

        dropout = float(cfg.get("dropout", 0.2))
        self.input_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.anchor_head = nn.Linear(self.embedding_dim, self.num_classes)
        self.story_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.speaker_cell = nn.GRUCell(
            self.hidden_dim * 2 + self.num_classes, self.hidden_dim
        )
        self.listener_state_cell = nn.GRUCell(
            self.hidden_dim * 2 + self.num_classes, self.hidden_dim
        )

        context_dim = self.hidden_dim * 2 + self.num_classes
        self.context_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(context_dim + self.num_classes),
            nn.Linear(context_dim + self.num_classes, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        update_dim = self.hidden_dim * 3 + self.num_classes
        self.speaker_matrix_head = nn.Sequential(
            nn.LayerNorm(update_dim),
            nn.Linear(update_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes * self.num_classes),
        )
        self.listener_matrix_head = nn.Sequential(
            nn.LayerNorm(update_dim),
            nn.Linear(update_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes * self.num_classes),
        )
        self.speaker_update_gate = nn.Sequential(
            nn.Linear(update_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.listener_update_gate = nn.Sequential(
            nn.Linear(update_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        context_scale_init = float(cfg.get("context_scale_init", 0.20))
        if not 0.0 < context_scale_init < 1.0:
            raise ValueError("state_model.context_scale_init must be in (0, 1)")
        self.context_scale_logit = nn.Parameter(torch.tensor(_logit(context_scale_init)))
        self.context_scale_max = float(cfg.get("context_scale_max", 0.35))
        if not 0.0 < self.context_scale_max <= 1.0:
            raise ValueError("state_model.context_scale_max must be in (0, 1]")
        self.uncertainty_floor = float(cfg.get("uncertainty_floor", 0.10))
        if not 0.0 <= self.uncertainty_floor <= 1.0:
            raise ValueError("state_model.uncertainty_floor must be in [0, 1]")

        identity_bias = float(cfg.get("transition_identity_bias", 3.0))
        self.register_buffer(
            "identity_logits",
            torch.eye(self.num_classes) * identity_bias,
            persistent=True,
        )
        self.register_buffer(
            "identity_matrix", torch.eye(self.num_classes), persistent=False
        )
        self._initialize_safe_residuals(cfg)

    def _initialize_safe_residuals(self, cfg: dict[str, Any]) -> None:
        for head in (self.context_head, self.speaker_matrix_head, self.listener_matrix_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.constant_(
            self.reliability_head[-1].bias,
            _logit(float(cfg.get("reliability_init", 0.50))),
        )
        speaker_rate = float(cfg.get("speaker_update_init", 0.10))
        listener_rate = float(cfg.get("listener_update_init", 0.03))
        if not 0.0 < speaker_rate < self.speaker_update_max:
            raise ValueError("speaker_update_init must be in (0, speaker_update_max)")
        if not 0.0 < listener_rate < self.listener_update_max:
            raise ValueError("listener_update_init must be in (0, listener_update_max)")
        nn.init.zeros_(self.speaker_update_gate[-1].weight)
        nn.init.constant_(
            self.speaker_update_gate[-1].bias,
            _logit(speaker_rate / self.speaker_update_max),
        )
        nn.init.zeros_(self.listener_update_gate[-1].weight)
        nn.init.constant_(
            self.listener_update_gate[-1].bias,
            _logit(listener_rate / self.listener_update_max),
        )

    @property
    def uses_cached_logits(self) -> bool:
        return self.anchor_source == "cached_logits"

    def _anchor(
        self, embeddings: torch.Tensor, current_logits: torch.Tensor | None
    ) -> torch.Tensor:
        if self.uses_cached_logits:
            if current_logits is None:
                raise ValueError("Transition-matrix experiments require cached current_logits")
            if current_logits.shape[-1] != self.num_classes:
                raise ValueError("current_logits class dimension does not match the dataset")
            return current_logits.float()
        return self.anchor_head(embeddings.float())

    def _matrix(
        self, raw: torch.Tensor, update_rate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*raw.shape[:-1], self.num_classes, self.num_classes)
        logits = raw.view(shape) + self.identity_logits.to(raw.dtype)
        transition = torch.softmax(logits.float(), dim=-1).to(raw.dtype)
        identity = self.identity_matrix.to(raw.dtype)
        while identity.ndim < transition.ndim:
            identity = identity.unsqueeze(0)
        effective = (1.0 - update_rate.unsqueeze(-1)) * identity + update_rate.unsqueeze(
            -1
        ) * transition
        return transition, effective

    @staticmethod
    def _apply_distribution(
        distribution: torch.Tensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(distribution.unsqueeze(-2), matrix).squeeze(-2)

    def forward(
        self,
        embeddings: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        current_logits: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if embeddings.ndim != 3 or speaker_ids.shape != embeddings.shape[:2]:
            raise ValueError("Expected embeddings [B,T,D] and speaker_ids [B,T]")
        if mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must have shape [B,T]")
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {embeddings.shape[-1]}"
            )
        if torch.any((speaker_ids[mask] < 0) | (speaker_ids[mask] >= self.max_speakers)):
            raise ValueError("A valid speaker id is outside state_model.max_speakers")

        batch_size, turns, _ = embeddings.shape
        device = embeddings.device
        projected = self.input_projection(embeddings.float())
        dtype = projected.dtype
        anchor_logits = self._anchor(embeddings, current_logits).to(dtype)
        story = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        speaker_states = torch.zeros(
            batch_size, self.max_speakers, self.hidden_dim, device=device, dtype=dtype
        )
        uniform = torch.full(
            (batch_size, self.max_speakers, self.num_classes),
            1.0 / self.num_classes,
            device=device,
            dtype=dtype,
        )
        beliefs = uniform
        last_own_beliefs = uniform.clone()
        identity = self.identity_matrix.to(device=device, dtype=dtype)
        operators = identity.view(1, 1, self.num_classes, self.num_classes).expand(
            batch_size, self.max_speakers, -1, -1
        ).clone()
        seen = torch.zeros(batch_size, self.max_speakers, device=device, dtype=torch.bool)
        story_seen = torch.zeros(batch_size, device=device, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=device)

        output_logits: list[torch.Tensor] = []
        prior_distributions: list[torch.Tensor] = []
        posterior_distributions: list[torch.Tensor] = []
        context_outputs: list[torch.Tensor] = []
        change_logits: list[torch.Tensor] = []
        change_probabilities: list[torch.Tensor] = []
        fusion_gates: list[torch.Tensor] = []
        update_gates: list[torch.Tensor] = []
        known_masks: list[torch.Tensor] = []
        event_matrices: list[torch.Tensor] = []

        base_context_scale = self.context_scale_max * torch.sigmoid(
            self.context_scale_logit
        )
        log_classes = math.log(self.num_classes)

        for turn in range(turns):
            valid = mask[:, turn]
            speaker = speaker_ids[:, turn].clamp(0, self.max_speakers - 1)
            current_state = speaker_states[batch_indices, speaker]
            current_belief = beliefs[batch_indices, speaker]
            current_last_own = last_own_beliefs[batch_indices, speaker]
            current_operator = operators[batch_indices, speaker]
            speaker_known = seen[batch_indices, speaker] & valid
            belief_context = torch.where(
                speaker_known.unsqueeze(-1), current_belief, uniform[:, 0]
            )

            context_features = torch.cat([story, current_state, belief_context], dim=-1)
            context_logits = self.context_head(context_features)
            reliability = torch.sigmoid(
                self.reliability_head(
                    torch.cat([context_features, anchor_logits[:, turn].detach()], dim=-1)
                )
            )
            entropy = -(
                belief_context.float()
                * torch.log(belief_context.float().clamp_min(1e-8))
            ).sum(dim=-1, keepdim=True)
            confidence = (1.0 - entropy / log_classes).clamp(0.0, 1.0).to(dtype)
            uncertainty_factor = (
                self.uncertainty_floor
                + (1.0 - self.uncertainty_floor) * confidence
                if self.uncertainty_aware
                else torch.ones_like(confidence)
            )
            context_available = (story_seen | speaker_known) & valid
            fusion_gate = (
                base_context_scale
                * reliability
                * uncertainty_factor
                * context_available.to(dtype).unsqueeze(-1)
            )
            fused = anchor_logits[:, turn] + fusion_gate * context_logits
            fused = torch.where(valid.unsqueeze(-1), fused, torch.zeros_like(fused))
            posterior = torch.softmax(fused.float(), dim=-1).to(dtype)

            stable_probability = (
                current_last_own
                * torch.diagonal(current_operator, dim1=-2, dim2=-1)
            ).sum(dim=-1)
            change_probability = (1.0 - stable_probability).clamp(1e-5, 1.0 - 1e-5)
            change_probability = torch.where(
                speaker_known, change_probability, torch.full_like(change_probability, 0.5)
            )
            change_logit = torch.logit(change_probability.float()).to(dtype)

            output_logits.append(fused)
            prior_distributions.append(belief_context)
            posterior_distributions.append(posterior)
            context_outputs.append(context_logits)
            change_logits.append(change_logit)
            change_probabilities.append(change_probability)
            fusion_gates.append(fusion_gate)
            known_masks.append(speaker_known)

            x = projected[:, turn]
            story_candidate = self.story_cell(x, story)
            next_story = torch.where(valid.unsqueeze(-1), story_candidate, story)
            speaker_update_features = torch.cat(
                [x, next_story, current_state, posterior], dim=-1
            )
            speaker_rate = self.speaker_update_max * torch.sigmoid(
                self.speaker_update_gate(speaker_update_features)
            )
            _, speaker_effective_matrix = self._matrix(
                self.speaker_matrix_head(speaker_update_features), speaker_rate
            )
            next_speaker_belief = self._apply_distribution(
                posterior, speaker_effective_matrix
            )
            next_speaker_state = self.speaker_cell(
                torch.cat([x, next_story, posterior], dim=-1), current_state
            )

            next_states = speaker_states
            next_beliefs = beliefs
            next_last_own = last_own_beliefs
            next_operators = operators
            listener_gate_mean = torch.zeros(
                batch_size, 1, device=device, dtype=dtype
            )
            if self.use_listener:
                expanded_x = x.unsqueeze(1).expand(-1, self.max_speakers, -1)
                expanded_story = next_story.unsqueeze(1).expand(
                    -1, self.max_speakers, -1
                )
                listener_features = torch.cat(
                    [expanded_x, expanded_story, speaker_states, beliefs], dim=-1
                )
                listener_rate = self.listener_update_max * torch.sigmoid(
                    self.listener_update_gate(listener_features)
                )
                _, listener_effective_matrix = self._matrix(
                    self.listener_matrix_head(listener_features), listener_rate
                )
                listener_belief_candidate = self._apply_distribution(
                    beliefs, listener_effective_matrix
                )
                listener_operator_candidate = torch.matmul(
                    operators, listener_effective_matrix
                )
                flat_listener_input = torch.cat(
                    [expanded_x, expanded_story, beliefs], dim=-1
                ).reshape(-1, self.hidden_dim * 2 + self.num_classes)
                listener_state_candidate = self.listener_state_cell(
                    flat_listener_input, speaker_states.reshape(-1, self.hidden_dim)
                ).view(batch_size, self.max_speakers, self.hidden_dim)
                slots = torch.arange(self.max_speakers, device=device).unsqueeze(0)
                listener_mask = (
                    seen & slots.ne(speaker.unsqueeze(1)) & valid.unsqueeze(1)
                )
                next_states = torch.where(
                    listener_mask.unsqueeze(-1), listener_state_candidate, next_states
                )
                next_beliefs = torch.where(
                    listener_mask.unsqueeze(-1), listener_belief_candidate, next_beliefs
                )
                next_operators = torch.where(
                    listener_mask.unsqueeze(-1).unsqueeze(-1),
                    listener_operator_candidate,
                    next_operators,
                )
                listener_count = listener_mask.sum(dim=1, keepdim=True).clamp_min(1)
                listener_gate_mean = (
                    listener_rate.squeeze(-1) * listener_mask.to(dtype)
                ).sum(dim=1, keepdim=True) / listener_count.to(dtype)

            current_slot = torch.nn.functional.one_hot(
                speaker, num_classes=self.max_speakers
            ).bool() & valid.unsqueeze(-1)
            next_states = torch.where(
                current_slot.unsqueeze(-1), next_speaker_state.unsqueeze(1), next_states
            )
            next_beliefs = torch.where(
                current_slot.unsqueeze(-1), next_speaker_belief.unsqueeze(1), next_beliefs
            )
            next_last_own = torch.where(
                current_slot.unsqueeze(-1), posterior.unsqueeze(1), next_last_own
            )
            next_operators = torch.where(
                current_slot.unsqueeze(-1).unsqueeze(-1),
                speaker_effective_matrix.unsqueeze(1),
                next_operators,
            )
            speaker_states = next_states
            beliefs = next_beliefs
            last_own_beliefs = next_last_own
            operators = next_operators
            seen = seen | current_slot
            story = next_story
            story_seen = story_seen | valid
            update_gate = (
                speaker_rate + listener_gate_mean
            ) / (2.0 if self.use_listener else 1.0)
            update_gates.append(
                torch.where(valid.unsqueeze(-1), update_gate, torch.zeros_like(update_gate))
            )
            event_matrices.append(speaker_effective_matrix)

        logits = torch.stack(output_logits, dim=1)
        if not return_aux:
            return logits
        aux = {
            "anchor_logits": anchor_logits,
            "context_logits": torch.stack(context_outputs, dim=1),
            "prior_emotion_distribution": torch.stack(prior_distributions, dim=1),
            "posterior_emotion_distribution": torch.stack(
                posterior_distributions, dim=1
            ),
            "transition_logits": torch.stack(change_logits, dim=1),
            "transition_probability": torch.stack(change_probabilities, dim=1),
            "influence_gate": torch.stack(fusion_gates, dim=1),
            "state_update_gate": torch.stack(update_gates, dim=1),
            "state_known_mask": torch.stack(known_masks, dim=1),
            "event_transition_matrix": torch.stack(event_matrices, dim=1),
            "context_scale": base_context_scale.expand(batch_size),
        }
        return logits, aux
