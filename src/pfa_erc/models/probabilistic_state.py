from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .scene_gate import SceneGate


class ProbabilisticEmotionStateTracker(nn.Module):
    """Causal event-conditioned emotion belief tracker for ERC.

    The model predicts turn ``t`` from the current-utterance anchor and states
    produced by turns ``< t``. Only after making that prediction does it update
    the story, speaker, and listener states with the current utterance. This
    ordering prevents future/current-label leakage into the contextual prior.
    """

    VALID_VARIANTS = {"anchor_only", "story_only", "speaker_state", "full"}

    def __init__(self, config: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        cfg = config.get("state_model", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        self.num_classes = int(num_classes)
        self.embedding_dim = int(cfg.get("embedding_dim", model_cfg.get("embedding_dim", 1024)))
        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.max_speakers = int(cfg.get("max_speakers", data_cfg.get("max_speakers", 16)))
        self.variant = str(cfg.get("variant", "full"))
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"state_model.variant must be one of {sorted(self.VALID_VARIANTS)}, "
                f"got {self.variant!r}"
            )
        self.anchor_source = str(cfg.get("anchor_source", "cached_logits"))
        if self.anchor_source not in {"cached_logits", "learned"}:
            raise ValueError("state_model.anchor_source must be 'cached_logits' or 'learned'")
        self.use_scene_gate = bool(cfg.get("use_scene_gate", False))

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
        context_dim = self.hidden_dim * 2 + self.num_classes
        self.context_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.transition_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        update_dim = self.hidden_dim * 3 + self.num_classes
        self.speaker_belief_gate = nn.Sequential(
            nn.Linear(update_dim, self.hidden_dim // 2), nn.GELU(), nn.Linear(self.hidden_dim // 2, 1)
        )
        self.listener_state_cell = nn.GRUCell(
            self.hidden_dim * 2 + self.num_classes, self.hidden_dim
        )
        self.listener_delta = nn.Sequential(
            nn.Linear(update_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.listener_gate = nn.Sequential(
            nn.Linear(update_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.scene_gate = (
            SceneGate(
                hidden_dim=self.hidden_dim,
                dropout=float(cfg.get("scene_gate_dropout", dropout)),
                init_probability=float(cfg.get("scene_gate_init", 0.05)),
                max_probability=float(cfg.get("scene_gate_max", 1.0)),
            )
            if self.use_scene_gate
            else None
        )

        context_scale_init = float(cfg.get("context_scale_init", 0.25))
        if not 0.0 < context_scale_init < 1.0:
            raise ValueError("state_model.context_scale_init must be in (0, 1)")
        self.context_scale_logit = nn.Parameter(
            torch.tensor(math.log(context_scale_init / (1.0 - context_scale_init)))
        )
        self.context_scale_max = float(cfg.get("context_scale_max", 1.0))
        if not 0.0 < self.context_scale_max <= 1.0:
            raise ValueError("state_model.context_scale_max must be in (0, 1]")

    @property
    def uses_cached_logits(self) -> bool:
        return self.anchor_source == "cached_logits"

    def _anchor(self, embeddings: torch.Tensor, current_logits: torch.Tensor | None) -> torch.Tensor:
        if self.uses_cached_logits:
            if current_logits is None:
                raise ValueError(
                    "This state model requires cached current_logits from the 3A checkpoint"
                )
            if current_logits.shape[-1] != self.num_classes:
                raise ValueError("current_logits class dimension does not match the dataset")
            return current_logits.float()
        return self.anchor_head(embeddings.float())

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
        seen = torch.zeros(batch_size, self.max_speakers, device=device, dtype=torch.bool)
        story_seen = torch.zeros(batch_size, device=device, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=device)
        previous_projected = torch.zeros(
            batch_size, self.hidden_dim, device=device, dtype=dtype
        )
        previous_speaker = torch.full(
            (batch_size,), -1, device=device, dtype=torch.long
        )
        last_seen_turn = torch.full(
            (batch_size, self.max_speakers), -1, device=device, dtype=torch.long
        )

        logits_by_turn: list[torch.Tensor] = []
        prior_by_turn: list[torch.Tensor] = []
        posterior_by_turn: list[torch.Tensor] = []
        transition_by_turn: list[torch.Tensor] = []
        influence_by_turn: list[torch.Tensor] = []
        context_by_turn: list[torch.Tensor] = []
        known_by_turn: list[torch.Tensor] = []
        scene_by_turn: list[torch.Tensor] = []

        use_context = self.variant != "anchor_only"
        use_story = self.variant in {"story_only", "speaker_state", "full"}
        use_speaker = self.variant in {"speaker_state", "full"}
        use_listener = self.variant == "full"
        context_scale = self.context_scale_max * torch.sigmoid(self.context_scale_logit)

        for turn in range(turns):
            valid = mask[:, turn]
            valid_f = valid.to(dtype).unsqueeze(-1)
            speaker = speaker_ids[:, turn].clamp(0, self.max_speakers - 1)
            x = projected[:, turn]

            # Backward-compatible default. Configurations without a scene gate
            # exactly retain the original recurrent story update.
            scene_prob = torch.zeros(
                batch_size, 1, device=device, dtype=dtype
            )
            if self.scene_gate is not None and use_story:
                speaker_change = (
                    story_seen
                    & previous_speaker.ne(speaker)
                    & valid
                ).to(dtype).unsqueeze(-1)
                current_last_seen = last_seen_turn[batch_indices, speaker]
                raw_gap = torch.where(
                    current_last_seen.ge(0),
                    torch.full_like(current_last_seen, turn) - current_last_seen,
                    torch.full_like(current_last_seen, turn + 1),
                )
                speaker_gap = (
                    torch.log1p(raw_gap.float())
                    / math.log1p(max(turns, 1))
                ).clamp(0.0, 1.0).to(dtype).unsqueeze(-1)
                scene_prob = self.scene_gate(
                    current=x,
                    previous=previous_projected,
                    story=story,
                    speaker_change=speaker_change,
                    speaker_gap=speaker_gap,
                )
                scene_available = story_seen & valid
                scene_prob = scene_prob * scene_available.to(dtype).unsqueeze(-1)

            # This line must remain outside the conditional above. It guarantees
            # a defined, zero-valued gate for disabled configurations and padding.
            scene_prob = torch.where(
                valid.unsqueeze(-1),
                scene_prob,
                torch.zeros_like(scene_prob),
            )

            current_state = speaker_states[batch_indices, speaker]
            current_belief = beliefs[batch_indices, speaker]
            speaker_known = seen[batch_indices, speaker] & valid

            story_context = (
                (1.0 - scene_prob) * story
                if use_story
                else torch.zeros_like(story)
            )
            speaker_context = current_state if use_speaker else torch.zeros_like(current_state)
            belief_context = (
                torch.where(speaker_known.unsqueeze(-1), current_belief, uniform[:, 0])
                if use_speaker
                else uniform[:, 0]
            )
            context_features = torch.cat(
                [story_context, speaker_context, belief_context], dim=-1
            )
            context_logits = self.context_head(context_features)
            context_available = (story_seen | speaker_known) & valid
            if use_context:
                fused = anchor_logits[:, turn] + (
                    context_scale * context_available.to(dtype).unsqueeze(-1) * context_logits
                )
            else:
                fused = anchor_logits[:, turn]
            fused = torch.where(valid.unsqueeze(-1), fused, torch.zeros_like(fused))
            posterior = torch.softmax(fused.float(), dim=-1).to(dtype)
            transition_logit = self.transition_head(context_features).squeeze(-1)

            logits_by_turn.append(fused)
            prior_by_turn.append(belief_context)
            posterior_by_turn.append(posterior)
            transition_by_turn.append(transition_logit)
            context_by_turn.append(context_logits)
            known_by_turn.append(speaker_known)
            scene_by_turn.append(scene_prob.squeeze(-1))

            if use_story:
                continued_story = self.story_cell(x, story)
                if self.scene_gate is not None:
                    reset_story = self.story_cell(x, torch.zeros_like(story))
                    story_candidate = (
                        (1.0 - scene_prob) * continued_story
                        + scene_prob * reset_story
                    )
                else:
                    story_candidate = continued_story
            else:
                story_candidate = story
            next_story = torch.where(valid.unsqueeze(-1), story_candidate, story)
            update_features = torch.cat([x, next_story, current_state, posterior], dim=-1)
            speaker_gate = torch.sigmoid(self.speaker_belief_gate(update_features))
            if not use_speaker:
                speaker_gate = torch.ones_like(speaker_gate)
            next_speaker_belief = (
                (1.0 - speaker_gate) * belief_context + speaker_gate * posterior
            )
            next_speaker_belief = next_speaker_belief / next_speaker_belief.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            speaker_cell_input = torch.cat([x, next_story, posterior], dim=-1)
            next_speaker_state = (
                self.speaker_cell(speaker_cell_input, current_state)
                if use_speaker
                else current_state
            )

            next_states = speaker_states
            next_beliefs = beliefs
            listener_gate_mean = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            if use_listener:
                expanded_x = x.unsqueeze(1).expand(-1, self.max_speakers, -1)
                expanded_story = next_story.unsqueeze(1).expand(-1, self.max_speakers, -1)
                listener_update = torch.cat(
                    [expanded_x, expanded_story, speaker_states, beliefs], dim=-1
                )
                flat_input = torch.cat(
                    [expanded_x, expanded_story, beliefs], dim=-1
                ).reshape(-1, self.hidden_dim * 2 + self.num_classes)
                listener_state_candidate = self.listener_state_cell(
                    flat_input, speaker_states.reshape(-1, self.hidden_dim)
                ).view(batch_size, self.max_speakers, self.hidden_dim)
                listener_gate_values = torch.sigmoid(self.listener_gate(listener_update))
                listener_logits = torch.log(beliefs.clamp_min(1e-8)) + self.listener_delta(
                    listener_update
                )
                listener_candidate = torch.softmax(listener_logits.float(), dim=-1).to(dtype)
                listener_belief_candidate = (
                    (1.0 - listener_gate_values) * beliefs
                    + listener_gate_values * listener_candidate
                )
                speaker_slots = torch.arange(self.max_speakers, device=device).unsqueeze(0)
                listener_mask = (
                    seen
                    & speaker_slots.ne(speaker.unsqueeze(1))
                    & valid.unsqueeze(1)
                )
                next_states = torch.where(
                    listener_mask.unsqueeze(-1), listener_state_candidate, speaker_states
                )
                next_beliefs = torch.where(
                    listener_mask.unsqueeze(-1), listener_belief_candidate, beliefs
                )
                listener_count = listener_mask.sum(dim=1, keepdim=True).clamp_min(1)
                listener_gate_mean = (
                    listener_gate_values.squeeze(-1) * listener_mask.to(dtype)
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
            speaker_states = next_states
            beliefs = next_beliefs
            seen = seen | current_slot
            story = next_story
            story_seen = story_seen | valid
            previous_projected = torch.where(
                valid.unsqueeze(-1), x, previous_projected
            )
            previous_speaker = torch.where(valid, speaker, previous_speaker)
            turn_value = torch.full(
                (batch_size, 1), turn, device=device, dtype=torch.long
            )
            last_seen_turn = torch.where(
                current_slot,
                turn_value.expand(-1, self.max_speakers),
                last_seen_turn,
            )
            influence_by_turn.append(
                torch.where(valid_f.bool(), speaker_gate + listener_gate_mean, torch.zeros_like(speaker_gate))
                / (2.0 if use_listener else 1.0)
            )

        logits = torch.stack(logits_by_turn, dim=1)
        if not return_aux:
            return logits
        aux = {
            "anchor_logits": anchor_logits,
            "context_logits": torch.stack(context_by_turn, dim=1),
            "prior_emotion_distribution": torch.stack(prior_by_turn, dim=1),
            "posterior_emotion_distribution": torch.stack(posterior_by_turn, dim=1),
            "transition_logits": torch.stack(transition_by_turn, dim=1),
            "influence_gate": torch.stack(influence_by_turn, dim=1),
            "state_known_mask": torch.stack(known_by_turn, dim=1),
            "scene_transition_probability": torch.stack(scene_by_turn, dim=1),
            "context_scale": context_scale.expand(batch_size),
        }
        return logits, aux
