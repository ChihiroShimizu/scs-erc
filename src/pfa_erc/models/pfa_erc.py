from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .class_residual import ClassSpecificResidualCorrector
from .directed_affect import DirectedProbabilisticAffectUpdater
from .hitft import HiTFTForecaster
from .speaker_encoder import SpeakerEmotionEncoder


class EmotionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.match_context = nn.Linear(hidden_dim, hidden_dim)
        self.emotion_embedding = nn.Embedding(num_classes, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.output_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        context: torch.Tensor,
        speaker_state: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
    ) -> torch.Tensor:
        batch_size, length, hidden = context.shape
        recurrent = context.new_zeros(batch_size, hidden)
        previous_distribution = context.new_zeros(batch_size, self.num_classes)
        logits: list[torch.Tensor] = []
        for t in range(length):
            match = F.relu(speaker_state[:, t] * self.match_context(context[:, t]))
            if t == 0:
                previous_emotion = context.new_zeros(batch_size, hidden)
            else:
                soft_emotion = previous_distribution @ self.emotion_embedding.weight
                if self.training and labels is not None and teacher_forcing > 0:
                    teacher_mask = torch.rand(batch_size, device=context.device) < teacher_forcing
                    valid_teacher = labels[:, t - 1].ge(0) & mask[:, t - 1]
                    teacher_mask = teacher_mask & valid_teacher
                    teacher_emotion = self.emotion_embedding(labels[:, t - 1].clamp_min(0))
                    previous_emotion = torch.where(
                        teacher_mask.unsqueeze(-1), teacher_emotion, soft_emotion
                    )
                else:
                    previous_emotion = soft_emotion
            candidate = self.gru(torch.cat([match, previous_emotion], dim=-1), recurrent)
            recurrent = torch.where(mask[:, t : t + 1], candidate, recurrent)
            hidden_output = F.relu(
                self.output_hidden(torch.cat([context[:, t], recurrent], dim=-1))
            )
            step_logits = self.classifier(self.dropout(hidden_output))
            logits.append(step_logits)
            previous_distribution = torch.softmax(step_logits, dim=-1)
        return torch.stack(logits, dim=1)


class PFAERC(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        num_classes: int,
        hitft: HiTFTForecaster | None = None,
    ) -> None:
        super().__init__()
        model = config["model"]
        self.embedding_dim = int(model["embedding_dim"])
        self.hidden_dim = int(model["hidden_dim"])
        self.num_classes = num_classes
        self.future_steps = int(model["future_steps"])
        self.history_steps = int(model.get("history_steps", 8))
        self.use_hitft = bool(model.get("use_hitft", True))
        self.use_speaker_encoder = bool(model.get("use_speaker_encoder", True))
        self.use_decoder = bool(model.get("use_decoder", True))
        dropout = float(model.get("dropout", 0.3))

        if self.use_hitft and hitft is None:
            raise ValueError("A pretrained HiTFT instance is required when model.use_hitft=true")
        self.hitft = hitft
        if self.hitft is not None:
            self.hitft.requires_grad_(False)
            self.hitft.eval()

        self.future_fusion_name = str(model.get("future_fusion", "flatten_project"))
        self.current_only = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim), nn.ReLU(), nn.LayerNorm(self.hidden_dim)
        )
        self.future_fusion: nn.Module | None = None
        self.future_projection: nn.Module | None = None
        self.stable_future_projection: nn.Module | None = None
        self.transition_future_projection: nn.Module | None = None
        self.future_gate: nn.Linear | None = None
        self.future_fusion_norm: nn.Module | None = None
        if self.future_fusion_name == "flatten_project":
            self.future_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim * (1 + self.future_steps), self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.hidden_dim),
            )
        elif self.future_fusion_name == "residual_gated":
            self.future_projection = nn.Sequential(
                nn.Linear(self.embedding_dim * self.future_steps, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.hidden_dim),
            )
            # One interpretable gate per utterance. Starting near 0.1 keeps the
            # initial model close to the current-only Enc+Dec baseline instead
            # of letting four smooth future vectors dominate the representation.
            initial_gate = float(model.get("future_gate_init", 0.1))
            if not 0.0 < initial_gate < 1.0:
                raise ValueError("model.future_gate_init must be strictly between 0 and 1")
            self.future_gate = nn.Linear(self.hidden_dim * 2, 1)
            nn.init.zeros_(self.future_gate.weight)
            nn.init.constant_(
                self.future_gate.bias,
                math.log(initial_gate / (1.0 - initial_gate)),
            )
            self.future_fusion_norm = nn.LayerNorm(self.hidden_dim)
        elif self.future_fusion_name == "transition_gated":
            forecaster_mode = getattr(self.hitft, "forecaster_mode", "absolute")
            if forecaster_mode not in {"transition_aware", "transition_residual"}:
                raise ValueError(
                    "transition_gated fusion requires a transition-aware forecaster mode"
                )
            self.stable_future_projection = nn.Sequential(
                nn.Linear(self.embedding_dim * self.future_steps, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.hidden_dim),
            )
            self.transition_future_projection = nn.Sequential(
                nn.Linear(self.embedding_dim * self.future_steps, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.hidden_dim),
            )
            initial_gate = float(model.get("future_gate_init", 0.1))
            if not 0.0 < initial_gate < 1.0:
                raise ValueError("model.future_gate_init must be strictly between 0 and 1")
            self.future_gate = nn.Linear(self.hidden_dim * 2 + 1, 1)
            nn.init.zeros_(self.future_gate.weight)
            nn.init.constant_(
                self.future_gate.bias,
                math.log(initial_gate / (1.0 - initial_gate)),
            )
            self.future_fusion_norm = nn.LayerNorm(self.hidden_dim)
        else:
            raise ValueError(f"Unsupported future_fusion: {self.future_fusion_name}")
        self.speaker_encoder = SpeakerEmotionEncoder(config) if self.use_speaker_encoder else None
        self.decoder = (
            EmotionDecoder(self.hidden_dim, num_classes, dropout)
            if self.use_decoder
            else None
        )
        self.direct_classifier = (
            None
            if self.use_decoder
            else nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, num_classes),
            )
        )

        self.logit_fusion_name = str(model.get("logit_fusion", "none"))
        self.use_directed_update = (
            self.logit_fusion_name == "directed_probabilistic_update"
        )
        self.use_class_residual = (
            self.logit_fusion_name == "current_anchored_class_residual"
        )
        self.use_current_logit_anchor = self.logit_fusion_name in {
            "current_anchored_transition_gate",
            "current_anchored_class_residual",
            "directed_probabilistic_update",
        }
        self.context_gate_mode = str(model.get("context_gate_mode", "transition"))
        self.transition_gate_power = float(model.get("transition_gate_power", 1.0))
        self.context_gate_max = float(model.get("context_gate_max", 1.0))
        self.context_gate_fixed_value = float(model.get("context_gate_init", 0.1))
        self.context_helpfulness_margin = float(
            model.get("context_helpfulness_margin", 0.0)
        )
        self.helpfulness_prior = float(model.get("helpfulness_prior", 0.55))
        self.helpfulness_gate_power = float(
            model.get("helpfulness_gate_power", 1.0)
        )
        self.context_logit_gate: nn.Linear | None = None
        self.transition_logit_head: nn.Linear | None = None
        self.helpfulness_logit_head: nn.Linear | None = None
        self.class_residual_corrector: ClassSpecificResidualCorrector | None = None
        self.directed_affect_updater: DirectedProbabilisticAffectUpdater | None = None
        self.uses_transition_suppression = self.context_gate_mode in {
            "transition",
            "transition_helpfulness",
        }
        self.uses_helpfulness_gate = self.context_gate_mode in {
            "helpfulness",
            "helpfulness_v2",
            "transition_helpfulness",
        }
        if self.logit_fusion_name not in {
            "none",
            "current_anchored_transition_gate",
            "current_anchored_class_residual",
            "directed_probabilistic_update",
        }:
            raise ValueError(f"Unsupported model.logit_fusion: {self.logit_fusion_name}")
        if self.use_directed_update:
            self.context_gate_mode = "directed_probabilistic"
            self.uses_transition_suppression = False
            self.uses_helpfulness_gate = False
            self.directed_affect_updater = DirectedProbabilisticAffectUpdater(
                config, num_classes
            )
        elif self.use_class_residual:
            self.context_gate_mode = "class_residual"
            self.uses_transition_suppression = False
            self.uses_helpfulness_gate = False
            self.class_residual_corrector = ClassSpecificResidualCorrector(
                config, num_classes
            )
        elif self.use_current_logit_anchor:
            desired_gate = float(model.get("context_gate_init", 0.1))
            transition_prior = float(model.get("transition_prior", 0.54))
            allowed_gate_modes = {
                "fixed",
                "learned",
                "transition",
                "helpfulness",
                "helpfulness_v2",
                "transition_helpfulness",
            }
            if self.context_gate_mode not in allowed_gate_modes:
                raise ValueError(
                    "model.context_gate_mode must be one of "
                    f"{sorted(allowed_gate_modes)}, got {self.context_gate_mode!r}"
                )
            if not 0.0 < transition_prior < 1.0:
                raise ValueError("model.transition_prior must be strictly between 0 and 1")
            if not 0.0 < desired_gate < 1.0:
                raise ValueError("model.context_gate_init must be strictly between 0 and 1")
            if not 0.0 < self.context_gate_max <= 1.0:
                raise ValueError("model.context_gate_max must be in (0, 1]")
            if self.transition_gate_power <= 0.0:
                raise ValueError("model.transition_gate_power must be positive")
            if self.context_helpfulness_margin < 0.0:
                raise ValueError("model.context_helpfulness_margin cannot be negative")
            if not 0.0 < self.helpfulness_prior < 1.0:
                raise ValueError("model.helpfulness_prior must be strictly between 0 and 1")
            if self.helpfulness_gate_power <= 0.0:
                raise ValueError("model.helpfulness_gate_power must be positive")
            gate_features = self.hidden_dim * 2 + num_classes
            self.transition_logit_head = nn.Linear(gate_features, 1)
            nn.init.zeros_(self.transition_logit_head.weight)
            nn.init.constant_(
                self.transition_logit_head.bias,
                math.log(transition_prior / (1.0 - transition_prior)),
            )
            if self.context_gate_mode == "fixed" and desired_gate > self.context_gate_max:
                raise ValueError(
                    "model.context_gate_init cannot exceed model.context_gate_max"
                )
            if self.context_gate_mode == "learned":
                base_gate = desired_gate / self.context_gate_max
                if not 0.0 < base_gate < 1.0:
                    raise ValueError(
                        "model.context_gate_init must be smaller than "
                        "model.context_gate_max for a learned gate"
                    )
                self.context_logit_gate = nn.Linear(gate_features, 1)
                nn.init.zeros_(self.context_logit_gate.weight)
                nn.init.constant_(
                    self.context_logit_gate.bias,
                    math.log(base_gate / (1.0 - base_gate)),
                )
            elif self.context_gate_mode == "transition":
                suppression_at_prior = (1.0 - transition_prior) ** self.transition_gate_power
                base_gate = desired_gate / (
                    self.context_gate_max * suppression_at_prior
                )
                if not 0.0 < base_gate < 1.0:
                    raise ValueError(
                        "model.context_gate_init is too large for the configured "
                        "context_gate_max, transition_prior and transition_gate_power; "
                        "reduce the initial gate or the suppression power, or increase "
                        "the gate maximum"
                    )
                self.context_logit_gate = nn.Linear(gate_features, 1)
                nn.init.zeros_(self.context_logit_gate.weight)
                nn.init.constant_(
                    self.context_logit_gate.bias,
                    math.log(base_gate / (1.0 - base_gate)),
                )
            elif self.context_gate_mode == "helpfulness":
                base_gate = desired_gate / self.context_gate_max
                if not 0.0 < base_gate < 1.0:
                    raise ValueError(
                        "model.context_gate_init must be smaller than "
                        "model.context_gate_max for a helpfulness gate"
                    )
                self.helpfulness_logit_head = nn.Linear(gate_features, 1)
                nn.init.zeros_(self.helpfulness_logit_head.weight)
                nn.init.constant_(
                    self.helpfulness_logit_head.bias,
                    math.log(base_gate / (1.0 - base_gate)),
                )
            elif self.context_gate_mode in {
                "helpfulness_v2",
                "transition_helpfulness",
            }:
                transition_factor = 1.0
                if self.context_gate_mode == "transition_helpfulness":
                    transition_factor = (
                        1.0 - transition_prior
                    ) ** self.transition_gate_power
                helpfulness_factor = (
                    self.helpfulness_prior ** self.helpfulness_gate_power
                )
                base_gate = desired_gate / (
                    self.context_gate_max
                    * transition_factor
                    * helpfulness_factor
                )
                if not 0.0 < base_gate < 1.0:
                    raise ValueError(
                        "model.context_gate_init is unreachable for the configured "
                        "transition/helpfulness priors and gate powers; reduce the "
                        "initial gate or increase context_gate_max"
                    )
                self.context_logit_gate = nn.Linear(gate_features, 1)
                nn.init.zeros_(self.context_logit_gate.weight)
                nn.init.constant_(
                    self.context_logit_gate.bias,
                    math.log(base_gate / (1.0 - base_gate)),
                )
                # The detector sees both class distributions and the predicted
                # transition probability. Gold labels are deliberately absent;
                # they are used only to create an auxiliary training target.
                helpfulness_features = gate_features + num_classes + 1
                self.helpfulness_logit_head = nn.Linear(helpfulness_features, 1)
                nn.init.zeros_(self.helpfulness_logit_head.weight)
                nn.init.constant_(
                    self.helpfulness_logit_head.bias,
                    math.log(
                        self.helpfulness_prior / (1.0 - self.helpfulness_prior)
                    ),
                )

    def train(self, mode: bool = True) -> "PFAERC":
        super().train(mode)
        if self.hitft is not None:
            self.hitft.eval()
        return self

    def _history_windows(
        self, embeddings: torch.Tensor, speakers: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length, dim = embeddings.shape
        history = embeddings.new_zeros(batch_size, length, self.history_steps, dim)
        history_speakers = speakers.new_zeros(batch_size, length, self.history_steps)
        history_mask = mask.new_zeros(batch_size, length, self.history_steps)
        for t in range(length):
            start = max(0, t - self.history_steps + 1)
            count = t - start + 1
            history[:, t, -count:] = embeddings[:, start : t + 1]
            history_speakers[:, t, -count:] = speakers[:, start : t + 1]
            history_mask[:, t, -count:] = mask[:, start : t + 1]
        flat_shape = (batch_size * length, self.history_steps)
        return (
            history.reshape(batch_size * length, self.history_steps, dim),
            history_speakers.reshape(flat_shape),
            history_mask.reshape(flat_shape),
        )

    def _pseudo_future_details(
        self, embeddings: torch.Tensor, speakers: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch_size, length, _ = embeddings.shape
        empty_future = embeddings.new_zeros(
            batch_size, length, self.future_steps, self.embedding_dim
        )
        empty_probability = embeddings.new_zeros(batch_size, length, self.future_steps)
        details = {
            "prediction": empty_future,
            "stable_prediction": empty_future.clone(),
            "transition_prediction": empty_future.clone(),
            "transition_probability": empty_probability,
        }
        if self.hitft is None:
            return details
        history, history_speakers, history_mask = self._history_windows(embeddings, speakers, mask)
        # Dialogue batches are padded to a common length. Some flattened padding rows
        # have no observed history at all, so only send real utterance positions to
        # HiTFT and scatter their predictions back into a zero-initialized tensor.
        valid_rows = mask.reshape(-1)
        flat_shape = batch_size * length
        flat_future = embeddings.new_zeros(
            flat_shape,
            self.future_steps,
            self.embedding_dim,
        )
        forecaster_mode = getattr(self.hitft, "forecaster_mode", "absolute")
        transition_aware = forecaster_mode in {"transition_aware", "transition_residual"}
        flat_stable = flat_future.clone()
        flat_transition = flat_future.clone()
        flat_probability = embeddings.new_zeros(flat_shape, self.future_steps)
        if valid_rows.any():
            with torch.no_grad():
                if transition_aware:
                    prediction, auxiliary = self.hitft(
                        history[valid_rows],
                        history_speakers[valid_rows],
                        history_mask[valid_rows],
                        return_aux=True,
                    )
                    flat_stable[valid_rows] = auxiliary["stable_prediction"].to(
                        dtype=flat_stable.dtype
                    )
                    flat_transition[valid_rows] = auxiliary["transition_prediction"].to(
                        dtype=flat_transition.dtype
                    )
                    flat_probability[valid_rows] = auxiliary["transition_probability"].to(
                        dtype=flat_probability.dtype
                    )
                else:
                    prediction = self.hitft(
                        history[valid_rows],
                        history_speakers[valid_rows],
                        history_mask[valid_rows],
                    )
                flat_future[valid_rows] = prediction.to(dtype=flat_future.dtype)
        details["prediction"] = flat_future.reshape(
            batch_size, length, self.future_steps, self.embedding_dim
        )
        details["stable_prediction"] = flat_stable.reshape(
            batch_size, length, self.future_steps, self.embedding_dim
        )
        details["transition_prediction"] = flat_transition.reshape(
            batch_size, length, self.future_steps, self.embedding_dim
        )
        details["transition_probability"] = flat_probability.reshape(
            batch_size, length, self.future_steps
        )
        return details

    def _pseudo_future(
        self, embeddings: torch.Tensor, speakers: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self._pseudo_future_details(embeddings, speakers, mask)["prediction"]

    def forward(
        self,
        embeddings: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
        return_aux: bool = False,
        current_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future_gate = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1], 1
        )
        stable_future = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1], self.future_steps, self.embedding_dim
        )
        transition_future = stable_future.clone()
        transition_probability = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1], self.future_steps
        )
        if self.use_hitft:
            future_details = self._pseudo_future_details(embeddings, speaker_ids, mask)
            future = future_details["prediction"]
            stable_future = future_details["stable_prediction"]
            transition_future = future_details["transition_prediction"]
            transition_probability = future_details["transition_probability"]
            if self.future_fusion_name == "flatten_project":
                assert self.future_fusion is not None
                fused_input = torch.cat([embeddings, future.flatten(start_dim=2)], dim=-1)
                context = self.future_fusion(fused_input)
            elif self.future_fusion_name == "residual_gated":
                assert self.future_projection is not None
                assert self.future_gate is not None
                assert self.future_fusion_norm is not None
                current_context = self.current_only(embeddings)
                future_context = self.future_projection(future.flatten(start_dim=2))
                future_gate = torch.sigmoid(
                    self.future_gate(torch.cat([current_context, future_context], dim=-1))
                )
                context = self.future_fusion_norm(
                    current_context + future_gate * future_context
                )
            else:
                assert self.stable_future_projection is not None
                assert self.transition_future_projection is not None
                assert self.future_gate is not None
                assert self.future_fusion_norm is not None
                current_context = self.current_only(embeddings)
                stable_context = self.stable_future_projection(
                    stable_future.flatten(start_dim=2)
                )
                transition_context = self.transition_future_projection(
                    transition_future.flatten(start_dim=2)
                )
                route_probability = transition_probability.max(dim=-1, keepdim=True).values
                routed_future = stable_context + route_probability * (
                    transition_context - stable_context
                )
                future_gate = torch.sigmoid(
                    self.future_gate(
                        torch.cat(
                            [current_context, routed_future, route_probability], dim=-1
                        )
                    )
                )
                context = self.future_fusion_norm(
                    current_context + future_gate * routed_future
                )
        else:
            future = embeddings.new_zeros(
                embeddings.shape[0], embeddings.shape[1], self.future_steps, self.embedding_dim
            )
            context = self.current_only(embeddings)
        context = context * mask.unsqueeze(-1).to(context.dtype)

        if self.use_directed_update:
            if current_logits is None:
                raise ValueError(
                    "current_logits are required for directed probabilistic updates"
                )
            assert self.directed_affect_updater is not None
            directed = self.directed_affect_updater(
                context=context,
                speaker_ids=speaker_ids,
                mask=mask,
                anchor_logits=current_logits,
            )
            logits = directed["logits"]
            if not return_aux:
                return logits
            active_gates = []
            if self.directed_affect_updater.use_self_channel:
                active_gates.append(directed["self_gate"])
            if self.directed_affect_updater.use_other_channel:
                active_gates.append(directed["other_gate"])
            if self.directed_affect_updater.use_dynamic_transition:
                active_gates.append(directed["transition_gate"])
            context_gate = (
                torch.stack(active_gates, dim=0).mean(dim=0)
                if active_gates
                else embeddings.new_zeros((*embeddings.shape[:2], 1))
            )
            zeros = embeddings.new_zeros(embeddings.shape[:2])
            ones = embeddings.new_ones((*embeddings.shape[:2], 1))
            auxiliary = {
                "pseudo_future": future,
                "stable_future": stable_future,
                "transition_future": transition_future,
                "transition_probability": transition_probability,
                "future_gate": future_gate,
                "context": context,
                "speaker_state": directed["self_memory"],
                "context_logits": directed["context_logits"],
                "context_gate": context_gate,
                "base_context_gate": context_gate,
                "capped_base_context_gate": context_gate,
                "transition_suppression": ones,
                "helpfulness_modulation": ones,
                "predicted_transition_logits": zeros,
                "predicted_transition_probability": zeros,
                "predicted_helpfulness_logits": zeros,
                "predicted_helpfulness_probability": zeros,
                **directed,
            }
            return logits, auxiliary

        if self.speaker_encoder is not None:
            speaker_state = self.speaker_encoder(context, speaker_ids, mask)
        else:
            speaker_state = context
        if self.decoder is not None:
            context_logits = self.decoder(
                context,
                speaker_state,
                mask,
                labels=labels,
                teacher_forcing=teacher_forcing,
            )
        else:
            assert self.direct_classifier is not None
            context_logits = self.direct_classifier(
                torch.cat([context, speaker_state], dim=-1)
            )

        context_gate = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1], 1
        )
        predicted_transition_logits = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1]
        )
        predicted_transition_probability = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1]
        )
        predicted_helpfulness_logits = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1]
        )
        predicted_helpfulness_probability = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[1]
        )
        base_context_gate = context_gate.clone()
        capped_base_context_gate = context_gate.clone()
        transition_suppression = torch.ones_like(context_gate)
        helpfulness_modulation = torch.ones_like(context_gate)
        logits = context_logits
        if self.use_current_logit_anchor:
            if current_logits is None:
                raise ValueError(
                    "current_logits are required for a current-anchored model"
                )
            expected_shape = (*embeddings.shape[:2], self.num_classes)
            if current_logits.shape != expected_shape:
                raise ValueError(
                    f"Expected current_logits shape {expected_shape}, "
                    f"got {tuple(current_logits.shape)}"
                )
            anchor_logits = current_logits.to(dtype=context_logits.dtype)
            if self.use_class_residual:
                assert self.class_residual_corrector is not None
                correction = self.class_residual_corrector(
                    context, speaker_ids, mask, anchor_logits
                )
                logits = correction["logits"]
                context_logits = correction["residual_expert_logits"]
                predicted_transition_logits = correction[
                    "predicted_transition_logits"
                ]
                predicted_transition_probability = correction[
                    "predicted_transition_probability"
                ]
                context_gate = correction["class_gate"].mean(dim=-1, keepdim=True)
                base_context_gate = context_gate
                capped_base_context_gate = context_gate
            else:
                assert self.transition_logit_head is not None
                anchor_probability = torch.softmax(anchor_logits, dim=-1)
                context_probability = torch.softmax(context_logits, dim=-1)
                gate_features = torch.cat(
                    [context, speaker_state, anchor_probability], dim=-1
                )
                predicted_transition_logits = self.transition_logit_head(
                    gate_features
                ).squeeze(-1)
                predicted_transition_probability = torch.sigmoid(
                    predicted_transition_logits
                )
                if self.context_gate_mode == "fixed":
                    base_context_gate = torch.full_like(
                        predicted_transition_probability.unsqueeze(-1),
                        self.context_gate_fixed_value / self.context_gate_max,
                    )
                    capped_base_context_gate = self.context_gate_max * base_context_gate
                    context_gate = capped_base_context_gate
                elif self.context_gate_mode == "learned":
                    assert self.context_logit_gate is not None
                    base_context_gate = torch.sigmoid(
                        self.context_logit_gate(gate_features)
                    )
                    capped_base_context_gate = self.context_gate_max * base_context_gate
                    context_gate = capped_base_context_gate
                elif self.context_gate_mode == "transition":
                    assert self.context_logit_gate is not None
                    base_context_gate = torch.sigmoid(
                        self.context_logit_gate(gate_features)
                    )
                    capped_base_context_gate = self.context_gate_max * base_context_gate
                    transition_suppression = (
                        1.0 - predicted_transition_probability.unsqueeze(-1)
                    ).pow(self.transition_gate_power)
                    context_gate = capped_base_context_gate * transition_suppression
                elif self.context_gate_mode == "helpfulness":
                    assert self.helpfulness_logit_head is not None
                    predicted_helpfulness_logits = self.helpfulness_logit_head(
                        gate_features
                    ).squeeze(-1)
                    predicted_helpfulness_probability = torch.sigmoid(
                        predicted_helpfulness_logits
                    )
                    base_context_gate = predicted_helpfulness_probability.unsqueeze(-1)
                    capped_base_context_gate = self.context_gate_max * base_context_gate
                    context_gate = capped_base_context_gate
                    helpfulness_modulation = base_context_gate
                else:
                    assert self.context_logit_gate is not None
                    assert self.helpfulness_logit_head is not None
                    base_context_gate = torch.sigmoid(
                        self.context_logit_gate(gate_features)
                    )
                    capped_base_context_gate = self.context_gate_max * base_context_gate
                    helpfulness_features = torch.cat(
                        [
                            gate_features,
                            context_probability,
                            predicted_transition_probability.unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    predicted_helpfulness_logits = self.helpfulness_logit_head(
                        helpfulness_features
                    ).squeeze(-1)
                    predicted_helpfulness_probability = torch.sigmoid(
                        predicted_helpfulness_logits
                    )
                    helpfulness_modulation = (
                        predicted_helpfulness_probability.unsqueeze(-1)
                    ).pow(self.helpfulness_gate_power)
                    if self.context_gate_mode == "transition_helpfulness":
                        transition_suppression = (
                            1.0 - predicted_transition_probability.unsqueeze(-1)
                        ).pow(self.transition_gate_power)
                    context_gate = (
                        capped_base_context_gate
                        * transition_suppression
                        * helpfulness_modulation
                    )
                context_gate = context_gate * mask.unsqueeze(-1).to(context_gate.dtype)
                base_context_gate = base_context_gate * mask.unsqueeze(-1).to(
                    base_context_gate.dtype
                )
                capped_base_context_gate = capped_base_context_gate * mask.unsqueeze(-1).to(
                    capped_base_context_gate.dtype
                )
                transition_suppression = transition_suppression * mask.unsqueeze(-1).to(
                    transition_suppression.dtype
                )
                helpfulness_modulation = helpfulness_modulation * mask.unsqueeze(-1).to(
                    helpfulness_modulation.dtype
                )
                logits = anchor_logits + context_gate * (context_logits - anchor_logits)
        if not return_aux:
            return logits
        auxiliary = {
            "pseudo_future": future,
            "stable_future": stable_future,
            "transition_future": transition_future,
            "transition_probability": transition_probability,
            "future_gate": future_gate,
            "context": context,
            "speaker_state": speaker_state,
            "context_logits": context_logits,
            "context_gate": context_gate,
            "base_context_gate": base_context_gate,
            "capped_base_context_gate": capped_base_context_gate,
            "transition_suppression": transition_suppression,
            "helpfulness_modulation": helpfulness_modulation,
            "predicted_transition_logits": predicted_transition_logits,
            "predicted_transition_probability": predicted_transition_probability,
            "predicted_helpfulness_logits": predicted_helpfulness_logits,
            "predicted_helpfulness_probability": predicted_helpfulness_probability,
        }
        if self.use_class_residual:
            auxiliary.update(correction)
            auxiliary["context_logits"] = context_logits
        return logits, auxiliary
