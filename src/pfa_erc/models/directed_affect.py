from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .layers import masked_attention_pool


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Gate initialization probabilities must be in (0, 1)")
    return math.log(probability / (1.0 - probability))


def _jensen_shannon(
    first: torch.Tensor,
    second: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    first = first.clamp_min(epsilon)
    second = second.clamp_min(epsilon)
    mixture = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - mixture.log())).sum(dim=-1, keepdim=True)
        + (second * (second.log() - mixture.log())).sum(dim=-1, keepdim=True)
    )


def _normalized_entropy(
    probability: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    classes = probability.shape[-1]
    if classes <= 1:
        return probability.new_zeros((*probability.shape[:-1], 1))
    safe = probability.clamp_min(epsilon)
    entropy = -(safe * safe.log()).sum(dim=-1, keepdim=True)
    return entropy / math.log(classes)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denominator


class DirectedProbabilisticAffectUpdater(nn.Module):
    """Causal, speaker-conditioned probability updates for Experiments 8B/8C.

    The current-utterance RoBERTa logits remain the prediction anchor. Past
    utterances are split into same-speaker and other-speaker channels, and a
    separate transition operator propagates the current speaker's last
    predicted distribution. Experiment 8C changes one mechanism at a time via
    ``directed_matrix_mode`` and ``directed_prior_mode``. No emotion label,
    stable/transition flag, future turn, or dialogue-external utterance is
    supplied to ``forward``.
    """

    VALID_VARIANTS = {
        "self_only",
        "global_context",
        "dual_channel",
        "dynamic_only",
        "full",
    }
    VALID_MATRIX_MODES = {
        "dynamic",
        "static",
        "logit_residual",
        "static_logit_residual",
        "direct_residual",
    }
    VALID_PRIOR_MODES = {"speaker", "uniform", "other_speaker"}
    VALID_MATRIX_FEATURE_MODES = {"current_conditioned", "history_only"}
    VALID_FUSION_MODES = {
        "probability_interpolation",
        "logit_residual",
        "bayes_evidence",
    }

    def __init__(self, config: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        self.hidden_dim = int(model["hidden_dim"])
        self.num_classes = int(num_classes)
        self.max_speakers = int(data.get("max_speakers", 16))
        self.variant = str(model.get("directed_variant", "full"))
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                "model.directed_variant must be one of "
                f"{sorted(self.VALID_VARIANTS)}, got {self.variant!r}"
            )
        self.matrix_mode = str(model.get("directed_matrix_mode", "dynamic"))
        if self.matrix_mode not in self.VALID_MATRIX_MODES:
            raise ValueError(
                "model.directed_matrix_mode must be one of "
                f"{sorted(self.VALID_MATRIX_MODES)}, got {self.matrix_mode!r}"
            )
        self.prior_mode = str(model.get("directed_prior_mode", "speaker"))
        if self.prior_mode not in self.VALID_PRIOR_MODES:
            raise ValueError(
                "model.directed_prior_mode must be one of "
                f"{sorted(self.VALID_PRIOR_MODES)}, got {self.prior_mode!r}"
            )
        self.matrix_feature_mode = str(
            model.get("directed_matrix_feature_mode", "current_conditioned")
        )
        if self.matrix_feature_mode not in self.VALID_MATRIX_FEATURE_MODES:
            raise ValueError(
                "model.directed_matrix_feature_mode must be one of "
                f"{sorted(self.VALID_MATRIX_FEATURE_MODES)}, "
                f"got {self.matrix_feature_mode!r}"
            )
        legacy_fusion_mode = (
            "logit_residual"
            if self.matrix_mode
            in {
                "logit_residual",
                "static_logit_residual",
                "direct_residual",
            }
            else "probability_interpolation"
        )
        self.fusion_mode = str(
            model.get("directed_fusion_mode", legacy_fusion_mode)
        )
        if self.fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                "model.directed_fusion_mode must be one of "
                f"{sorted(self.VALID_FUSION_MODES)}, got {self.fusion_mode!r}"
            )

        self.use_self_channel = self.variant in {
            "self_only",
            "global_context",
            "dual_channel",
            "full",
        }
        self.use_other_channel = self.variant in {"dual_channel", "full"}
        self.use_dynamic_transition = self.variant in {"dynamic_only", "full"}
        self.is_capacity_matched_direct = (
            self.matrix_mode == "direct_residual"
        )
        self.detach_belief_history = bool(
            model.get("directed_detach_belief_history", True)
        )
        self.correction_scale = float(model.get("directed_correction_scale", 1.0))
        self.transition_scale = float(model.get("directed_transition_scale", 1.0))
        if self.correction_scale < 0.0 or self.transition_scale < 0.0:
            raise ValueError("Directed update scales cannot be negative")

        dropout = float(model.get("dropout", 0.3))
        token_dim = self.hidden_dim + self.num_classes
        self.history_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_projection = nn.Linear(token_dim, self.hidden_dim)
        self.self_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.other_key = nn.Linear(self.hidden_dim, self.hidden_dim)

        channel_dim = self.hidden_dim * 2 + self.num_classes
        self.self_delta_head = self._delta_head(channel_dim, dropout)
        self.other_delta_head = self._delta_head(channel_dim, dropout)
        self.self_gate_head = nn.Linear(channel_dim, 1)
        self.other_gate_head = nn.Linear(channel_dim, 1)

        gate_init = float(model.get("directed_gate_init", 0.05))
        nn.init.zeros_(self.self_gate_head.weight)
        nn.init.zeros_(self.other_gate_head.weight)
        nn.init.constant_(self.self_gate_head.bias, _logit(gate_init))
        nn.init.constant_(self.other_gate_head.bias, _logit(gate_init))

        if self.matrix_feature_mode == "history_only":
            # Two strictly pre-current-turn memories, the stored speaker belief,
            # and four causal reliability features: same/other availability,
            # whether the speaker has appeared, and the normalized turn gap.
            dynamic_dim = self.hidden_dim * 2 + self.num_classes + 4
        else:
            # Backward-compatible Experiment 8 input: the matrix may read the
            # current utterance representation and current Anchor probability.
            dynamic_dim = self.hidden_dim * 3 + self.num_classes * 2
        matrix_hidden = int(model.get("directed_matrix_hidden_dim", self.hidden_dim))
        self.transition_matrix_head = nn.Sequential(
            nn.LayerNorm(dynamic_dim),
            nn.Linear(dynamic_dim, matrix_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(matrix_hidden, self.num_classes * self.num_classes),
        )
        nn.init.zeros_(self.transition_matrix_head[-1].weight)
        nn.init.zeros_(self.transition_matrix_head[-1].bias)

        diagonal_bias = float(model.get("directed_transition_diagonal_bias", 2.0))
        initial_transition = torch.zeros(self.num_classes, self.num_classes)
        initial_transition.fill_diagonal_(diagonal_bias)
        self.base_transition_logits = nn.Parameter(initial_transition)

        configured_class_prior = model.get(
            "directed_class_prior",
            [1.0 / self.num_classes] * self.num_classes,
        )
        if (
            not isinstance(configured_class_prior, list)
            or len(configured_class_prior) != self.num_classes
        ):
            raise ValueError(
                "model.directed_class_prior must contain exactly "
                f"{self.num_classes} values in label-id order"
            )
        class_prior = torch.tensor(configured_class_prior, dtype=torch.float32)
        if not torch.isfinite(class_prior).all() or (class_prior <= 0).any():
            raise ValueError(
                "model.directed_class_prior values must be finite and positive"
            )
        self.register_buffer(
            "class_prior",
            class_prior / class_prior.sum(),
            persistent=False,
        )

        transition_gate_dim = dynamic_dim + 1
        self.transition_gate_head = nn.Linear(transition_gate_dim, 1)
        transition_gate_init = float(
            model.get("directed_transition_gate_init", gate_init)
        )
        nn.init.zeros_(self.transition_gate_head.weight)
        nn.init.constant_(
            self.transition_gate_head.bias, _logit(transition_gate_init)
        )
        if bool(model.get("directed_freeze_unused_parameters", False)):
            self._freeze_unused_parameters()

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def _freeze_unused_parameters(self) -> None:
        """Exclude inactive branches from the scientific parameter count.

        The reviewer-closure experiment compares modules that share the same
        RoBERTa features and training code.  Freezing inactive heads prevents
        nominal (but gradient-free) parameters from making a baseline appear
        larger than it really is.  The belief-transition model and its
        capacity-matched direct control keep exactly the same active modules.
        """

        if not self.use_self_channel:
            self._set_trainable(self.self_delta_head, False)
            self._set_trainable(self.self_gate_head, False)
        if not self.use_other_channel:
            self._set_trainable(self.other_delta_head, False)
            self._set_trainable(self.other_gate_head, False)
        if not self.use_dynamic_transition:
            self._set_trainable(self.transition_matrix_head, False)
            self._set_trainable(self.transition_gate_head, False)
            self.base_transition_logits.requires_grad_(False)
        if self.variant in {"self_only", "global_context"}:
            self._set_trainable(self.other_key, False)

    def _delta_head(self, input_dim: int, dropout: float) -> nn.Sequential:
        head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        # Every trainable variant starts at the frozen 3A anchor. This makes
        # optimization measure learned corrections rather than random damage.
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @staticmethod
    def _center(delta: torch.Tensor) -> torch.Tensor:
        # Adding a constant to every logit changes no probability, so removing
        # it makes correction magnitude and gate diagnostics identifiable.
        return delta - delta.mean(dim=-1, keepdim=True)

    def forward(
        self,
        context: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        anchor_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError("context must have shape [B,T,H]")
        if context.shape[:2] != speaker_ids.shape or mask.shape != speaker_ids.shape:
            raise ValueError("speaker_ids and mask must match context [B,T]")
        expected_logits = (*context.shape[:2], self.num_classes)
        if tuple(anchor_logits.shape) != expected_logits:
            raise ValueError(
                f"Expected anchor_logits shape {expected_logits}, "
                f"got {tuple(anchor_logits.shape)}"
            )
        if context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {context.shape[-1]}"
            )

        batch_size, turns, _ = context.shape
        dtype = context.dtype
        device = context.device
        speakers = speaker_ids.clamp(0, self.max_speakers - 1)
        anchor_logits = anchor_logits.to(dtype)
        anchor_probability = torch.softmax(anchor_logits.float(), dim=-1).to(dtype)
        history_tokens = self.history_projection(
            torch.cat([context, anchor_probability], dim=-1)
        )

        uniform = torch.full(
            (batch_size, self.max_speakers, self.num_classes),
            1.0 / self.num_classes,
            device=device,
            dtype=dtype,
        )
        beliefs = uniform
        seen = torch.zeros(
            batch_size, self.max_speakers, device=device, dtype=torch.bool
        )
        last_seen_turn = torch.full(
            (batch_size, self.max_speakers),
            -1,
            device=device,
            dtype=torch.long,
        )
        batch_indices = torch.arange(batch_size, device=device)

        final_logits_by_turn: list[torch.Tensor] = []
        channel_logits_by_turn: list[torch.Tensor] = []
        self_delta_by_turn: list[torch.Tensor] = []
        other_delta_by_turn: list[torch.Tensor] = []
        transition_delta_by_turn: list[torch.Tensor] = []
        self_gate_by_turn: list[torch.Tensor] = []
        other_gate_by_turn: list[torch.Tensor] = []
        transition_gate_by_turn: list[torch.Tensor] = []
        self_available_by_turn: list[torch.Tensor] = []
        other_available_by_turn: list[torch.Tensor] = []
        speaker_known_by_turn: list[torch.Tensor] = []
        self_memory_by_turn: list[torch.Tensor] = []
        other_memory_by_turn: list[torch.Tensor] = []
        prior_by_turn: list[torch.Tensor] = []
        js_by_turn: list[torch.Tensor] = []
        matrix_by_turn: list[torch.Tensor] = []
        matrix_delta_rms_by_turn: list[torch.Tensor] = []
        anchor_entropy_by_turn: list[torch.Tensor] = []
        prior_entropy_by_turn: list[torch.Tensor] = []
        speaker_gap_by_turn: list[torch.Tensor] = []
        self_attention_by_turn: list[torch.Tensor] = []
        other_attention_by_turn: list[torch.Tensor] = []

        for turn in range(turns):
            valid = mask[:, turn]
            valid_f = valid.to(dtype).unsqueeze(-1)
            current_speaker = speakers[:, turn]
            current_anchor = anchor_probability[:, turn]
            current_context = context[:, turn]
            query = self.query_projection(
                torch.cat([current_context, current_anchor], dim=-1)
            )

            if turn == 0:
                self_memory = torch.zeros_like(current_context)
                other_memory = torch.zeros_like(current_context)
                self_weights = context.new_zeros(batch_size, 0)
                other_weights = context.new_zeros(batch_size, 0)
                self_available = torch.zeros(
                    batch_size, device=device, dtype=torch.bool
                )
                other_available = torch.zeros_like(self_available)
            else:
                prior_tokens = history_tokens[:, :turn]
                prior_speakers = speakers[:, :turn]
                prior_valid = mask[:, :turn]
                same_mask = (
                    prior_valid
                    & prior_speakers.eq(current_speaker.unsqueeze(-1))
                    & valid.unsqueeze(-1)
                )
                other_mask = (
                    prior_valid
                    & prior_speakers.ne(current_speaker.unsqueeze(-1))
                    & valid.unsqueeze(-1)
                )
                self_mask = (
                    prior_valid & valid.unsqueeze(-1)
                    if self.variant == "global_context"
                    else same_mask
                )
                self_memory, self_weights = masked_attention_pool(
                    query,
                    self.self_key(prior_tokens),
                    prior_tokens,
                    self_mask,
                )
                other_memory, other_weights = masked_attention_pool(
                    query,
                    self.other_key(prior_tokens),
                    prior_tokens,
                    other_mask,
                )
                self_available = self_mask.any(dim=-1)
                other_available = other_mask.any(dim=-1)

            self_features = torch.cat(
                [current_context, self_memory, current_anchor], dim=-1
            )
            other_features = torch.cat(
                [current_context, other_memory, current_anchor], dim=-1
            )
            raw_self_delta = self._center(self.self_delta_head(self_features))
            raw_other_delta = self._center(self.other_delta_head(other_features))
            self_gate = (
                torch.sigmoid(self.self_gate_head(self_features))
                * self_available.to(dtype).unsqueeze(-1)
                * valid_f
            )
            other_gate = (
                torch.sigmoid(self.other_gate_head(other_features))
                * other_available.to(dtype).unsqueeze(-1)
                * valid_f
            )
            if not self.use_self_channel:
                self_gate = torch.zeros_like(self_gate)
            if not self.use_other_channel:
                other_gate = torch.zeros_like(other_gate)
            applied_self_delta = (
                self.correction_scale * self_gate * raw_self_delta
                if self.use_self_channel
                else torch.zeros_like(raw_self_delta)
            )
            applied_other_delta = (
                self.correction_scale * other_gate * raw_other_delta
                if self.use_other_channel
                else torch.zeros_like(raw_other_delta)
            )
            channel_logits = (
                anchor_logits[:, turn] + applied_self_delta + applied_other_delta
            )

            true_speaker_belief = beliefs[batch_indices, current_speaker]
            speaker_known = seen[batch_indices, current_speaker] & valid
            if self.prior_mode == "uniform":
                previous_belief = uniform[:, 0]
            elif self.prior_mode == "other_speaker":
                other_last_seen = last_seen_turn.clone()
                other_last_seen.scatter_(
                    1, current_speaker.unsqueeze(-1), -1
                )
                other_turn, other_speaker = other_last_seen.max(dim=-1)
                other_belief = beliefs[batch_indices, other_speaker]
                previous_belief = torch.where(
                    other_turn.ge(0).unsqueeze(-1),
                    other_belief,
                    uniform[:, 0],
                )
            else:
                previous_belief = true_speaker_belief
            # RoleTrans and its direct control receive the same detached
            # recurrent speaker state.  The only intervention is downstream:
            # RoleTrans left-multiplies a KxK operator by this belief, whereas
            # the control maps the same KxK-sized head directly to K logits.
            # This isolates transition geometry instead of confounding it with
            # whether a recurrent speaker state is available.
            conditioning_distribution = previous_belief
            previous_seen_turn = last_seen_turn[batch_indices, current_speaker]
            raw_speaker_gap = torch.where(
                speaker_known,
                torch.full_like(previous_seen_turn, turn) - previous_seen_turn,
                torch.zeros_like(previous_seen_turn),
            )
            gap_denominator = math.log1p(max(turns, 1))
            normalized_speaker_gap = (
                torch.log1p(raw_speaker_gap.to(dtype)) / gap_denominator
            ).unsqueeze(-1)

            if self.matrix_feature_mode == "history_only":
                if turn == 0:
                    transition_self_memory = torch.zeros_like(current_context)
                    transition_other_memory = torch.zeros_like(current_context)
                else:
                    # Unlike the correction-channel attention above, these
                    # summaries do not use a query derived from x_t or a_t.
                    # They therefore remain valid pre-current-turn features.
                    transition_self_memory = _masked_mean(
                        prior_tokens, same_mask
                    )
                    transition_other_memory = _masked_mean(
                        prior_tokens, other_mask
                    )
                dynamic_features = torch.cat(
                    [
                        transition_self_memory,
                        transition_other_memory,
                        conditioning_distribution,
                        self_available.to(dtype).unsqueeze(-1),
                        other_available.to(dtype).unsqueeze(-1),
                        speaker_known.to(dtype).unsqueeze(-1),
                        normalized_speaker_gap,
                    ],
                    dim=-1,
                )
            else:
                dynamic_features = torch.cat(
                    [
                        current_context,
                        self_memory,
                        other_memory,
                        current_anchor,
                        conditioning_distribution,
                    ],
                    dim=-1,
                )
            raw_matrix_delta = self.transition_matrix_head(dynamic_features).reshape(
                batch_size, self.num_classes, self.num_classes
            )
            matrix_delta = (
                torch.zeros_like(raw_matrix_delta)
                if self.matrix_mode in {"static", "static_logit_residual"}
                else raw_matrix_delta
            )
            transition_logits = (
                self.base_transition_logits.unsqueeze(0)
                + self.transition_scale * matrix_delta
            )
            transition_matrix = torch.softmax(
                transition_logits.float(), dim=-1
            ).to(dtype)
            if self.is_capacity_matched_direct:
                # This is deliberately *not* a belief transition.  The same
                # D->H->K^2 head and K^2 base tensor are retained so the
                # trainable parameter count exactly matches RoleTrans, but the
                # K^2 values are treated as an over-parameterized direct
                # residual. LogSumExp pooling makes every K^2 coordinate
                # functionally active while remaining invariant to arbitrary
                # row ordering. No belief is multiplied by an operator.
                direct_residual = (
                    torch.logsumexp(transition_logits, dim=1)
                    - math.log(self.num_classes)
                )
                transition_update = self._center(direct_residual)
                dynamic_prior = torch.softmax(
                    direct_residual.float(), dim=-1
                ).to(dtype)
                js_divergence = _jensen_shannon(
                    current_anchor, conditioning_distribution
                )
            else:
                dynamic_prior = torch.bmm(
                    previous_belief.unsqueeze(1), transition_matrix
                ).squeeze(1)
                js_divergence = _jensen_shannon(
                    current_anchor, dynamic_prior
                )
            transition_gate = (
                torch.sigmoid(
                    self.transition_gate_head(
                        torch.cat([dynamic_features, js_divergence], dim=-1)
                    )
                )
                * speaker_known.to(dtype).unsqueeze(-1)
                * valid_f
            )
            if not self.use_dynamic_transition:
                transition_gate = torch.zeros_like(transition_gate)
            if not self.is_capacity_matched_direct:
                if self.fusion_mode == "logit_residual":
                    raw_residual = torch.bmm(
                        previous_belief.unsqueeze(1), transition_logits
                    ).squeeze(1)
                    transition_update = self._center(raw_residual)
                elif self.fusion_mode == "bayes_evidence":
                    class_prior = self.class_prior.to(
                        device=device, dtype=dynamic_prior.dtype
                    )
                    transition_update = self._center(
                        dynamic_prior.clamp_min(1e-8).log()
                        - class_prior.clamp_min(1e-8).log().unsqueeze(0)
                    )
                else:
                    transition_update = (
                        dynamic_prior.clamp_min(1e-8).log()
                        - current_anchor.clamp_min(1e-8).log()
                    )
            applied_transition_delta = (
                transition_gate * transition_update
                if self.use_dynamic_transition
                else torch.zeros_like(transition_update)
            )
            final_logits = channel_logits + applied_transition_delta
            final_logits = torch.where(
                valid.unsqueeze(-1), final_logits, torch.zeros_like(final_logits)
            )
            posterior = torch.softmax(final_logits.float(), dim=-1).to(dtype)

            # A speaker is updated only when that speaker produces a valid turn.
            # While absent, their stored probability distribution is unchanged.
            stored_posterior = (
                posterior.detach() if self.detach_belief_history else posterior
            )
            speaker_one_hot = F.one_hot(
                current_speaker, self.max_speakers
            ).to(dtype).unsqueeze(-1)
            update = speaker_one_hot * valid.to(dtype)[:, None, None]
            beliefs = beliefs * (1.0 - update) + stored_posterior.unsqueeze(1) * update
            seen = seen | (
                F.one_hot(current_speaker, self.max_speakers).bool()
                & valid.unsqueeze(-1)
            )
            observed_speaker = (
                F.one_hot(current_speaker, self.max_speakers).bool()
                & valid.unsqueeze(-1)
            )
            last_seen_turn = torch.where(
                observed_speaker,
                torch.full_like(last_seen_turn, turn),
                last_seen_turn,
            )

            final_logits_by_turn.append(final_logits)
            channel_logits_by_turn.append(channel_logits)
            self_delta_by_turn.append(applied_self_delta)
            other_delta_by_turn.append(applied_other_delta)
            transition_delta_by_turn.append(applied_transition_delta)
            self_gate_by_turn.append(self_gate)
            other_gate_by_turn.append(other_gate)
            transition_gate_by_turn.append(transition_gate)
            self_available_by_turn.append(self_available & valid)
            other_available_by_turn.append(other_available & valid)
            speaker_known_by_turn.append(speaker_known)
            self_memory_by_turn.append(self_memory)
            other_memory_by_turn.append(other_memory)
            prior_by_turn.append(dynamic_prior)
            js_by_turn.append(js_divergence * valid_f)
            matrix_by_turn.append(transition_matrix)
            matrix_delta_rms_by_turn.append(
                matrix_delta.square().mean(dim=(-2, -1), keepdim=False).sqrt()
                * valid.to(dtype)
            )
            anchor_entropy_by_turn.append(
                _normalized_entropy(current_anchor) * valid_f
            )
            prior_entropy_by_turn.append(
                _normalized_entropy(dynamic_prior) * valid_f
            )
            speaker_gap_by_turn.append(
                normalized_speaker_gap.squeeze(-1) * valid.to(dtype)
            )
            self_attention_by_turn.append(
                F.pad(self_weights, (0, turns - self_weights.shape[-1]))
            )
            other_attention_by_turn.append(
                F.pad(other_weights, (0, turns - other_weights.shape[-1]))
            )

        return {
            "logits": torch.stack(final_logits_by_turn, dim=1),
            "context_logits": torch.stack(channel_logits_by_turn, dim=1),
            "self_delta": torch.stack(self_delta_by_turn, dim=1),
            "other_delta": torch.stack(other_delta_by_turn, dim=1),
            "transition_delta": torch.stack(transition_delta_by_turn, dim=1),
            "self_gate": torch.stack(self_gate_by_turn, dim=1),
            "other_gate": torch.stack(other_gate_by_turn, dim=1),
            "transition_gate": torch.stack(transition_gate_by_turn, dim=1),
            "self_available": torch.stack(self_available_by_turn, dim=1),
            "other_available": torch.stack(other_available_by_turn, dim=1),
            "speaker_known": torch.stack(speaker_known_by_turn, dim=1),
            "self_memory": torch.stack(self_memory_by_turn, dim=1),
            "other_memory": torch.stack(other_memory_by_turn, dim=1),
            "dynamic_prior": torch.stack(prior_by_turn, dim=1),
            "js_divergence": torch.stack(js_by_turn, dim=1),
            "transition_matrix": torch.stack(matrix_by_turn, dim=1),
            "matrix_delta_rms": torch.stack(matrix_delta_rms_by_turn, dim=1),
            "anchor_entropy": torch.stack(anchor_entropy_by_turn, dim=1),
            "history_prior_entropy": torch.stack(
                prior_entropy_by_turn, dim=1
            ),
            "speaker_gap": torch.stack(speaker_gap_by_turn, dim=1),
            "self_attention": torch.stack(self_attention_by_turn, dim=1),
            "other_attention": torch.stack(other_attention_by_turn, dim=1),
        }
