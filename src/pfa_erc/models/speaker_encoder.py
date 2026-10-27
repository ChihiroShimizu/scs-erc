from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .layers import causal_attention_mask, masked_attention_pool


class SpeakerEmotionEncoder(nn.Module):
    """Causal inter-/intra-speaker state encoder."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        hidden = int(model["hidden_dim"])
        self.hidden_dim = hidden
        self.max_speakers = int(data.get("max_speakers", 16))
        self.inter_speaker_only = bool(model.get("inter_speaker_only", True))
        dropout = float(model.get("dropout", 0.3))
        heads = int(model.get("num_heads", 4))

        self.intra_query = nn.Linear(hidden * 2, hidden)
        self.inter_query = nn.Linear(hidden, hidden)
        self.intra_key = nn.Linear(hidden, hidden)
        self.inter_key = nn.Linear(hidden, hidden)
        self.combine = nn.Linear(hidden * 2, hidden)
        self.current_projection = nn.Linear(hidden, hidden, bias=False)
        self.self_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        context: torch.Tensor,
        speaker_ids: torch.Tensor,
        mask: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        batch_size, length, hidden = context.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {hidden}")
        speaker_ids = speaker_ids.clamp(min=0, max=self.max_speakers - 1)
        speaker_state = context.new_zeros(batch_size, self.max_speakers, hidden)
        state_history: list[torch.Tensor] = []
        intra_weights: list[torch.Tensor] = []
        inter_weights: list[torch.Tensor] = []

        for t in range(length):
            current = context[:, t]
            current_speaker = speaker_ids[:, t]
            previous_same_state = speaker_state.gather(
                1, current_speaker[:, None, None].expand(-1, 1, hidden)
            ).squeeze(1)
            if t == 0:
                intra = torch.zeros_like(current)
                inter = torch.zeros_like(current)
                intra_weight = current.new_zeros(batch_size, 0)
                inter_weight = current.new_zeros(batch_size, 0)
            else:
                prior_context = context[:, :t]
                prior_speakers = speaker_ids[:, :t]
                prior_mask = mask[:, :t]
                intra_mask = prior_mask & prior_speakers.eq(current_speaker.unsqueeze(1))
                intra_query = self.intra_query(torch.cat([previous_same_state, current], dim=-1))
                intra, intra_weight = masked_attention_pool(
                    intra_query,
                    self.intra_key(prior_context),
                    prior_context,
                    intra_mask,
                )

                prior_states = torch.stack(state_history, dim=1)
                inter_mask = prior_mask
                if self.inter_speaker_only:
                    inter_mask = inter_mask & prior_speakers.ne(current_speaker.unsqueeze(1))
                inter, inter_weight = masked_attention_pool(
                    self.inter_query(current),
                    self.inter_key(prior_states),
                    prior_states,
                    inter_mask,
                )
            # Hp in the paper includes h_i. The residual makes the current
            # utterance available even when there is no prior speaker history.
            raw_state = torch.tanh(
                self.combine(torch.cat([intra, inter], dim=-1)) + self.current_projection(current)
            )
            raw_state = torch.where(mask[:, t : t + 1], raw_state, torch.zeros_like(raw_state))
            state_history.append(raw_state)
            intra_weights.append(intra_weight)
            inter_weights.append(inter_weight)

            one_hot = F.one_hot(current_speaker, self.max_speakers).to(raw_state.dtype).unsqueeze(-1)
            update = one_hot * mask[:, t, None, None].to(raw_state.dtype)
            speaker_state = speaker_state * (1.0 - update) + raw_state.unsqueeze(1) * update

        raw_states = torch.stack(state_history, dim=1)
        attended, global_weights = self.self_attention(
            raw_states,
            raw_states,
            raw_states,
            attn_mask=causal_attention_mask(length, context.device),
            key_padding_mask=~mask,
            need_weights=return_aux,
        )
        output = self.norm1(raw_states + self.dropout(attended))
        output = self.norm2(output + self.dropout(self.ffn(output)))
        output = output * mask.unsqueeze(-1).to(output.dtype)
        if not return_aux:
            return output
        return output, {
            "intra_weights": intra_weights,
            "inter_weights": inter_weights,
            "global_weights": global_weights,
        }
