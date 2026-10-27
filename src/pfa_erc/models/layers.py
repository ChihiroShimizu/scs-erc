from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * self.value(x)


class GatedResidualNetwork(nn.Module):
    """TFT-style GRN with optional static context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.context_layer = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.input_layer(x)
        if context is not None:
            if self.context_layer is None:
                raise ValueError("Context was supplied to a GRN without context_dim")
            while context.ndim < hidden.ndim:
                context = context.unsqueeze(1)
            hidden = hidden + self.context_layer(context)
        hidden = F.elu(hidden)
        hidden = self.dropout(self.hidden_layer(hidden))
        return self.norm(self.skip(x) + self.glu(hidden))


class VariableSelectionNetwork(nn.Module):
    """Select among semantically meaningful variable groups.

    The paper does not specify its actual covariate list. This implementation
    uses one GRN per variable group and produces instance-wise softmax weights.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dims = list(input_dims)
        self.variable_grns = nn.ModuleList(
            [GatedResidualNetwork(dim, hidden_dim, dropout=dropout) for dim in input_dims]
        )
        self.weight_grn = GatedResidualNetwork(
            sum(input_dims), hidden_dim, context_dim=context_dim, dropout=dropout
        )
        self.weight_head = nn.Linear(hidden_dim, len(input_dims))

    def forward(
        self, variables: Sequence[torch.Tensor], context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(variables) != len(self.input_dims):
            raise ValueError(f"Expected {len(self.input_dims)} variables, got {len(variables)}")
        joint = torch.cat(list(variables), dim=-1)
        weights = torch.softmax(self.weight_head(self.weight_grn(joint, context)), dim=-1)
        transformed = torch.stack(
            [grn(variable) for grn, variable in zip(self.variable_grns, variables)], dim=-2
        )
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights


def masked_attention_pool(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product pooling that safely handles an all-masked history."""
    scale = max(query.shape[-1], 1) ** -0.5
    scores = torch.einsum("bd,btd->bt", query, keys) * scale
    scores = scores.masked_fill(~mask, -1e4)
    weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pooled = torch.einsum("bt,btd->bd", weights, values)
    return pooled, weights


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def causal_padding_attention_mask(
    valid_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Combine causal and padding masks without creating all-masked queries.

    ``nn.MultiheadAttention`` may return NaN when every key is hidden for a
    query. This occurs with left-padded histories: an early padded query can
    only see earlier padded keys under a causal mask. For those padded queries
    only, expose the first real key as a numerical sentinel. Callers must still
    zero padded query outputs after attention. Valid queries keep the exact
    causal and key-padding restrictions, so no future information is exposed
    to any real history position.

    The returned boolean mask has the 3-D shape expected by PyTorch MHA:
    ``[batch * num_heads, query_length, key_length]``.
    """
    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must be [B,L], got {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have dtype torch.bool")
    if num_heads < 1:
        raise ValueError("num_heads must be at least 1")

    batch_size, length = valid_mask.shape
    if length == 0:
        raise ValueError("Attention sequences must contain at least one position")
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("Every attention sample must contain at least one valid key")

    combined = causal_attention_mask(length, valid_mask.device).unsqueeze(0).expand(
        batch_size, -1, -1
    ).clone()
    combined = combined | (~valid_mask).unsqueeze(1)

    # A padded query is discarded after attention, so letting it attend one
    # real key cannot leak information into any valid query. It only prevents
    # softmax from receiving a row made entirely of -inf values.
    padded_batch, padded_query = torch.nonzero(~valid_mask, as_tuple=True)
    if padded_batch.numel() > 0:
        first_valid_key = valid_mask.to(torch.int64).argmax(dim=1)
        combined[
            padded_batch,
            padded_query,
            first_valid_key[padded_batch],
        ] = False

    return combined.repeat_interleave(num_heads, dim=0)
