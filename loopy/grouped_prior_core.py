from __future__ import annotations

import torch
from torch import nn


class GroupedPriorCore(nn.Module):
    def __init__(
        self,
        group_vocab_sizes: list[int],
        group_embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.group_vocab_sizes = group_vocab_sizes
        self.group_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, group_embed_dim) for vocab_size in group_vocab_sizes]
        )
        self.input_proj = nn.Linear(group_embed_dim * len(group_vocab_sizes), hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size) for vocab_size in group_vocab_sizes])

    def embed_ids(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded_groups = [
            embedding(inputs[..., group_index])
            for group_index, embedding in enumerate(self.group_embeddings)
        ]
        return torch.cat(embedded_groups, dim=-1)

    def embed_probabilities(self, probability_groups: list[torch.Tensor]) -> torch.Tensor:
        embedded_groups = [
            probs @ embedding.weight
            for probs, embedding in zip(probability_groups, self.group_embeddings)
        ]
        return torch.cat(embedded_groups, dim=-1)

    def forward_embedded(self, embedded_inputs: torch.Tensor) -> list[torch.Tensor]:
        hidden, _ = self.gru(self.input_proj(embedded_inputs))
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.heads]

    def forward_ids(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_embedded(self.embed_ids(inputs))

    def forward_probabilities(self, probability_groups: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.forward_embedded(self.embed_probabilities(probability_groups))


class HardGroupedPrior(nn.Module):
    def __init__(
        self,
        group_vocab_sizes: list[int],
        group_embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.core = GroupedPriorCore(group_vocab_sizes, group_embed_dim, hidden_size, num_layers, dropout)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        return self.core.forward_ids(inputs)
