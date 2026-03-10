from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from loopy.dataset import VOCAB_SIZE


@dataclass
class CodecOutput:
    logits: torch.Tensor
    code_indices: torch.Tensor
    vq_loss: torch.Tensor
    perplexity: torch.Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, embedding_dim: int, commitment_cost: float) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_codes, embedding_dim)
        self.commitment_cost = commitment_cost
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_inputs = inputs.reshape(-1, inputs.size(-1))
        codebook = self.codebook.weight

        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * flat_inputs @ codebook.t()
        )
        code_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(code_indices).view_as(inputs)

        commitment_loss = F.mse_loss(inputs, quantized.detach())
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = inputs + (quantized - inputs).detach()

        one_hot = F.one_hot(code_indices, num_classes=self.codebook.num_embeddings).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, code_indices.view(inputs.shape[0], inputs.shape[1]), vq_loss, perplexity

    def lookup(self, code_indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(code_indices)


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ChunkEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(ResidualBlock(d_model, dropout) for _ in range(max(1, num_layers)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, chunk_inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(chunk_inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)


class ChunkDecoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, chunk_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ResidualBlock(d_model, dropout) for _ in range(max(1, num_layers)))
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, chunk_size * VOCAB_SIZE)
        self.chunk_size = chunk_size

    def forward(self, chunk_codes: torch.Tensor) -> torch.Tensor:
        hidden = chunk_codes
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        logits = self.output(hidden)
        batch_size, num_chunks, _ = logits.shape
        logits = logits.view(batch_size, num_chunks, self.chunk_size, VOCAB_SIZE)
        return logits.reshape(batch_size, num_chunks * self.chunk_size, VOCAB_SIZE)


class LoopyCodec(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        chunk_size: int,
        d_model: int,
        encoder_layers: int,
        decoder_layers: int,
        num_heads: int,
        dropout: float,
        codebook_size: int,
        commitment_cost: float,
    ) -> None:
        super().__init__()
        del num_heads
        if max_seq_len % chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")

        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.max_chunks = max_seq_len // chunk_size

        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.encoder = ChunkEncoder(chunk_size * d_model, d_model, encoder_layers, dropout)
        self.quantizer = VectorQuantizer(codebook_size, d_model, commitment_cost)
        self.decoder = ChunkDecoder(d_model, decoder_layers, chunk_size, dropout)

    def _chunk_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(input_ids)
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        if seq_len != self.max_seq_len:
            raise ValueError(f"Expected seq_len={self.max_seq_len}, got {seq_len}")

        token_embeddings = token_embeddings.view(batch_size, self.max_chunks, self.chunk_size, hidden_dim)
        chunk_mask = attention_mask.view(batch_size, self.max_chunks, self.chunk_size, 1).float()
        masked = token_embeddings * chunk_mask
        return masked.reshape(batch_size, self.max_chunks, self.chunk_size * hidden_dim)

    def encode_to_codes(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        chunk_inputs = self._chunk_inputs(input_ids, attention_mask)
        encoded = self.encoder(chunk_inputs)
        _, code_indices, _, _ = self.quantizer(encoded)
        return code_indices

    def decode_from_codes(self, code_indices: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.lookup(code_indices)
        return self.decoder(quantized)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> CodecOutput:
        chunk_inputs = self._chunk_inputs(input_ids, attention_mask)
        encoded = self.encoder(chunk_inputs)
        quantized, code_indices, vq_loss, perplexity = self.quantizer(encoded)
        logits = self.decoder(quantized)
        return CodecOutput(
            logits=logits,
            code_indices=code_indices,
            vq_loss=vq_loss,
            perplexity=perplexity,
        )
