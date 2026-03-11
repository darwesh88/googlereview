from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from loopy.binary_codec_v2 import BYTE_VOCAB_SIZE, PAD_BYTE_ID
from loopy.v3_config import SymbolicCodecConfig


@dataclass
class SymbolicForward:
    logits: torch.Tensor
    symbol_ids: torch.Tensor
    recon_loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    predictive_loss: torch.Tensor
    codebook_perplexity: torch.Tensor

    @property
    def total_loss(self) -> torch.Tensor:
        return self.recon_loss + self.commitment_loss + self.codebook_loss + self.predictive_loss


class V3PatchEncoder(nn.Module):
    def __init__(self, config: SymbolicCodecConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.embed_dim, padding_idx=PAD_BYTE_ID)
        self.position_embedding = nn.Parameter(torch.randn(config.patch_size, config.embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(config.embed_dim * config.patch_size),
            nn.Linear(config.embed_dim * config.patch_size, config.latent_dim * 2),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

    def forward(self, patch_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, patch_size = patch_ids.shape
        flat = patch_ids.reshape(batch_size * num_patches, patch_size)
        padding_mask = flat.eq(PAD_BYTE_ID)
        embedded = self.byte_embedding(flat) + self.position_embedding.unsqueeze(0)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).float().unsqueeze(-1)
        flattened = (encoded * valid_mask).reshape(batch_size * num_patches, patch_size * encoded.size(-1))
        latents = self.to_latent(flattened)
        return latents.reshape(batch_size, num_patches, -1)


class V3PatchDecoder(nn.Module):
    def __init__(self, config: SymbolicCodecConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.position_embedding = nn.Parameter(torch.randn(config.patch_size, config.embed_dim) * 0.02)
        self.expand = nn.Sequential(
            nn.Linear(config.latent_dim, config.embed_dim * config.patch_size),
            nn.GELU(),
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.decoder_layers)
        self.output = nn.Linear(config.embed_dim, BYTE_VOCAB_SIZE)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, latent_dim = latents.shape
        flat = latents.reshape(batch_size * num_patches, latent_dim)
        expanded = self.expand(flat).reshape(batch_size * num_patches, self.patch_size, -1)
        decoded = self.decoder(expanded + self.position_embedding.unsqueeze(0))
        logits = self.output(decoded)
        return logits.reshape(batch_size, num_patches, self.patch_size, BYTE_VOCAB_SIZE)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_weight: float,
        codebook_weight: float,
    ) -> None:
        super().__init__()
        if latent_dim % num_codebooks != 0:
            raise ValueError("latent_dim must be divisible by num_codebooks")
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.sub_dim = latent_dim // num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, self.sub_dim) for _ in range(num_codebooks)])
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self,
        latents: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_patches, latent_dim = latents.shape
        split_latents = latents.view(batch_size, num_patches, self.num_codebooks, self.sub_dim)

        all_symbol_ids = []
        all_quantized = []
        all_commit_losses = []
        all_codebook_losses = []
        perplexities = []
        active = patch_mask.unsqueeze(-1)

        for codebook_index, codebook in enumerate(self.codebooks):
            sub_latents = split_latents[:, :, codebook_index, :]
            flat = sub_latents.reshape(-1, self.sub_dim)
            entries = codebook.weight
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                + entries.pow(2).sum(dim=1)
                - 2.0 * flat @ entries.t()
            )
            symbol_ids = distances.argmin(dim=1).reshape(batch_size, num_patches)
            quantized = codebook(symbol_ids)

            all_symbol_ids.append(symbol_ids)
            all_quantized.append(quantized)
            all_commit_losses.append(F.mse_loss(sub_latents, quantized.detach()))
            all_codebook_losses.append(F.mse_loss(quantized, sub_latents.detach()))

            one_hot = F.one_hot(symbol_ids, num_classes=self.codebook_size).to(latents.dtype)
            avg_probs = (one_hot * active).sum(dim=(0, 1)) / active.sum().clamp_min(1.0)
            perplexities.append(torch.exp(-(avg_probs * (avg_probs + 1e-8).log()).sum()))

        symbol_ids = torch.stack(all_symbol_ids, dim=-1)
        quantized = torch.cat(all_quantized, dim=-1)
        st_quantized = latents + (quantized - latents).detach()
        commitment_loss = torch.stack(all_commit_losses).mean() * self.commitment_weight
        codebook_loss = torch.stack(all_codebook_losses).mean() * self.codebook_weight
        perplexity = torch.stack(perplexities).mean()
        return symbol_ids, quantized, st_quantized, commitment_loss, codebook_loss, perplexity


class SymbolicCodecV3(nn.Module):
    def __init__(self, config: SymbolicCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = V3PatchEncoder(config)
        self.quantizer = VectorQuantizer(
            config.latent_dim,
            config.num_codebooks,
            config.sub_codebook_size,
            config.commitment_weight,
            config.codebook_weight,
        )
        self.decoder = V3PatchDecoder(config)
        self.next_symbol_predictor = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.latent_dim),
                    nn.Linear(config.latent_dim, config.latent_dim),
                    nn.GELU(),
                    nn.Linear(config.latent_dim, config.sub_codebook_size),
                )
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> SymbolicForward:
        latents = self.encoder(patch_ids)
        symbol_ids, quantized, st_quantized, commitment_loss, codebook_loss, perplexity = self.quantizer(latents, patch_mask)
        logits = self.decoder(st_quantized)

        recon_loss = F.cross_entropy(
            logits.reshape(-1, BYTE_VOCAB_SIZE),
            patch_ids.reshape(-1),
            ignore_index=PAD_BYTE_ID,
        )

        if st_quantized.size(1) > 1:
            pair_mask = patch_mask[:, :-1] * patch_mask[:, 1:]
            next_targets = symbol_ids[:, 1:].detach()
            all_next_losses = []
            for codebook_index, predictor in enumerate(self.next_symbol_predictor):
                next_logits = predictor(st_quantized[:, :-1])
                next_loss = F.cross_entropy(
                    next_logits.reshape(-1, next_logits.size(-1)),
                    next_targets[..., codebook_index].reshape(-1),
                    reduction="none",
                ).reshape_as(next_targets[..., codebook_index])
                all_next_losses.append((next_loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0))
            predictive_loss = torch.stack(all_next_losses).mean()
            predictive_loss = predictive_loss * self.config.predictive_weight
        else:
            predictive_loss = recon_loss.new_zeros(())

        return SymbolicForward(
            logits=logits,
            symbol_ids=symbol_ids,
            recon_loss=recon_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            predictive_loss=predictive_loss,
            codebook_perplexity=perplexity,
        )

    @torch.no_grad()
    def reconstruct(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        forward = self.forward(patch_ids, patch_mask)
        return forward.logits.argmax(dim=-1)
