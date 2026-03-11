from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from loopy.v2_config import BinaryCodecConfig

BYTE_VOCAB_SIZE = 257
PAD_BYTE_ID = 256


@dataclass
class CodecForward:
    logits: torch.Tensor
    bit_probs: torch.Tensor
    bit_values: torch.Tensor
    recon_loss: torch.Tensor
    rate_loss: torch.Tensor
    balance_loss: torch.Tensor
    align_loss: torch.Tensor
    predictive_loss: torch.Tensor

    @property
    def total_loss(self) -> torch.Tensor:
        return self.recon_loss + self.rate_loss + self.balance_loss + self.align_loss + self.predictive_loss


class GroupedBinaryQuantizer(nn.Module):
    def __init__(self, latent_dim: int, total_bits: int) -> None:
        super().__init__()
        self.to_bits = nn.Linear(latent_dim, total_bits)
        self.from_bits = nn.Linear(total_bits, latent_dim)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bit_logits = self.to_bits(latents)
        bit_probs = torch.sigmoid(bit_logits)
        hard_bits = (bit_probs > 0.5).float()
        st_bits = hard_bits + bit_probs - bit_probs.detach()
        restored = self.from_bits(st_bits * 2.0 - 1.0)
        return bit_probs, st_bits, restored


class PatchEncoder(nn.Module):
    def __init__(self, config: BinaryCodecConfig) -> None:
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
        # Preserve within-patch byte order instead of collapsing positions with a mean.
        flattened = (encoded * valid_mask).reshape(batch_size * num_patches, patch_size * encoded.size(-1))
        latents = self.to_latent(flattened)
        return latents.reshape(batch_size, num_patches, -1)


class PatchDecoder(nn.Module):
    def __init__(self, config: BinaryCodecConfig) -> None:
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


class SemanticBinaryCodec(nn.Module):
    def __init__(self, config: BinaryCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PatchEncoder(config)
        self.quantizer = GroupedBinaryQuantizer(config.latent_dim, config.total_bits)
        self.decoder = PatchDecoder(config)
        self.next_bit_predictor = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.total_bits),
        )

    def forward(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> CodecForward:
        latents = self.encoder(patch_ids)
        bit_probs, bit_values, restored = self.quantizer(latents)
        decoded_latents = restored
        logits = self.decoder(decoded_latents)

        recon_loss = F.cross_entropy(
            logits.reshape(-1, BYTE_VOCAB_SIZE),
            patch_ids.reshape(-1),
            ignore_index=PAD_BYTE_ID,
        )

        eps = 1e-6
        entropy = -(
            bit_probs.clamp(eps, 1 - eps) * torch.log2(bit_probs.clamp(eps, 1 - eps))
            + (1 - bit_probs).clamp(eps, 1 - eps) * torch.log2((1 - bit_probs).clamp(eps, 1 - eps))
        )
        rate_loss = entropy.mean() * self.config.rate_weight

        patch_presence = patch_mask.unsqueeze(-1)
        active_probs = bit_probs * patch_presence
        denom = patch_presence.sum().clamp_min(1.0)
        mean_prob = active_probs.sum(dim=(0, 1)) / denom
        balance_loss = ((mean_prob - 0.5) ** 2).mean() * self.config.balance_weight

        align_error = ((restored - latents.detach()) ** 2) * patch_presence
        align_loss = align_error.mean() * self.config.align_weight

        if decoded_latents.size(1) > 1:
            pair_mask = (patch_mask[:, :-1] * patch_mask[:, 1:]).unsqueeze(-1)
            next_bit_logits = self.next_bit_predictor(decoded_latents[:, :-1])
            next_bit_targets = bit_values[:, 1:].detach()
            predictive_error = F.binary_cross_entropy_with_logits(
                next_bit_logits,
                next_bit_targets,
                reduction="none",
            )
            predictive_denom = (pair_mask.sum() * next_bit_logits.size(-1)).clamp_min(1.0)
            predictive_loss = (predictive_error * pair_mask).sum() / predictive_denom
            predictive_loss = predictive_loss * self.config.predictive_weight
        else:
            predictive_loss = recon_loss.new_zeros(())

        return CodecForward(
            logits=logits,
            bit_probs=bit_probs,
            bit_values=bit_values,
            recon_loss=recon_loss,
            rate_loss=rate_loss,
            balance_loss=balance_loss,
            align_loss=align_loss,
            predictive_loss=predictive_loss,
        )

    @torch.no_grad()
    def reconstruct(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        forward = self.forward(patch_ids, patch_mask)
        return forward.logits.argmax(dim=-1)


def tensor_bit_density(bit_values: torch.Tensor, patch_mask: torch.Tensor) -> float:
    active = patch_mask.unsqueeze(-1)
    total = active.sum().item() * bit_values.size(-1)
    if total <= 0:
        return 0.0
    return float((bit_values * active).sum().item() / total)


def estimated_patch_bpb(bit_probs: torch.Tensor, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> float:
    eps = 1e-6
    entropy = -(
        bit_probs.clamp(eps, 1 - eps) * torch.log2(bit_probs.clamp(eps, 1 - eps))
        + (1 - bit_probs).clamp(eps, 1 - eps) * torch.log2((1 - bit_probs).clamp(eps, 1 - eps))
    )
    active = patch_mask.unsqueeze(-1)
    bits = (entropy * active).sum().item()
    active_bytes = (patch_ids != PAD_BYTE_ID).sum().item()
    if active_bytes <= 0:
        return 0.0
    return float(bits / active_bytes)

