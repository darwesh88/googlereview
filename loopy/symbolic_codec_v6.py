from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from loopy.binary_codec_v2 import BYTE_VOCAB_SIZE, PAD_BYTE_ID
from loopy.symbolic_codec_v4 import PatchContextTransformer
from loopy.symbolic_codec_v5 import V5VectorQuantizer
from loopy.v6_config import DynamicSymbolicCodecConfig


@dataclass
class DynamicSymbolicForward:
    logits: torch.Tensor
    symbol_ids: torch.Tensor
    recon_loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    usage_loss: torch.Tensor
    residual_usage_loss: torch.Tensor
    codebook_perplexity: torch.Tensor

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            self.recon_loss
            + self.commitment_loss
            + self.codebook_loss
            + self.usage_loss
            + self.residual_usage_loss
        )


class DynamicPatchEncoder(nn.Module):
    def __init__(self, config: DynamicSymbolicCodecConfig) -> None:
        super().__init__()
        self.max_patch_size = config.max_patch_size
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.embed_dim, padding_idx=PAD_BYTE_ID)
        self.position_embedding = nn.Parameter(torch.randn(config.max_patch_size, config.embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(config.embed_dim * config.max_patch_size),
            nn.Linear(config.embed_dim * config.max_patch_size, config.latent_dim * 2),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

    def forward(self, patch_ids: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, max_patch_size = patch_ids.shape
        flat_ids = patch_ids.reshape(batch_size * num_patches, max_patch_size)
        flat_byte_mask = byte_mask.reshape(batch_size * num_patches, max_patch_size)
        padding_mask = flat_byte_mask.eq(0)
        safe_padding_mask = padding_mask.clone()
        all_padding = safe_padding_mask.all(dim=1)
        safe_padding_mask[all_padding] = False

        embedded = self.byte_embedding(flat_ids) + self.position_embedding.unsqueeze(0)
        encoded = self.encoder(embedded, src_key_padding_mask=safe_padding_mask)
        valid_mask = flat_byte_mask.unsqueeze(-1)
        flattened = (encoded * valid_mask).reshape(batch_size * num_patches, max_patch_size * encoded.size(-1))
        latents = self.to_latent(flattened)
        return latents.reshape(batch_size, num_patches, -1)


class DynamicPatchDecoder(nn.Module):
    def __init__(self, config: DynamicSymbolicCodecConfig) -> None:
        super().__init__()
        self.max_patch_size = config.max_patch_size
        self.position_embedding = nn.Parameter(torch.randn(config.max_patch_size, config.embed_dim) * 0.02)
        self.expand = nn.Sequential(
            nn.Linear(config.latent_dim, config.embed_dim * config.max_patch_size),
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

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, latent_dim = latents.shape
        flat = latents.reshape(batch_size * num_patches, latent_dim)
        expanded = self.expand(flat).reshape(batch_size * num_patches, self.max_patch_size, -1)
        hidden = self.decoder(expanded + self.position_embedding.unsqueeze(0))
        logits = self.output(hidden)
        hidden = hidden.reshape(batch_size, num_patches, self.max_patch_size, -1)
        logits = logits.reshape(batch_size, num_patches, self.max_patch_size, BYTE_VOCAB_SIZE)
        return hidden, logits


class SymbolicCodecV6(nn.Module):
    def __init__(self, config: DynamicSymbolicCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DynamicPatchEncoder(config)
        self.pre_context = PatchContextTransformer(config, config.pre_context_layers)
        self.quantizer = V5VectorQuantizer(
            config.latent_dim,
            config.num_codebooks,
            config.sub_codebook_size,
            config.assignment_temp,
            config.commitment_weight,
            config.codebook_weight,
            config.usage_weight,
        )
        self.post_context = PatchContextTransformer(config, config.post_context_layers)
        self.decoder = DynamicPatchDecoder(config)
        self.use_residual_detail = config.use_residual_detail

        if self.use_residual_detail:
            self.residual_gate = nn.Linear(config.embed_dim, 1)
            nn.init.constant_(self.residual_gate.bias, config.residual_gate_bias)
            self.residual_head = nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, BYTE_VOCAB_SIZE),
            )
        else:
            self.residual_gate = None
            self.residual_head = None

    def forward(
        self,
        patch_ids: torch.Tensor,
        patch_mask: torch.Tensor,
        byte_mask: torch.Tensor,
    ) -> DynamicSymbolicForward:
        local_latents = self.encoder(patch_ids, byte_mask)
        contextual_latents = self.pre_context(local_latents, patch_mask)
        (
            symbol_ids,
            _hard_quantized,
            st_quantized,
            _assignment_probs,
            commitment_loss,
            codebook_loss,
            usage_loss,
            perplexity,
        ) = self.quantizer(contextual_latents, patch_mask)
        decoded_latents = self.post_context(st_quantized, patch_mask)
        byte_hidden, base_logits = self.decoder(decoded_latents)

        if self.use_residual_detail:
            gate = torch.sigmoid(self.residual_gate(byte_hidden))
            residual_logits = self.residual_head(byte_hidden)
            logits = base_logits + gate * residual_logits
            residual_usage_loss = (
                (gate.squeeze(-1) * byte_mask).sum() / byte_mask.sum().clamp_min(1.0)
            ) * self.config.residual_usage_weight
        else:
            logits = base_logits
            residual_usage_loss = base_logits.new_zeros(())

        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            patch_ids.reshape(-1),
            ignore_index=PAD_BYTE_ID,
        )
        return DynamicSymbolicForward(
            logits=logits,
            symbol_ids=symbol_ids,
            recon_loss=recon_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            usage_loss=usage_loss,
            residual_usage_loss=residual_usage_loss,
            codebook_perplexity=perplexity,
        )

    @torch.no_grad()
    def reconstruct(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        forward = self.forward(patch_ids, patch_mask, byte_mask)
        return forward.logits.argmax(dim=-1)
