from __future__ import annotations

import torch
from torch import nn

from loopy.binary_codec_v2 import PAD_BYTE_ID
from loopy.symbolic_codec_v3 import SymbolicForward, V3PatchDecoder, V3PatchEncoder, VectorQuantizer
from loopy.v4_config import ContextualSymbolicCodecConfig


class PatchContextTransformer(nn.Module):
    def __init__(self, config: ContextualSymbolicCodecConfig, num_layers: int) -> None:
        super().__init__()
        self.enabled = num_layers > 0
        if not self.enabled:
            self.position_embedding = None
            self.encoder = None
            self.norm = None
            return

        self.position_embedding = nn.Parameter(torch.randn(config.num_patches, config.latent_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.latent_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, latents: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return latents
        padding_mask = patch_mask.eq(0)
        positions = self.position_embedding[: latents.size(1)].unsqueeze(0)
        contextual = self.encoder(latents + positions, src_key_padding_mask=padding_mask)
        return self.norm(contextual)


class SymbolicCodecV4(nn.Module):
    def __init__(self, config: ContextualSymbolicCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = V3PatchEncoder(config)
        self.pre_context = PatchContextTransformer(config, config.pre_context_layers)
        self.quantizer = VectorQuantizer(
            config.latent_dim,
            config.num_codebooks,
            config.sub_codebook_size,
            config.assignment_temp,
            config.commitment_weight,
            config.codebook_weight,
            config.usage_weight,
        )
        self.post_context = PatchContextTransformer(config, config.post_context_layers)
        self.decoder = V3PatchDecoder(config)

    def forward(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> SymbolicForward:
        local_latents = self.encoder(patch_ids)
        contextual_latents = self.pre_context(local_latents, patch_mask)
        symbol_ids, _quantized, st_quantized, commitment_loss, codebook_loss, usage_loss, perplexity = self.quantizer(
            contextual_latents,
            patch_mask,
        )
        decoded_latents = self.post_context(st_quantized, patch_mask)
        logits = self.decoder(decoded_latents, patch_ids)
        recon_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            patch_ids.reshape(-1),
            ignore_index=PAD_BYTE_ID,
        )

        # Predictive loss is intentionally disabled in the first v4 branch.
        # The new contextual encoder is bidirectional, so reusing the old v3
        # next-symbol objective would leak future information.
        predictive_loss = recon_loss.new_zeros(())

        return SymbolicForward(
            logits=logits,
            symbol_ids=symbol_ids,
            recon_loss=recon_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            usage_loss=usage_loss,
            predictive_loss=predictive_loss,
            codebook_perplexity=perplexity,
        )

    @torch.no_grad()
    def reconstruct(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        forward = self.forward(patch_ids, patch_mask)
        return forward.logits.argmax(dim=-1)
