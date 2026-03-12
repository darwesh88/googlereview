from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from loopy.binary_codec_v2 import BYTE_VOCAB_SIZE, PAD_BYTE_ID
from loopy.symbolic_codec_v3 import V3PatchEncoder, VectorQuantizer
from loopy.v4_config import ContextualSymbolicCodecConfig


class ContextualForward:
    def __init__(
        self,
        *,
        logits: torch.Tensor,
        symbol_ids: torch.Tensor,
        recon_loss: torch.Tensor,
        commitment_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
        usage_loss: torch.Tensor,
        predictive_loss: torch.Tensor,
        residual_usage_loss: torch.Tensor,
        codebook_perplexity: torch.Tensor,
    ) -> None:
        self.logits = logits
        self.symbol_ids = symbol_ids
        self.recon_loss = recon_loss
        self.commitment_loss = commitment_loss
        self.codebook_loss = codebook_loss
        self.usage_loss = usage_loss
        self.predictive_loss = predictive_loss
        self.residual_usage_loss = residual_usage_loss
        self.codebook_perplexity = codebook_perplexity

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            self.recon_loss
            + self.commitment_loss
            + self.codebook_loss
            + self.usage_loss
            + self.predictive_loss
            + self.residual_usage_loss
        )


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


class V4PatchDecoder(nn.Module):
    def __init__(self, config: ContextualSymbolicCodecConfig) -> None:
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

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, latent_dim = latents.shape
        flat = latents.reshape(batch_size * num_patches, latent_dim)
        expanded = self.expand(flat).reshape(batch_size * num_patches, self.patch_size, -1)
        hidden = self.decoder(expanded + self.position_embedding.unsqueeze(0))
        logits = self.output(hidden)
        hidden = hidden.reshape(batch_size, num_patches, self.patch_size, -1)
        logits = logits.reshape(batch_size, num_patches, self.patch_size, BYTE_VOCAB_SIZE)
        return hidden, logits


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
        self.decoder = V4PatchDecoder(config)
        self.use_residual_detail = config.use_residual_detail
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.latent_dim) * 0.02)
        self.predictive_context = PatchContextTransformer(config, config.post_context_layers)
        self.masked_symbol_predictor = nn.ModuleList(
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

    def _sample_predictive_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        if self.config.predictive_weight <= 0.0:
            return patch_mask.new_zeros(patch_mask.shape, dtype=torch.bool)
        valid = patch_mask.gt(0)
        sampled = (torch.rand_like(patch_mask) < self.config.predictive_mask_prob) & valid
        for batch_index in range(sampled.size(0)):
            if sampled[batch_index].any() or not valid[batch_index].any():
                continue
            valid_positions = valid[batch_index].nonzero(as_tuple=False).flatten()
            sampled[batch_index, valid_positions[0]] = True
        return sampled

    def forward(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> ContextualForward:
        local_latents = self.encoder(patch_ids)
        contextual_latents = self.pre_context(local_latents, patch_mask)
        symbol_ids, _quantized, st_quantized, commitment_loss, codebook_loss, usage_loss, perplexity = self.quantizer(
            contextual_latents,
            patch_mask,
        )
        decoded_latents = self.post_context(st_quantized, patch_mask)
        byte_hidden, base_logits = self.decoder(decoded_latents)

        if self.use_residual_detail:
            gate = torch.sigmoid(self.residual_gate(byte_hidden))
            residual_logits = self.residual_head(byte_hidden)
            logits = base_logits + gate * residual_logits
            valid_mask = patch_ids.ne(PAD_BYTE_ID).float()
            residual_usage_loss = (
                (gate.squeeze(-1) * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
            ) * self.config.residual_usage_weight
        else:
            logits = base_logits
            residual_usage_loss = base_logits.new_zeros(())

        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            patch_ids.reshape(-1),
            ignore_index=PAD_BYTE_ID,
        )

        predictive_mask = self._sample_predictive_mask(patch_mask)
        if predictive_mask.any():
            masked_quantized = st_quantized.masked_fill(predictive_mask.unsqueeze(-1), 0.0)
            masked_quantized = masked_quantized + predictive_mask.unsqueeze(-1) * self.mask_token
            predictive_latents = self.predictive_context(masked_quantized, patch_mask)
            masked_targets = symbol_ids.detach()
            masked_positions = predictive_mask.float()
            per_codebook_losses = []
            for codebook_index, predictor in enumerate(self.masked_symbol_predictor):
                next_logits = predictor(predictive_latents)
                codebook_loss_value = F.cross_entropy(
                    next_logits.reshape(-1, next_logits.size(-1)),
                    masked_targets[..., codebook_index].reshape(-1),
                    reduction="none",
                ).reshape_as(masked_positions)
                per_codebook_losses.append(
                    (codebook_loss_value * masked_positions).sum() / masked_positions.sum().clamp_min(1.0)
                )
            predictive_loss = torch.stack(per_codebook_losses).mean() * self.config.predictive_weight
        else:
            predictive_loss = recon_loss.new_zeros(())

        return ContextualForward(
            logits=logits,
            symbol_ids=symbol_ids,
            recon_loss=recon_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            usage_loss=usage_loss,
            predictive_loss=predictive_loss,
            residual_usage_loss=residual_usage_loss,
            codebook_perplexity=perplexity,
        )

    @torch.no_grad()
    def reconstruct(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        forward = self.forward(patch_ids, patch_mask)
        return forward.logits.argmax(dim=-1)
