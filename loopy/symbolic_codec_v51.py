from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from loopy.binary_codec_v2 import BYTE_VOCAB_SIZE, PAD_BYTE_ID
from loopy.grouped_prior_core import GroupedPriorCore
from loopy.symbolic_codec_v3 import V3PatchEncoder
from loopy.symbolic_codec_v4 import PatchContextTransformer, V4PatchDecoder
from loopy.symbolic_codec_v5 import PriorAwareForward, V5VectorQuantizer, compute_grouped_prior_metrics
from loopy.v51_config import PriorAlignedSymbolicCodecConfig


class SymbolicCodecV51(nn.Module):
    def __init__(self, config: PriorAlignedSymbolicCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = V3PatchEncoder(config)
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
        self.decoder = V4PatchDecoder(config)
        self.prior_core = GroupedPriorCore(
            [config.sub_codebook_size] * config.num_codebooks,
            config.prior_group_embed_dim,
            config.prior_hidden_size,
            config.prior_num_layers,
            config.prior_dropout,
        )
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

    def _compute_prior_match(
        self,
        symbol_ids: torch.Tensor,
        assignment_probs: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if symbol_ids.size(1) <= 1 or self.config.prior_weight <= 0.0:
            zero = patch_ids.new_zeros((), dtype=torch.float32)
            return zero, zero, zero

        pair_mask = patch_mask[:, :-1] * patch_mask[:, 1:]
        byte_counts = patch_ids.ne(PAD_BYTE_ID).sum(dim=-1).to(torch.float32)[:, 1:]
        target_symbol_ids = symbol_ids[:, 1:]
        target_probs = assignment_probs[:, 1:]

        prev_probability_groups = [
            assignment_probs[:, :-1, codebook_index, :]
            for codebook_index in range(self.config.num_codebooks)
        ]
        logits_list = self.prior_core.forward_probabilities(prev_probability_groups)

        valid_weight = pair_mask.sum().clamp_min(1.0)
        kl_terms = []
        for codebook_index, logits in enumerate(logits_list):
            q = target_probs[..., codebook_index, :].clamp_min(1e-8)
            log_q = q.log()
            log_p = F.log_softmax(logits, dim=-1)
            kl = (q * (log_q - log_p)).sum(dim=-1)
            kl_terms.append((kl * pair_mask).sum() / valid_weight)

        prior_match_loss = torch.stack(kl_terms).mean() * self.config.prior_weight
        prior_ce_loss, _prior_accuracy, prior_bpb = compute_grouped_prior_metrics(
            logits_list,
            target_symbol_ids,
            pair_mask,
            byte_counts,
        )
        return prior_match_loss, prior_ce_loss, prior_bpb

    def forward(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> PriorAwareForward:
        local_latents = self.encoder(patch_ids)
        contextual_latents = self.pre_context(local_latents, patch_mask)
        (
            symbol_ids,
            _hard_quantized,
            st_quantized,
            assignment_probs,
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
        prior_match_loss, prior_ce_loss, prior_bpb = self._compute_prior_match(
            symbol_ids,
            assignment_probs,
            patch_mask,
            patch_ids,
        )

        return PriorAwareForward(
            logits=logits,
            symbol_ids=symbol_ids,
            recon_loss=recon_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            usage_loss=usage_loss,
            residual_usage_loss=residual_usage_loss,
            prior_match_loss=prior_match_loss,
            prior_ce_loss=prior_ce_loss,
            prior_bpb=prior_bpb,
            codebook_perplexity=perplexity,
        )

    @torch.no_grad()
    def reconstruct(self, patch_ids: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        forward = self.forward(patch_ids, patch_mask)
        return forward.logits.argmax(dim=-1)
