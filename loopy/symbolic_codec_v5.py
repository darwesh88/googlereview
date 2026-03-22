from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from loopy.binary_codec_v2 import BYTE_VOCAB_SIZE, PAD_BYTE_ID
from loopy.symbolic_codec_v3 import V3PatchEncoder
from loopy.symbolic_codec_v4 import PatchContextTransformer, V4PatchDecoder
from loopy.v5_config import PriorAwareSymbolicCodecConfig


class PriorAwareForward:
    def __init__(
        self,
        *,
        logits: torch.Tensor,
        symbol_ids: torch.Tensor,
        recon_loss: torch.Tensor,
        commitment_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
        usage_loss: torch.Tensor,
        residual_usage_loss: torch.Tensor,
        prior_match_loss: torch.Tensor,
        prior_ce_loss: torch.Tensor,
        prior_bpb: torch.Tensor,
        codebook_perplexity: torch.Tensor,
    ) -> None:
        self.logits = logits
        self.symbol_ids = symbol_ids
        self.recon_loss = recon_loss
        self.commitment_loss = commitment_loss
        self.codebook_loss = codebook_loss
        self.usage_loss = usage_loss
        self.residual_usage_loss = residual_usage_loss
        self.prior_match_loss = prior_match_loss
        self.prior_ce_loss = prior_ce_loss
        self.prior_bpb = prior_bpb
        self.codebook_perplexity = codebook_perplexity

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            self.recon_loss
            + self.commitment_loss
            + self.codebook_loss
            + self.usage_loss
            + self.residual_usage_loss
            + self.prior_match_loss
        )


class V5VectorQuantizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        assignment_temp: float,
        commitment_weight: float,
        codebook_weight: float,
        usage_weight: float,
    ) -> None:
        super().__init__()
        if latent_dim % num_codebooks != 0:
            raise ValueError("latent_dim must be divisible by num_codebooks")
        self.sub_dim = latent_dim // num_codebooks
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.assignment_temp = assignment_temp
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.usage_weight = usage_weight
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, self.sub_dim) for _ in range(num_codebooks)])
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self,
        latents: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_patches, _ = latents.shape
        split_latents = latents.view(batch_size, num_patches, self.num_codebooks, self.sub_dim)
        active = patch_mask.unsqueeze(-1)

        all_symbol_ids = []
        hard_quantized_chunks = []
        soft_quantized_chunks = []
        soft_assignments = []
        all_commit_losses = []
        all_codebook_losses = []
        usage_losses = []
        perplexities = []

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
            logits = -distances / self.assignment_temp
            probs = F.softmax(logits, dim=-1).reshape(batch_size, num_patches, self.codebook_size)
            hard_quantized = codebook(symbol_ids)
            soft_quantized = probs @ entries

            all_symbol_ids.append(symbol_ids)
            hard_quantized_chunks.append(hard_quantized)
            soft_quantized_chunks.append(soft_quantized)
            soft_assignments.append(probs)
            all_commit_losses.append(F.mse_loss(sub_latents, soft_quantized.detach()))
            all_codebook_losses.append(F.mse_loss(soft_quantized, sub_latents.detach()))

            avg_probs = (probs * active).sum(dim=(0, 1)) / active.sum().clamp_min(1.0)
            usage_losses.append(
                (
                    avg_probs
                    * (avg_probs.clamp_min(1e-8).log() + torch.log(avg_probs.new_tensor(self.codebook_size)))
                ).sum()
            )
            perplexities.append(torch.exp(-(avg_probs * (avg_probs + 1e-8).log()).sum()))

        symbol_ids = torch.stack(all_symbol_ids, dim=-1)
        hard_quantized = torch.cat(hard_quantized_chunks, dim=-1)
        soft_quantized = torch.cat(soft_quantized_chunks, dim=-1)
        st_quantized = hard_quantized + (soft_quantized - soft_quantized.detach())
        commitment_loss = torch.stack(all_commit_losses).mean() * self.commitment_weight
        codebook_loss = torch.stack(all_codebook_losses).mean() * self.codebook_weight
        usage_loss = torch.stack(usage_losses).mean() * self.usage_weight
        perplexity = torch.stack(perplexities).mean()
        assignment_probs = torch.stack(soft_assignments, dim=2)
        return (
            symbol_ids,
            hard_quantized,
            st_quantized,
            assignment_probs,
            commitment_loss,
            codebook_loss,
            usage_loss,
            perplexity,
        )


class CausalGroupedPriorHead(nn.Module):
    def __init__(self, config: PriorAwareSymbolicCodecConfig) -> None:
        super().__init__()
        gru_dropout = config.prior_dropout if config.prior_num_layers > 1 else 0.0
        self.group_embeddings = nn.ModuleList(
            [nn.Embedding(config.sub_codebook_size, config.prior_hidden_size) for _ in range(config.num_codebooks)]
        )
        self.input_proj = nn.Linear(config.prior_hidden_size * config.num_codebooks, config.prior_hidden_size)
        self.gru = nn.GRU(
            config.prior_hidden_size,
            config.prior_hidden_size,
            num_layers=config.prior_num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(config.prior_dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(config.prior_hidden_size, config.sub_codebook_size) for _ in range(config.num_codebooks)]
        )

    def forward(self, prev_symbol_ids: torch.Tensor) -> list[torch.Tensor]:
        embedded_groups = [
            embedding(prev_symbol_ids[..., group_index])
            for group_index, embedding in enumerate(self.group_embeddings)
        ]
        hidden_inputs = torch.cat(embedded_groups, dim=-1)
        hidden, _ = self.gru(self.input_proj(hidden_inputs))
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.heads]


def compute_grouped_prior_metrics(
    logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    byte_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not logits_list:
        zero = targets.new_zeros((), dtype=torch.float32)
        return zero, zero, zero

    masked_losses = []
    correct = 0.0
    valid_groups = 0.0

    for group_index, logits in enumerate(logits_list):
        group_targets = targets[..., group_index]
        group_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            group_targets.reshape(-1),
            reduction="none",
        ).reshape_as(group_targets)
        masked_group_loss = group_loss * mask
        masked_losses.append(masked_group_loss)

        predictions = logits.argmax(dim=-1)
        valid = mask.bool()
        correct += ((predictions == group_targets) & valid).sum().item()
        valid_groups += valid.sum().item()

    total_loss = torch.stack(masked_losses, dim=0).sum(dim=0)
    total_group_tokens = (mask.sum() * len(logits_list)).clamp_min(1.0)
    mean_loss = total_loss.sum() / total_group_tokens
    accuracy = targets.new_tensor(correct / max(1.0, valid_groups), dtype=torch.float32)
    total_nll_bits = total_loss.sum() / math.log(2.0)
    total_bytes = (byte_counts * mask).sum().clamp_min(1.0)
    bpb = total_nll_bits / total_bytes
    return mean_loss, accuracy, bpb


class SymbolicCodecV5(nn.Module):
    def __init__(self, config: PriorAwareSymbolicCodecConfig) -> None:
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
        self.prior_head = CausalGroupedPriorHead(config)
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

        prev_symbol_ids = symbol_ids[:, :-1].detach()
        target_symbol_ids = symbol_ids[:, 1:]
        target_probs = assignment_probs[:, 1:]
        pair_mask = patch_mask[:, :-1] * patch_mask[:, 1:]
        byte_counts = patch_ids.ne(PAD_BYTE_ID).sum(dim=-1).to(torch.float32)[:, 1:]

        logits_list = self.prior_head(prev_symbol_ids)
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
