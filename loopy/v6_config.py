from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class DynamicSymbolicCodecConfig:
    data_path: str = "loopy/example_corpus.txt"
    output_dir: str = "loopy/runs/v6_codec"
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_seq_len: int = 128
    max_patches: int = 64
    max_patch_size: int = 8
    min_patch_size: int = 1
    patching_mode: str = "boundary"
    embed_dim: int = 128
    latent_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    pre_context_layers: int = 1
    post_context_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0
    val_ratio: float = 0.1
    seed: int = 7
    device: str = "auto"
    overfit_all: bool = False
    num_codebooks: int = 4
    sub_codebook_size: int = 64
    assignment_temp: float = 0.5
    commitment_weight: float = 0.05
    codebook_weight: float = 0.25
    usage_weight: float = 0.05
    use_residual_detail: bool = False
    residual_usage_weight: float = 0.005
    residual_gate_bias: float = -2.0

    @property
    def num_patches(self) -> int:
        return self.max_patches

    @property
    def code_bits_per_patch(self) -> int:
        return self.num_codebooks * math.ceil(math.log2(self.sub_codebook_size))

    @property
    def min_capacity_bpb(self) -> float:
        return self.code_bits_per_patch / self.max_patch_size

    @property
    def max_capacity_bpb(self) -> float:
        return self.code_bits_per_patch / self.min_patch_size

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["num_patches"] = self.num_patches
        payload["code_bits_per_patch"] = self.code_bits_per_patch
        payload["min_capacity_bpb"] = self.min_capacity_bpb
        payload["max_capacity_bpb"] = self.max_capacity_bpb
        return payload
