from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class PriorAwareSymbolicCodecConfig:
    data_path: str = "loopy/example_corpus.txt"
    output_dir: str = "loopy/runs/v5_codec"
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_seq_len: int = 128
    patch_size: int = 4
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
    num_codebooks: int = 2
    sub_codebook_size: int = 256
    assignment_temp: float = 1.0
    commitment_weight: float = 0.25
    codebook_weight: float = 1.0
    usage_weight: float = 0.01
    use_residual_detail: bool = False
    residual_usage_weight: float = 0.01
    residual_gate_bias: float = -2.0
    prior_weight: float = 0.05
    prior_hidden_size: int = 128
    prior_num_layers: int = 2
    prior_dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return self.max_seq_len // self.patch_size

    @property
    def raw_capacity_bpb(self) -> float:
        return (self.num_codebooks * math.ceil(math.log2(self.sub_codebook_size))) / self.patch_size

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["num_patches"] = self.num_patches
        payload["raw_capacity_bpb"] = self.raw_capacity_bpb
        return payload
