from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BinaryCodecConfig:
    data_path: str = "loopy/example_corpus.txt"
    output_dir: str = "loopy/runs/v2_codec"
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_seq_len: int = 128
    patch_size: int = 16
    embed_dim: int = 128
    latent_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0
    val_ratio: float = 0.1
    seed: int = 7
    device: str = "auto"
    overfit_all: bool = False
    bit_groups: tuple[int, ...] = (4, 8, 8, 8)
    rate_weight: float = 0.0
    balance_weight: float = 0.01
    align_weight: float = 0.1
    predictive_weight: float = 0.0

    @property
    def total_bits(self) -> int:
        return sum(self.bit_groups)

    @property
    def num_patches(self) -> int:
        return self.max_seq_len // self.patch_size

    @property
    def raw_capacity_bpb(self) -> float:
        return self.total_bits / self.patch_size

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["total_bits"] = self.total_bits
        payload["num_patches"] = self.num_patches
        payload["raw_capacity_bpb"] = self.raw_capacity_bpb
        return payload
