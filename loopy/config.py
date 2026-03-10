from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    data_path: str = "loopy/example_corpus.txt"
    output_dir: str = "loopy/runs/codec"
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_seq_len: int = 128
    chunk_size: int = 8
    d_model: int = 192
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.1
    codebook_size: int = 1024
    commitment_cost: float = 0.25
    gradient_clip_norm: float = 1.0
    val_ratio: float = 0.1
    seed: int = 7
    device: str = "auto"
    overfit_all: bool = False

    def validate(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_seq_len % self.chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 < self.val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a discrete codec for Loopy.")
    parser.add_argument("--data-path", default=TrainConfig.data_path)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--max-seq-len", type=int, default=TrainConfig.max_seq_len)
    parser.add_argument("--chunk-size", type=int, default=TrainConfig.chunk_size)
    parser.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--encoder-layers", type=int, default=TrainConfig.encoder_layers)
    parser.add_argument("--decoder-layers", type=int, default=TrainConfig.decoder_layers)
    parser.add_argument("--num-heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--codebook-size", type=int, default=TrainConfig.codebook_size)
    parser.add_argument("--commitment-cost", type=float, default=TrainConfig.commitment_cost)
    parser.add_argument("--gradient-clip-norm", type=float, default=TrainConfig.gradient_clip_norm)
    parser.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--overfit-all", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        codebook_size=args.codebook_size,
        commitment_cost=args.commitment_cost,
        gradient_clip_norm=args.gradient_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        overfit_all=args.overfit_all,
    )
    config.validate()
    return config
