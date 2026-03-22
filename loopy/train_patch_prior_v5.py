from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset

from loopy.binary_codec_v2 import PAD_BYTE_ID
from loopy.dataset import load_text_samples, split_samples
from loopy.symbolic_codec_v5 import SymbolicCodecV5
from loopy.train_binary_codec_v2 import encode_text_to_patches
from loopy.train_patch_prior_v2 import GroupedPatchPrior, compute_grouped_metrics
from loopy.v5_config import PriorAwareSymbolicCodecConfig


@dataclass
class V5PriorConfig:
    data_path: str
    output_dir: str
    codec_run_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    num_layers: int
    dropout: float
    gradient_clip_norm: float
    val_ratio: float
    seed: int
    device: str
    max_seq_len: int
    patch_size: int
    group_embed_dim: int
    batch_encode_size: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class V5GroupedPatchDataset(Dataset):
    def __init__(
        self,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
        masks: list[torch.Tensor],
        byte_counts: list[torch.Tensor],
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.masks = masks
        self.byte_counts = byte_counts

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[index],
            "targets": self.targets[index],
            "mask": self.masks[index],
            "byte_counts": self.byte_counts[index],
        }


def parse_args(argv: list[str] | None = None) -> V5PriorConfig:
    parser = argparse.ArgumentParser(description="Train a grouped prior over Loopy v5 patch symbols.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codec-run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--group-embed-dim", type=int, default=16)
    parser.add_argument("--batch-encode-size", type=int, default=32)
    args = parser.parse_args(argv)
    return V5PriorConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        codec_run_dir=args.codec_run_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        max_seq_len=args.max_seq_len,
        patch_size=args.patch_size,
        group_embed_dim=args.group_embed_dim,
        batch_encode_size=args.batch_encode_size,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_codec(run_dir: str, device: torch.device) -> tuple[SymbolicCodecV5, PriorAwareSymbolicCodecConfig]:
    run_path = Path(run_dir)
    payload = torch.load(run_path / "v5_codec.pt", map_location=device)
    config_dict = payload["config"]
    config = PriorAwareSymbolicCodecConfig(
        **{
            key: value
            for key, value in config_dict.items()
            if key in PriorAwareSymbolicCodecConfig.__dataclass_fields__
        }
    )
    model = SymbolicCodecV5(config).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()
    return model, config


def build_grouped_dataset(
    samples: list[str],
    codec: SymbolicCodecV5,
    codec_config: PriorAwareSymbolicCodecConfig,
    device: torch.device,
    batch_size: int,
) -> V5GroupedPatchDataset:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    byte_counts: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            patch_ids_list = []
            patch_mask_list = []
            byte_count_list = []

            for sample in batch_samples:
                patch_ids, patch_mask = encode_text_to_patches(sample, codec_config.max_seq_len, codec_config.patch_size)
                patch_tensor = torch.tensor(patch_ids, dtype=torch.long)
                patch_ids_list.append(patch_ids)
                patch_mask_list.append(patch_mask)
                byte_count_list.append(patch_tensor.ne(PAD_BYTE_ID).sum(dim=-1).to(torch.float32))

            patch_ids_tensor = torch.tensor(patch_ids_list, dtype=torch.long, device=device)
            patch_mask_tensor = torch.tensor(patch_mask_list, dtype=torch.float32, device=device)
            forward = codec(patch_ids_tensor, patch_mask_tensor)
            symbol_ids = forward.symbol_ids.detach().cpu()
            patch_mask_cpu = patch_mask_tensor.detach().cpu()

            for index in range(len(batch_samples)):
                ids = symbol_ids[index]
                mask = patch_mask_cpu[index]
                counts = byte_count_list[index]
                inputs.append(ids[:-1].to(torch.long))
                targets.append(ids[1:].to(torch.long))
                masks.append(mask[1:].to(torch.float32))
                byte_counts.append(counts[1:])

    return V5GroupedPatchDataset(inputs, targets, masks, byte_counts)


def run_epoch(
    model: GroupedPatchPrior,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gradient_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "accuracy": 0.0, "bpb": 0.0, "batches": 0.0}
    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            byte_counts = batch["byte_counts"].to(device)

            logits_list = model(inputs)
            loss, accuracy, bpb = compute_grouped_metrics(logits_list, targets, mask, byte_counts)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            totals["loss"] += loss.item()
            totals["accuracy"] += accuracy
            totals["bpb"] += bpb
            totals["batches"] += 1

    denom = max(1.0, totals["batches"])
    return {
        "loss": totals["loss"] / denom,
        "accuracy": totals["accuracy"] / denom,
        "bpb": totals["bpb"] / denom,
    }


def save_artifacts(output_dir: Path, config: V5PriorConfig, model: GroupedPatchPrior, best_metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "metrics": best_metrics,
        },
        output_dir / "v5_patch_prior.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = choose_device(config.device)
    output_dir = Path(config.output_dir)

    samples = load_text_samples(config.data_path, dedupe=False)
    train_samples, val_samples = split_samples(samples, config.val_ratio, config.seed)

    codec, codec_config = load_codec(config.codec_run_dir, device)
    if codec_config.patch_size != config.patch_size:
        raise ValueError(f"patch_size mismatch: codec has {codec_config.patch_size}, prior config has {config.patch_size}")

    train_dataset = build_grouped_dataset(train_samples, codec, codec_config, device, config.batch_encode_size)
    val_dataset = build_grouped_dataset(val_samples, codec, codec_config, device, config.batch_encode_size)

    group_bits = int(math.ceil(math.log2(codec_config.sub_codebook_size)))
    model = GroupedPatchPrior(
        [group_bits] * codec_config.num_codebooks,
        config.group_embed_dim,
        config.hidden_size,
        config.num_layers,
        config.dropout,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    epoch_times: list[float] = []

    print(
        f"Training v5 grouped patch prior on {device.type} with "
        f"{len(train_dataset)} train samples, {len(val_dataset)} val samples"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, device, config.gradient_clip_norm)
        val_metrics = run_epoch(model, val_loader, None, device, config.gradient_clip_norm)
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_times.append(epoch_seconds)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bpb={val_metrics['bpb']:.4f} "
            f"epoch_s={epoch_seconds:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "bpb": val_metrics["bpb"],
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "best_epoch": epoch,
                "avg_epoch_seconds": sum(epoch_times) / len(epoch_times),
                "mode": "v5_grouped",
            }
            save_artifacts(output_dir, config, model, best_metrics)

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
