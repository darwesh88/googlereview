from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from loopy.binary_codec_v2 import PAD_BYTE_ID, SemanticBinaryCodec
from loopy.dataset import load_text_samples, split_samples
from loopy.train_binary_codec_v2 import encode_text_to_patches
from loopy.v2_config import BinaryCodecConfig


@dataclass
class PriorConfig:
    mode: str
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
    byte_embed_dim: int
    group_embed_dim: int
    batch_encode_size: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class LearnedPatchDataset(Dataset):
    def __init__(self, inputs: list[torch.Tensor], targets: list[torch.Tensor], masks: list[torch.Tensor], byte_counts: list[torch.Tensor]) -> None:
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


class RawPatchDataset(Dataset):
    def __init__(self, inputs: list[torch.Tensor], targets: list[torch.Tensor], masks: list[torch.Tensor], byte_counts: list[torch.Tensor]) -> None:
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


class GroupedPatchDataset(Dataset):
    def __init__(self, inputs: list[torch.Tensor], targets: list[torch.Tensor], masks: list[torch.Tensor], byte_counts: list[torch.Tensor]) -> None:
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


class LearnedPatchPrior(nn.Module):
    def __init__(self, total_bits: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.input_proj = nn.Linear(total_bits, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, total_bits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.input_proj(inputs))
        hidden = self.dropout(hidden)
        return self.output(hidden)


class RawPatchPrior(nn.Module):
    def __init__(self, patch_size: int, byte_embed_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.patch_size = patch_size
        self.byte_embedding = nn.Embedding(257, byte_embed_dim, padding_idx=PAD_BYTE_ID)
        self.input_proj = nn.Linear(byte_embed_dim * patch_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 257 * patch_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.byte_embedding(inputs).reshape(inputs.size(0), inputs.size(1), -1)
        hidden, _ = self.gru(self.input_proj(embedded))
        hidden = self.dropout(hidden)
        logits = self.output(hidden)
        return logits.reshape(inputs.size(0), inputs.size(1), self.patch_size, 257)


class GroupedPatchPrior(nn.Module):
    def __init__(self, group_sizes: list[int], group_embed_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.group_sizes = group_sizes
        self.group_embeddings = nn.ModuleList(
            [nn.Embedding(1 << group_size, group_embed_dim) for group_size in group_sizes]
        )
        self.input_proj = nn.Linear(group_embed_dim * len(group_sizes), hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1 << group_size) for group_size in group_sizes])

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        embedded_groups = [
            embedding(inputs[..., group_index])
            for group_index, embedding in enumerate(self.group_embeddings)
        ]
        embedded = torch.cat(embedded_groups, dim=-1)
        hidden, _ = self.gru(self.input_proj(embedded))
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.heads]


def parse_args(argv: list[str] | None = None) -> PriorConfig:
    parser = argparse.ArgumentParser(description="Train a patch-level prior over Loopy v2 learned or raw patch streams.")
    parser.add_argument("--mode", choices=["learned", "raw", "grouped"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codec-run-dir", default="")
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
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--byte-embed-dim", type=int, default=16)
    parser.add_argument("--group-embed-dim", type=int, default=16)
    parser.add_argument("--batch-encode-size", type=int, default=32)
    args = parser.parse_args(argv)
    return PriorConfig(
        mode=args.mode,
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
        byte_embed_dim=args.byte_embed_dim,
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


def load_codec(run_dir: str, device: torch.device) -> tuple[SemanticBinaryCodec, BinaryCodecConfig]:
    run_path = Path(run_dir)
    checkpoint_path = run_path / "v2_codec.pt"
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    allowed = {field.name for field in fields(BinaryCodecConfig)}
    config_kwargs = {key: value for key, value in config_dict.items() if key in allowed}
    config = BinaryCodecConfig(**config_kwargs)
    model = SemanticBinaryCodec(config).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()
    return model, config


def raw_patch_tensors(text: str, max_seq_len: int, patch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_ids, patch_mask = encode_text_to_patches(text, max_seq_len, patch_size)
    patch_tensor = torch.tensor(patch_ids, dtype=torch.long)
    patch_mask_tensor = torch.tensor(patch_mask, dtype=torch.float32)
    byte_counts = patch_tensor.ne(PAD_BYTE_ID).sum(dim=-1).to(torch.float32)
    return (
        patch_tensor[:-1],
        patch_tensor[1:],
        patch_mask_tensor[1:],
        byte_counts[1:],
    )


def build_raw_dataset(samples: list[str], max_seq_len: int, patch_size: int) -> RawPatchDataset:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    byte_counts: list[torch.Tensor] = []
    for sample in samples:
        input_tensor, target_tensor, mask_tensor, count_tensor = raw_patch_tensors(sample, max_seq_len, patch_size)
        inputs.append(input_tensor)
        targets.append(target_tensor)
        masks.append(mask_tensor)
        byte_counts.append(count_tensor)
    return RawPatchDataset(inputs, targets, masks, byte_counts)


def build_learned_dataset(
    samples: list[str],
    codec: SemanticBinaryCodec,
    codec_config: BinaryCodecConfig,
    device: torch.device,
    batch_size: int,
) -> LearnedPatchDataset:
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
            bit_values = forward.bit_values.detach().cpu()
            patch_mask_cpu = patch_mask_tensor.detach().cpu()

            for index in range(len(batch_samples)):
                bits = bit_values[index]
                mask = patch_mask_cpu[index]
                counts = byte_count_list[index]
                inputs.append(bits[:-1].to(torch.float32))
                targets.append(bits[1:].to(torch.float32))
                masks.append(mask[1:].to(torch.float32))
                byte_counts.append(counts[1:])

    return LearnedPatchDataset(inputs, targets, masks, byte_counts)


def grouped_symbol_ids(bit_values: torch.Tensor, bit_groups: tuple[int, ...]) -> torch.Tensor:
    groups: list[torch.Tensor] = []
    cursor = 0
    for group_size in bit_groups:
        group_bits = bit_values[..., cursor : cursor + group_size].to(torch.long)
        shifts = torch.arange(group_size - 1, -1, -1, device=group_bits.device, dtype=torch.long)
        group_values = (group_bits * (2 ** shifts)).sum(dim=-1)
        groups.append(group_values)
        cursor += group_size
    return torch.stack(groups, dim=-1)


def build_grouped_dataset(
    samples: list[str],
    codec: SemanticBinaryCodec,
    codec_config: BinaryCodecConfig,
    device: torch.device,
    batch_size: int,
) -> GroupedPatchDataset:
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
            grouped_ids = grouped_symbol_ids(forward.bit_values.detach(), codec_config.bit_groups).cpu()
            patch_mask_cpu = patch_mask_tensor.detach().cpu()

            for index in range(len(batch_samples)):
                ids = grouped_ids[index]
                mask = patch_mask_cpu[index]
                counts = byte_count_list[index]
                inputs.append(ids[:-1].to(torch.long))
                targets.append(ids[1:].to(torch.long))
                masks.append(mask[1:].to(torch.float32))
                byte_counts.append(counts[1:])

    return GroupedPatchDataset(inputs, targets, masks, byte_counts)


def compute_learned_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, byte_counts: torch.Tensor) -> tuple[torch.Tensor, float]:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked_loss = loss * mask.unsqueeze(-1)
    total_bits = mask.sum() * targets.size(-1)
    mean_loss = masked_loss.sum() / total_bits.clamp_min(1.0)
    predicted = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
    correct = ((predicted == targets) * mask.unsqueeze(-1)).sum().item()
    denom = (mask.sum().item() * targets.size(-1))
    bit_acc = float(correct / max(1.0, denom))
    total_nll_bits = masked_loss.sum().item() / math.log(2.0)
    total_bytes = (byte_counts * mask).sum().item()
    bpb = float(total_nll_bits / max(1.0, total_bytes))
    return mean_loss, bit_acc, bpb


def compute_raw_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, byte_counts: torch.Tensor) -> tuple[torch.Tensor, float]:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_BYTE_ID,
        reduction="none",
    ).reshape_as(targets)
    masked_loss = loss * targets.ne(PAD_BYTE_ID).to(loss.dtype) * mask.unsqueeze(-1)
    total_bytes = (byte_counts * mask).sum()
    mean_loss = masked_loss.sum() / total_bytes.clamp_min(1.0)
    predictions = logits.argmax(dim=-1)
    valid = targets.ne(PAD_BYTE_ID) & mask.unsqueeze(-1).bool()
    byte_acc = float(((predictions == targets) & valid).sum().item() / max(1.0, valid.sum().item()))
    total_nll_bits = masked_loss.sum().item() / math.log(2.0)
    bpb = float(total_nll_bits / max(1.0, total_bytes.item()))
    return mean_loss, byte_acc, bpb


def compute_grouped_metrics(
    logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    byte_counts: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
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
    accuracy = float(correct / max(1.0, valid_groups))
    total_nll_bits = total_loss.sum().item() / math.log(2.0)
    total_bytes = (byte_counts * mask).sum().item()
    bpb = float(total_nll_bits / max(1.0, total_bytes))
    return mean_loss, accuracy, bpb


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gradient_clip_norm: float,
    mode: str,
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

            logits = model(inputs)
            if mode == "learned":
                loss, accuracy, bpb = compute_learned_metrics(logits, targets, mask, byte_counts)
            elif mode == "grouped":
                loss, accuracy, bpb = compute_grouped_metrics(logits, targets, mask, byte_counts)
            else:
                loss, accuracy, bpb = compute_raw_metrics(logits, targets, mask, byte_counts)

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


def save_artifacts(output_dir: Path, config: PriorConfig, model: nn.Module, best_metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "metrics": best_metrics,
        },
        output_dir / "patch_prior.pt",
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

    if config.mode in {"learned", "grouped"}:
        if not config.codec_run_dir:
            raise ValueError("--codec-run-dir is required for learned/grouped mode")
        codec, codec_config = load_codec(config.codec_run_dir, device)
        if config.mode == "learned":
            train_dataset = build_learned_dataset(train_samples, codec, codec_config, device, config.batch_encode_size)
            val_dataset = build_learned_dataset(val_samples, codec, codec_config, device, config.batch_encode_size)
            model = LearnedPatchPrior(codec_config.total_bits, config.hidden_size, config.num_layers, config.dropout).to(device)
        else:
            train_dataset = build_grouped_dataset(train_samples, codec, codec_config, device, config.batch_encode_size)
            val_dataset = build_grouped_dataset(val_samples, codec, codec_config, device, config.batch_encode_size)
            model = GroupedPatchPrior(
                list(codec_config.bit_groups),
                config.group_embed_dim,
                config.hidden_size,
                config.num_layers,
                config.dropout,
            ).to(device)
    else:
        train_dataset = build_raw_dataset(train_samples, config.max_seq_len, config.patch_size)
        val_dataset = build_raw_dataset(val_samples, config.max_seq_len, config.patch_size)
        model = RawPatchPrior(config.patch_size, config.byte_embed_dim, config.hidden_size, config.num_layers, config.dropout).to(device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    epoch_times: list[float] = []

    print(
        f"Training patch prior ({config.mode}) on {device.type} with "
        f"{len(train_dataset)} train samples, {len(val_dataset)} val samples"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, device, config.gradient_clip_norm, config.mode)
        val_metrics = run_epoch(model, val_loader, None, device, config.gradient_clip_norm, config.mode)
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
                "mode": config.mode,
            }
            save_artifacts(output_dir, config, model, best_metrics)

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
