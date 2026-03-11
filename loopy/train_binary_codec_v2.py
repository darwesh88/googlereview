from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset

from loopy.binary_codec_v2 import PAD_BYTE_ID, SemanticBinaryCodec, estimated_patch_bpb, tensor_bit_density
from loopy.dataset import load_text_samples, split_samples
from loopy.v2_config import BinaryCodecConfig


class BytePatchDataset(Dataset):
    def __init__(self, samples: list[str], max_seq_len: int, patch_size: int) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        if self.max_seq_len % self.patch_size != 0:
            raise ValueError("max_seq_len must be divisible by patch_size")
        self.num_patches = self.max_seq_len // self.patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        text = self.samples[index]
        patch_ids, patch_mask = encode_text_to_patches(text, self.max_seq_len, self.patch_size)
        return {
            "text": text,
            "patch_ids": torch.tensor(patch_ids, dtype=torch.long),
            "patch_mask": torch.tensor(patch_mask, dtype=torch.float32),
        }


def encode_text_to_patches(text: str, max_seq_len: int, patch_size: int) -> tuple[list[list[int]], list[float]]:
    byte_values = list(text.encode("utf-8", errors="ignore"))[:max_seq_len]
    padded = byte_values + [PAD_BYTE_ID] * (max_seq_len - len(byte_values))
    patches: list[list[int]] = []
    patch_mask: list[float] = []
    for start in range(0, max_seq_len, patch_size):
        patch = padded[start : start + patch_size]
        patches.append(patch)
        patch_mask.append(1.0 if any(value != PAD_BYTE_ID for value in patch) else 0.0)
    return patches, patch_mask


def decode_patch_ids(patch_ids: torch.Tensor, patch_mask: torch.Tensor | None = None) -> str:
    values = patch_ids.detach().cpu().reshape(-1).tolist()
    if patch_mask is not None:
        expanded_mask = patch_mask.detach().cpu().unsqueeze(-1).repeat(1, patch_ids.size(-1)).reshape(-1).tolist()
        values = [value for value, keep in zip(values, expanded_mask) if keep > 0.0]
    byte_values = [value for value in values if value != PAD_BYTE_ID]
    return bytes(byte_values).decode("utf-8", errors="ignore")


def byte_accuracy(logits: torch.Tensor, patch_ids: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    valid_mask = patch_ids.ne(PAD_BYTE_ID)
    total = valid_mask.sum().item()
    if total <= 0:
        return 0.0
    correct = ((predictions == patch_ids) & valid_mask).sum().item()
    return float(correct / total)


def parse_args(argv: list[str] | None = None) -> BinaryCodecConfig:
    parser = argparse.ArgumentParser(description="Train Loopy v2 semantic binary codec.")
    parser.add_argument("--data-path", default="loopy/example_corpus.txt")
    parser.add_argument("--output-dir", default="loopy/runs/v2_codec")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit-all", action="store_true")
    parser.add_argument("--bit-groups", default="4,8,8,8")
    parser.add_argument("--rate-weight", type=float, default=0.0)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--predictive-weight", type=float, default=0.0)
    args = parser.parse_args(argv)
    bit_groups = tuple(int(part) for part in args.bit_groups.split(",") if part.strip())
    return BinaryCodecConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        overfit_all=args.overfit_all,
        bit_groups=bit_groups,
        rate_weight=args.rate_weight,
        balance_weight=args.balance_weight,
        align_weight=args.align_weight,
        predictive_weight=args.predictive_weight,
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


def create_dataloaders(config: BinaryCodecConfig) -> tuple[DataLoader, DataLoader]:
    samples = load_text_samples(config.data_path, dedupe=False)
    if config.overfit_all:
        train_samples = list(samples)
        val_samples = list(samples)
    else:
        train_samples, val_samples = split_samples(samples, config.val_ratio, config.seed)
    train_dataset = BytePatchDataset(train_samples, config.max_seq_len, config.patch_size)
    val_dataset = BytePatchDataset(val_samples, config.max_seq_len, config.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def run_epoch(
    model: SemanticBinaryCodec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], dict[str, str]]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "rate_loss": 0.0,
        "balance_loss": 0.0,
        "align_loss": 0.0,
        "predictive_loss": 0.0,
        "estimated_bpb": 0.0,
        "byte_accuracy": 0.0,
        "bit_density": 0.0,
        "batches": 0.0,
    }
    preview = {"source": "", "reconstruction": ""}

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            patch_ids = batch["patch_ids"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            forward = model(patch_ids, patch_mask)
            loss = forward.total_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            totals["loss"] += loss.item()
            totals["recon_loss"] += forward.recon_loss.item()
            totals["rate_loss"] += forward.rate_loss.item()
            totals["balance_loss"] += forward.balance_loss.item()
            totals["align_loss"] += forward.align_loss.item()
            totals["predictive_loss"] += forward.predictive_loss.item()
            totals["estimated_bpb"] += estimated_patch_bpb(forward.bit_probs, patch_ids, patch_mask)
            totals["byte_accuracy"] += byte_accuracy(forward.logits, patch_ids)
            totals["bit_density"] += tensor_bit_density(forward.bit_values, patch_mask)
            totals["batches"] += 1

            if not preview["source"]:
                reconstruction = forward.logits.argmax(dim=-1)[0]
                preview = {
                    "source": batch["text"][0],
                    "reconstruction": decode_patch_ids(reconstruction, patch_mask[0]),
                }

    batch_count = max(1.0, totals["batches"])
    metrics = {key: totals[key] / batch_count for key in totals if key != "batches"}
    return metrics, preview


def save_artifacts(
    output_dir: Path,
    config: BinaryCodecConfig,
    model: SemanticBinaryCodec,
    best_metrics: dict[str, float],
    preview: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "metrics": best_metrics,
        },
        output_dir / "v2_codec.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    sample_text = (
        "Source:\n"
        f"{preview['source']}\n\n"
        "Reconstruction:\n"
        f"{preview['reconstruction']}\n"
    )
    (output_dir / "sample_reconstruction.txt").write_text(sample_text, encoding="utf-8")


def main() -> None:
    config = parse_args()
    if config.max_seq_len % config.patch_size != 0:
        raise ValueError("max_seq_len must be divisible by patch_size")

    seed_everything(config.seed)
    device = choose_device(config.device)
    output_dir = Path(config.output_dir)
    train_loader, val_loader = create_dataloaders(config)

    model = SemanticBinaryCodec(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    best_preview = {"source": "", "reconstruction": ""}
    best_epoch = 0
    epoch_times: list[float] = []

    print(
        f"Training v2 codec on {device.type} with {len(train_loader.dataset)} train samples, "
        f"{len(val_loader.dataset)} val samples, patches={config.num_patches}, bits={config.total_bits}, raw_capacity_bpb={config.raw_capacity_bpb:.2f}"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics, _ = run_epoch(model, train_loader, optimizer, device, config.gradient_clip_norm)
        val_metrics, val_preview = run_epoch(model, val_loader, None, device, config.gradient_clip_norm)
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_times.append(epoch_seconds)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"val_align={val_metrics['align_loss']:.4f} "
            f"val_pred={val_metrics['predictive_loss']:.4f} "
            f"val_bpb={val_metrics['estimated_bpb']:.3f} "
            f"byte_acc={val_metrics['byte_accuracy']:.3f} "
            f"bit_density={val_metrics['bit_density']:.3f} "
            f"epoch_s={epoch_seconds:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = {
                "loss": val_metrics["loss"],
                "recon_loss": val_metrics["recon_loss"],
                "rate_loss": val_metrics["rate_loss"],
                "balance_loss": val_metrics["balance_loss"],
                "align_loss": val_metrics["align_loss"],
                "predictive_loss": val_metrics["predictive_loss"],
                "estimated_bpb": val_metrics["estimated_bpb"],
                "byte_accuracy": val_metrics["byte_accuracy"],
                "bit_density": val_metrics["bit_density"],
                "raw_capacity_bpb": config.raw_capacity_bpb,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "best_epoch": best_epoch,
                "avg_epoch_seconds": sum(epoch_times) / len(epoch_times),
            }
            best_preview = val_preview
            save_artifacts(output_dir, config, model, best_metrics, best_preview)

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
