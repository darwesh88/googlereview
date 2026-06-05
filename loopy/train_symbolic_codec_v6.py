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

from loopy.dynamic_patching import decode_dynamic_patch_ids, encode_text_to_dynamic_patches
from loopy.dataset import load_text_samples, split_samples
from loopy.symbolic_codec_v6 import SymbolicCodecV6
from loopy.v6_config import DynamicSymbolicCodecConfig


class DynamicBytePatchDataset(Dataset):
    def __init__(
        self,
        samples: list[str],
        max_seq_len: int,
        max_patches: int,
        max_patch_size: int,
        min_patch_size: int,
        patching_mode: str,
    ) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.max_patches = max_patches
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.patching_mode = patching_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        text = self.samples[index]
        encoding = encode_text_to_dynamic_patches(
            text,
            self.max_seq_len,
            self.max_patches,
            self.max_patch_size,
            self.min_patch_size,
            self.patching_mode,
        )
        return {
            "text": text,
            "patch_ids": torch.tensor(encoding.patch_ids, dtype=torch.long),
            "patch_mask": torch.tensor(encoding.patch_mask, dtype=torch.float32),
            "byte_mask": torch.tensor(encoding.byte_mask, dtype=torch.float32),
        }


def parse_args(argv: list[str] | None = None) -> DynamicSymbolicCodecConfig:
    parser = argparse.ArgumentParser(description="Train Loopy v6 dynamic-patch symbolic codec.")
    parser.add_argument("--data-path", default="loopy/example_corpus.txt")
    parser.add_argument("--output-dir", default="loopy/runs/v6_codec")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-patches", type=int, default=64)
    parser.add_argument("--max-patch-size", type=int, default=8)
    parser.add_argument("--min-patch-size", type=int, default=1)
    parser.add_argument("--patching-mode", choices=["fixed", "space", "boundary"], default="boundary")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--pre-context-layers", type=int, default=1)
    parser.add_argument("--post-context-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit-all", action="store_true")
    parser.add_argument("--num-codebooks", type=int, default=4)
    parser.add_argument("--sub-codebook-size", type=int, default=64)
    parser.add_argument("--assignment-temp", type=float, default=0.5)
    parser.add_argument("--commitment-weight", type=float, default=0.05)
    parser.add_argument("--codebook-weight", type=float, default=0.25)
    parser.add_argument("--usage-weight", type=float, default=0.05)
    parser.add_argument("--use-residual-detail", action="store_true")
    parser.add_argument("--residual-usage-weight", type=float, default=0.005)
    parser.add_argument("--residual-gate-bias", type=float, default=-2.0)
    args = parser.parse_args(argv)
    return DynamicSymbolicCodecConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        max_patches=args.max_patches,
        max_patch_size=args.max_patch_size,
        min_patch_size=args.min_patch_size,
        patching_mode=args.patching_mode,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        pre_context_layers=args.pre_context_layers,
        post_context_layers=args.post_context_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        overfit_all=args.overfit_all,
        num_codebooks=args.num_codebooks,
        sub_codebook_size=args.sub_codebook_size,
        assignment_temp=args.assignment_temp,
        commitment_weight=args.commitment_weight,
        codebook_weight=args.codebook_weight,
        usage_weight=args.usage_weight,
        use_residual_detail=args.use_residual_detail,
        residual_usage_weight=args.residual_usage_weight,
        residual_gate_bias=args.residual_gate_bias,
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


def byte_accuracy(logits: torch.Tensor, patch_ids: torch.Tensor, byte_mask: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    valid = byte_mask.bool()
    total = valid.sum().item()
    if total <= 0:
        return 0.0
    correct = ((predictions == patch_ids) & valid).sum().item()
    return float(correct / total)


def empirical_capacity_bpb(config: DynamicSymbolicCodecConfig, patch_mask: torch.Tensor, byte_mask: torch.Tensor) -> float:
    active_patches = patch_mask.sum().item()
    active_bytes = byte_mask.sum().item()
    if active_bytes <= 0:
        return 0.0
    return float((active_patches * config.code_bits_per_patch) / active_bytes)


def average_patch_bytes(patch_mask: torch.Tensor, byte_mask: torch.Tensor) -> float:
    active_patches = patch_mask.sum().item()
    if active_patches <= 0:
        return 0.0
    return float(byte_mask.sum().item() / active_patches)


def create_dataloaders(config: DynamicSymbolicCodecConfig) -> tuple[DataLoader, DataLoader]:
    samples = load_text_samples(config.data_path, dedupe=False)
    if config.overfit_all:
        train_samples = list(samples)
        val_samples = list(samples)
    else:
        train_samples, val_samples = split_samples(samples, config.val_ratio, config.seed)
    train_dataset = DynamicBytePatchDataset(
        train_samples,
        config.max_seq_len,
        config.max_patches,
        config.max_patch_size,
        config.min_patch_size,
        config.patching_mode,
    )
    val_dataset = DynamicBytePatchDataset(
        val_samples,
        config.max_seq_len,
        config.max_patches,
        config.max_patch_size,
        config.min_patch_size,
        config.patching_mode,
    )
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
    )


def run_epoch(
    model: SymbolicCodecV6,
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
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "usage_loss": 0.0,
        "residual_usage_loss": 0.0,
        "byte_accuracy": 0.0,
        "codebook_perplexity": 0.0,
        "avg_patch_bytes": 0.0,
        "empirical_capacity_bpb": 0.0,
        "batches": 0.0,
    }
    preview = {"source": "", "reconstruction": ""}

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            patch_ids = batch["patch_ids"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            byte_mask = batch["byte_mask"].to(device)
            forward = model(patch_ids, patch_mask, byte_mask)
            loss = forward.total_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            totals["loss"] += loss.item()
            totals["recon_loss"] += forward.recon_loss.item()
            totals["commitment_loss"] += forward.commitment_loss.item()
            totals["codebook_loss"] += forward.codebook_loss.item()
            totals["usage_loss"] += forward.usage_loss.item()
            totals["residual_usage_loss"] += forward.residual_usage_loss.item()
            totals["byte_accuracy"] += byte_accuracy(forward.logits, patch_ids, byte_mask)
            totals["codebook_perplexity"] += forward.codebook_perplexity.item()
            totals["avg_patch_bytes"] += average_patch_bytes(patch_mask, byte_mask)
            totals["empirical_capacity_bpb"] += empirical_capacity_bpb(model.config, patch_mask, byte_mask)
            totals["batches"] += 1

            if not preview["source"]:
                reconstruction = model.reconstruct(patch_ids, patch_mask, byte_mask)[0]
                preview = {
                    "source": batch["text"][0],
                    "reconstruction": decode_dynamic_patch_ids(reconstruction, byte_mask[0]),
                }

    denom = max(1.0, totals["batches"])
    metrics = {key: totals[key] / denom for key in totals if key != "batches"}
    return metrics, preview


def save_artifacts(
    output_dir: Path,
    config: DynamicSymbolicCodecConfig,
    model: SymbolicCodecV6,
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
        output_dir / "v6_codec.pt",
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
    seed_everything(config.seed)
    device = choose_device(config.device)
    output_dir = Path(config.output_dir)
    train_loader, val_loader = create_dataloaders(config)

    model = SymbolicCodecV6(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    best_preview = {"source": "", "reconstruction": ""}
    epoch_times: list[float] = []

    print(
        f"Training v6 dynamic symbolic codec on {device.type} with {len(train_loader.dataset)} train samples, "
        f"{len(val_loader.dataset)} val samples, max_patches={config.max_patches}, "
        f"max_patch_size={config.max_patch_size}, mode={config.patching_mode}, "
        f"code_bits_per_patch={config.code_bits_per_patch}"
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
            f"val_codebook={val_metrics['codebook_loss']:.4f} "
            f"val_usage={val_metrics['usage_loss']:.4f} "
            f"val_commit={val_metrics['commitment_loss']:.4f} "
            f"val_residual={val_metrics['residual_usage_loss']:.4f} "
            f"byte_acc={val_metrics['byte_accuracy']:.3f} "
            f"codebook_ppl={val_metrics['codebook_perplexity']:.2f} "
            f"avg_patch_bytes={val_metrics['avg_patch_bytes']:.2f} "
            f"capacity_bpb={val_metrics['empirical_capacity_bpb']:.2f} "
            f"epoch_s={epoch_seconds:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {
                "loss": val_metrics["loss"],
                "recon_loss": val_metrics["recon_loss"],
                "commitment_loss": val_metrics["commitment_loss"],
                "codebook_loss": val_metrics["codebook_loss"],
                "usage_loss": val_metrics["usage_loss"],
                "residual_usage_loss": val_metrics["residual_usage_loss"],
                "byte_accuracy": val_metrics["byte_accuracy"],
                "codebook_perplexity": val_metrics["codebook_perplexity"],
                "avg_patch_bytes": val_metrics["avg_patch_bytes"],
                "empirical_capacity_bpb": val_metrics["empirical_capacity_bpb"],
                "code_bits_per_patch": config.code_bits_per_patch,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "best_epoch": epoch,
                "avg_epoch_seconds": sum(epoch_times) / len(epoch_times),
            }
            best_preview = val_preview
            save_artifacts(output_dir, config, model, best_metrics, best_preview)

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
