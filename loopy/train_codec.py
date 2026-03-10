from __future__ import annotations

import json
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from loopy.codec_model import LoopyCodec
from loopy.config import TrainConfig, parse_args
from loopy.dataset import PAD_ID, TextSpanDataset, decode_ids, estimate_symbol_count, load_text_samples, split_samples


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


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    samples = load_text_samples(config.data_path)
    if config.overfit_all:
        train_samples = list(samples)
        val_samples = list(samples)
    else:
        train_samples, val_samples = split_samples(samples, config.val_ratio, config.seed)

    train_dataset = TextSpanDataset(train_samples, config.max_seq_len)
    val_dataset = TextSpanDataset(val_samples, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(config: TrainConfig, device: torch.device) -> LoopyCodec:
    model = LoopyCodec(
        max_seq_len=config.max_seq_len,
        chunk_size=config.chunk_size,
        d_model=config.d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        codebook_size=config.codebook_size,
        commitment_cost=config.commitment_cost,
    )
    return model.to(device)


def reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
    )


def decode_predictions(logits: torch.Tensor) -> list[str]:
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    return [decode_ids(sequence) for sequence in predictions]


def run_epoch(
    model: LoopyCodec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: TrainConfig,
) -> tuple[dict[str, float], dict[str, str]]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "perplexity": 0.0,
        "tokens": 0.0,
        "symbols": 0.0,
        "batches": 0.0,
    }
    sample_preview = {"source": "", "reconstruction": ""}

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids, attention_mask)
            recon = reconstruction_loss(output.logits, input_ids)
            loss = recon + output.vq_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()

            symbol_count = estimate_symbol_count(attention_mask, config.chunk_size).float()
            token_count = attention_mask.sum(dim=1).float()

            totals["loss"] += loss.item()
            totals["recon_loss"] += recon.item()
            totals["vq_loss"] += output.vq_loss.item()
            totals["perplexity"] += output.perplexity.item()
            totals["tokens"] += token_count.sum().item()
            totals["symbols"] += symbol_count.sum().item()
            totals["batches"] += 1

            if not sample_preview["source"]:
                reconstructions = decode_predictions(output.logits)
                sample_preview = {
                    "source": batch["text"][0],
                    "reconstruction": reconstructions[0],
                }

    num_batches = max(1.0, totals["batches"])
    metrics = {
        "loss": totals["loss"] / num_batches,
        "recon_loss": totals["recon_loss"] / num_batches,
        "vq_loss": totals["vq_loss"] / num_batches,
        "codebook_perplexity": totals["perplexity"] / num_batches,
        "avg_tokens_per_span": totals["tokens"] / max(1.0, len(loader.dataset)),
        "avg_symbols_per_span": totals["symbols"] / max(1.0, len(loader.dataset)),
        "compression_ratio": totals["tokens"] / max(1.0, totals["symbols"]),
    }
    return metrics, sample_preview


def save_artifacts(
    output_dir: Path,
    config: TrainConfig,
    model: LoopyCodec,
    best_metrics: dict[str, float],
    best_preview: dict[str, str],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "metrics": best_metrics,
        },
        output_dir / "codec.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    preview_text = (
        "Source:\n"
        f"{best_preview['source']}\n\n"
        "Reconstruction:\n"
        f"{best_preview['reconstruction']}\n"
    )
    (output_dir / "sample_reconstruction.txt").write_text(preview_text, encoding="utf-8")


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    output_dir = ensure_output_dir(config.output_dir)
    device = choose_device(config.device)

    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    best_preview = {"source": "", "reconstruction": ""}

    print(f"Training on {device.type} with {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples")

    for epoch in range(1, config.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, device, config)
        val_metrics, val_preview = run_epoch(model, val_loader, None, device, config)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"compression={val_metrics['compression_ratio']:.2f}x "
            f"codebook_ppl={val_metrics['codebook_perplexity']:.1f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            best_preview = val_preview
            save_artifacts(output_dir, config, model, best_metrics, best_preview)

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    print("Saved checkpoint and sample reconstruction to", output_dir)


if __name__ == "__main__":
    main()
