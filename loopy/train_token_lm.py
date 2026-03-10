from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from loopy.dataset import load_text_samples, split_samples

TOKEN_RE = re.compile(r"<[A-Za-z0-9_:-]+>|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]")
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{"}


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_seq_len: int
    embed_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    gradient_clip_norm: float
    val_ratio: float
    seed: int
    device: str
    overfit_all: bool
    max_new_tokens: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Tokenizer:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}
        self.pad_id = token_to_id[PAD_TOKEN]
        self.bos_id = token_to_id[BOS_TOKEN]
        self.eos_id = token_to_id[EOS_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]

    @classmethod
    def build(cls, samples: list[str]) -> "Tokenizer":
        token_to_id = {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        for sample in samples:
            for token in tokenize(sample):
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_seq_len: int) -> tuple[list[int], list[int], list[int]]:
        tokens = tokenize(text)
        tokens = [BOS_TOKEN, *tokens[: max_seq_len - 2], EOS_TOKEN]
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        input_ids = ids[:-1]
        target_ids = ids[1:]
        attention_mask = [1] * len(input_ids)

        pad_length = max_seq_len - 1 - len(input_ids)
        if pad_length > 0:
            input_ids.extend([self.pad_id] * pad_length)
            target_ids.extend([self.pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)

        return input_ids, target_ids, attention_mask

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)
            if token in {PAD_TOKEN, BOS_TOKEN}:
                continue
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        return detokenize(tokens)


class SequenceDataset(Dataset):
    def __init__(self, samples: list[str], tokenizer: Tokenizer, max_seq_len: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        text = self.samples[index]
        input_ids, target_ids, attention_mask = self.tokenizer.encode(text, self.max_seq_len)
        return {
            "text": text,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TokenGRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        hidden, _ = self.gru(embedded)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def generate(self, prompt_ids: list[int], max_new_tokens: int, eos_id: int, device: torch.device) -> list[int]:
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            inputs = torch.tensor([generated], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.forward(inputs)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if next_id == eos_id:
                break
        return generated


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    pieces: list[str] = []
    for token in tokens:
        if not pieces:
            pieces.append(token)
            continue
        if token in PUNCT_NO_SPACE_BEFORE:
            pieces[-1] = pieces[-1] + token
        elif pieces[-1] in PUNCT_NO_SPACE_AFTER:
            pieces[-1] = pieces[-1] + token
        else:
            pieces.append(" " + token)
    return "".join(pieces)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a tiny token LM for corpus comparison.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit-all", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args(argv)
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        overfit_all=args.overfit_all,
        max_new_tokens=args.max_new_tokens,
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


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, Tokenizer]:
    # Preserve duplicates for LM experiments so corpus rewrites do not silently change dataset size.
    samples = load_text_samples(config.data_path, dedupe=False)
    if config.overfit_all:
        train_samples = list(samples)
        val_samples = list(samples)
    else:
        train_samples, val_samples = split_samples(samples, config.val_ratio, config.seed)
    tokenizer = Tokenizer.build(train_samples)
    train_dataset = SequenceDataset(train_samples, tokenizer, config.max_seq_len)
    val_dataset = SequenceDataset(val_samples, tokenizer, config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer


def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=pad_id)


def run_epoch(
    model: TokenGRULM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pad_id: int,
) -> tuple[dict[str, float], dict[str, str]]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "tokens": 0.0, "batches": 0.0}
    preview = {"source": "", "prediction": ""}

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids)
            loss = compute_loss(logits, target_ids, pad_id)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            totals["loss"] += loss.item()
            totals["tokens"] += attention_mask.sum().item()
            totals["batches"] += 1

            if not preview["source"]:
                prediction_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
                preview = {
                    "source": batch["text"][0],
                    "prediction": prediction_ids,
                }

    metrics = {
        "loss": totals["loss"] / max(1.0, totals["batches"]),
        "token_count": totals["tokens"],
    }
    return metrics, preview


def save_artifacts(
    output_dir: Path,
    config: TrainConfig,
    model: TokenGRULM,
    tokenizer: Tokenizer,
    best_metrics: dict[str, float],
    sample_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "token_to_id": tokenizer.token_to_id,
            "metrics": best_metrics,
        },
        output_dir / "token_lm.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(tokenizer.token_to_id, indent=2), encoding="utf-8")
    (output_dir / "sample_generation.txt").write_text(sample_text, encoding="utf-8")


def build_generation_sample(
    model: TokenGRULM,
    tokenizer: Tokenizer,
    sample_text: str,
    config: TrainConfig,
    device: torch.device,
) -> str:
    prompt_tokens = tokenize(sample_text)[:4]
    prompt_ids = [tokenizer.bos_id, *[tokenizer.token_to_id.get(token, tokenizer.unk_id) for token in prompt_tokens]]
    generated_ids = model.generate(prompt_ids, config.max_new_tokens, tokenizer.eos_id, device)
    prompt_text = detokenize(prompt_tokens)
    generated_text = tokenizer.decode(generated_ids)
    return f"Prompt:\n{prompt_text}\n\nGenerated:\n{generated_text}\n"


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = choose_device(config.device)
    output_dir = Path(config.output_dir)

    train_loader, val_loader, tokenizer = create_dataloaders(config)
    model = TokenGRULM(
        vocab_size=len(tokenizer.token_to_id),
        embed_dim=config.embed_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    best_sample = ""
    first_sample = train_loader.dataset.samples[0]

    print(
        f"Training on {device.type} with {len(train_loader.dataset)} train samples, "
        f"{len(val_loader.dataset)} val samples, vocab={len(tokenizer.token_to_id)}"
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, device, tokenizer.pad_id)
        val_metrics, _ = run_epoch(model, val_loader, None, device, tokenizer.pad_id)
        val_ppl = torch.exp(torch.tensor(val_metrics["loss"])).item()

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ppl={val_ppl:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {
                "loss": val_metrics["loss"],
                "perplexity": val_ppl,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "vocab_size": len(tokenizer.token_to_id),
            }
            best_sample = build_generation_sample(model, tokenizer, first_sample, config, device)
            save_artifacts(output_dir, config, model, tokenizer, best_metrics, best_sample)

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()

