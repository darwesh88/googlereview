from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

PAD_ID = 0
BYTE_OFFSET = 1
EOS_ID = 257
VOCAB_SIZE = 258


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def encode_text(text: str, max_seq_len: int) -> tuple[list[int], list[int]]:
    clean_text = normalize_text(text)
    byte_values = list(clean_text.encode("utf-8"))[: max_seq_len - 1]
    token_ids = [value + BYTE_OFFSET for value in byte_values]
    token_ids.append(EOS_ID)

    attention_mask = [1] * len(token_ids)
    padding = max_seq_len - len(token_ids)
    if padding > 0:
        token_ids.extend([PAD_ID] * padding)
        attention_mask.extend([0] * padding)

    return token_ids, attention_mask


def decode_ids(token_ids: Sequence[int]) -> str:
    byte_values: list[int] = []
    for token_id in token_ids:
        if token_id == EOS_ID:
            break
        if token_id <= PAD_ID:
            continue
        if 1 <= token_id <= 256:
            byte_values.append(token_id - BYTE_OFFSET)
    return bytes(byte_values).decode("utf-8", errors="ignore")


def _extract_units(raw_text: str) -> list[str]:
    units: list[str] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if "\n" in block:
            for line in block.splitlines():
                line = normalize_text(line)
                if len(line) >= 8:
                    units.append(line)
        else:
            block = normalize_text(block)
            if len(block) >= 8:
                units.append(block)
    return units


def load_text_samples(path: str | Path, dedupe: bool = True) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    samples: list[str] = []
    files = [path] if path.is_file() else sorted(path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {path}")

    for file_path in files:
        raw_text = file_path.read_text(encoding="utf-8")
        samples.extend(_extract_units(raw_text))

    cleaned = [sample for sample in samples if sample]
    if dedupe:
        cleaned = list(dict.fromkeys(cleaned))
    if len(cleaned) < 2:
        raise ValueError("Need at least 2 text samples to create train and validation splits")
    return cleaned


def split_samples(samples: Sequence[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        val_size = 1
    train_samples = shuffled[:-val_size]
    val_samples = shuffled[-val_size:]
    return train_samples, val_samples


class TextSpanDataset(Dataset):
    def __init__(self, samples: Sequence[str], max_seq_len: int) -> None:
        self.samples = list(samples)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        text = self.samples[index]
        input_ids, attention_mask = encode_text(text, self.max_seq_len)
        return {
            "text": text,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def estimate_symbol_count(attention_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    token_counts = attention_mask.sum(dim=1)
    return ((token_counts + chunk_size - 1) // chunk_size).clamp_min(1)

