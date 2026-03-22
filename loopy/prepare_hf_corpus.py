from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from loopy.dataset import normalize_text

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

DATASET_SPECS: dict[str, dict[str, str]] = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "config": "",
        "split": "train",
        "text_field": "text",
        "label": "TinyStories train split",
    },
    "wikitext103": {
        "path": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "label": "WikiText-103 raw train split",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean Hugging Face corpus for Loopy experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS.keys()), required=True)
    parser.add_argument("--output", required=True, help="Output .txt file with one sample per line")
    parser.add_argument("--split", help="Override the default dataset split")
    parser.add_argument("--text-field", help="Override the default text field")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=220)
    parser.add_argument("--min-tokens", type=int, default=6)
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate cleaned samples before sampling")
    parser.add_argument("--redact-emails", action="store_true")
    parser.add_argument("--redact-urls", action="store_true")
    parser.add_argument("--redact-long-numbers", action="store_true")
    return parser.parse_args()


def load_hf_dataset(dataset_key: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for prepare_hf_corpus.py. "
            "Install loopy/requirements.txt first."
        ) from exc

    spec = DATASET_SPECS[dataset_key]
    config_name = spec["config"] or None
    return load_dataset(spec["path"], config_name, split=split)


def apply_redactions(text: str, args: argparse.Namespace) -> str:
    rewritten = text
    if args.redact_emails:
        rewritten = EMAIL_RE.sub("<email>", rewritten)
    if args.redact_urls:
        rewritten = URL_RE.sub("<url>", rewritten)
    if args.redact_long_numbers:
        rewritten = LONG_NUMBER_RE.sub("<number>", rewritten)
    return rewritten


def validate_sample(text: str, args: argparse.Namespace) -> str | None:
    token_count = len(TOKEN_RE.findall(text))
    if len(text) < args.min_chars:
        return "too_short"
    if len(text) > args.max_chars:
        return "too_long"
    if token_count < args.min_tokens:
        return "too_few_tokens"
    if not re.search(r"[A-Za-z]", text):
        return "no_letters"
    return None


def extract_text(record: dict[str, Any], text_field: str) -> str:
    value = record.get(text_field)
    if not isinstance(value, str):
        return ""
    return normalize_text(value)


def prepare_samples(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    spec = DATASET_SPECS[args.dataset]
    split = args.split or spec["split"]
    text_field = args.text_field or spec["text_field"]
    dataset = load_hf_dataset(args.dataset, split)

    cleaned_samples: list[str] = []
    dropped_reasons: Counter[str] = Counter()

    for row in dataset:
        text = extract_text(row, text_field)
        if not text:
            dropped_reasons["empty_or_missing_text"] += 1
            continue
        text = apply_redactions(text, args)
        reason = validate_sample(text, args)
        if reason is not None:
            dropped_reasons[reason] += 1
            continue
        cleaned_samples.append(text)

    before_dedupe = len(cleaned_samples)
    if args.dedupe:
        cleaned_samples = list(dict.fromkeys(cleaned_samples))

    random.Random(args.seed).shuffle(cleaned_samples)
    if args.max_samples is not None:
        cleaned_samples = cleaned_samples[: args.max_samples]

    if len(cleaned_samples) < 2:
        raise ValueError("Prepared corpus needs at least 2 usable samples")

    report = {
        "dataset": args.dataset,
        "dataset_label": spec["label"],
        "dataset_path": spec["path"],
        "dataset_config": spec["config"] or None,
        "split": split,
        "text_field": text_field,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "dedupe": args.dedupe,
        "kept_before_dedupe": before_dedupe,
        "written_samples": len(cleaned_samples),
        "dropped_reasons": dict(dropped_reasons),
        "avg_chars": round(sum(len(sample) for sample in cleaned_samples) / len(cleaned_samples), 2),
        "preview": cleaned_samples[:5],
    }
    return cleaned_samples, report


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, report = prepare_samples(args)
    output_path.write_text("\n".join(samples) + "\n", encoding="utf-8")

    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote prepared corpus to {output_path}")
    print(f"Wrote preparation report to {report_path}")


if __name__ == "__main__":
    main()
