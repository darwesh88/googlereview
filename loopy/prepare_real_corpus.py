from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from loopy.dataset import normalize_text

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
ALLOWED_SUFFIXES = {".txt", ".jsonl"}


@dataclass
class PrepareConfig:
    input_path: str
    output_path: str
    text_key: str
    min_chars: int
    max_chars: int
    min_tokens: int
    max_samples: int | None
    seed: int
    dedupe: bool
    redact_emails: bool
    redact_urls: bool
    redact_long_numbers: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_args() -> PrepareConfig:
    parser = argparse.ArgumentParser(description="Prepare a real corpus for Loopy experiments.")
    parser.add_argument("--input", required=True, help="Input .txt file, .jsonl file, or directory")
    parser.add_argument("--output", required=True, help="Output .txt file with one sample per line")
    parser.add_argument("--text-key", default="text", help="Field to read from .jsonl records")
    parser.add_argument("--min-chars", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=280)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate samples after cleaning")
    parser.add_argument("--redact-emails", action="store_true")
    parser.add_argument("--redact-urls", action="store_true")
    parser.add_argument("--redact-long-numbers", action="store_true")
    args = parser.parse_args()
    return PrepareConfig(
        input_path=args.input,
        output_path=args.output,
        text_key=args.text_key,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_tokens=args.min_tokens,
        max_samples=args.max_samples,
        seed=args.seed,
        dedupe=args.dedupe,
        redact_emails=args.redact_emails,
        redact_urls=args.redact_urls,
        redact_long_numbers=args.redact_long_numbers,
    )


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {path.suffix}")
        return [path]
    files = [file_path for file_path in sorted(path.rglob("*")) if file_path.suffix.lower() in ALLOWED_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No supported corpus files found under {path}")
    return files


def extract_units_from_txt(raw_text: str) -> list[str]:
    lines = [normalize_text(line) for line in raw_text.splitlines()]
    units = [line for line in lines if line]
    if units:
        return units

    paragraphs: list[str] = []
    for block in raw_text.split("\n\n"):
        block = normalize_text(block)
        if block:
            paragraphs.append(block)
    return paragraphs


def extract_units_from_jsonl(raw_text: str, text_key: str) -> tuple[list[str], int]:
    units: list[str] = []
    invalid_rows = 0
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        value = payload.get(text_key)
        if isinstance(value, str):
            cleaned = normalize_text(value)
            if cleaned:
                units.append(cleaned)
        else:
            invalid_rows += 1
    return units, invalid_rows


def apply_redactions(text: str, config: PrepareConfig) -> str:
    rewritten = text
    if config.redact_emails:
        rewritten = EMAIL_RE.sub("<email>", rewritten)
    if config.redact_urls:
        rewritten = URL_RE.sub("<url>", rewritten)
    if config.redact_long_numbers:
        rewritten = LONG_NUMBER_RE.sub("<number>", rewritten)
    return rewritten


def validate_sample(text: str, config: PrepareConfig) -> str | None:
    token_count = len(TOKEN_RE.findall(text))
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    if token_count < config.min_tokens:
        return "too_few_tokens"
    if not re.search(r"[A-Za-z]", text):
        return "no_letters"
    return None


def collect_samples(config: PrepareConfig) -> tuple[list[str], dict[str, object]]:
    input_path = Path(config.input_path)
    files = iter_input_files(input_path)

    kept_samples: list[str] = []
    dropped_reasons: Counter[str] = Counter()
    invalid_jsonl_rows = 0
    source_type_counts: Counter[str] = Counter()

    for file_path in files:
        source_type_counts[file_path.suffix.lower()] += 1
        raw_text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".jsonl":
            units, invalid_rows = extract_units_from_jsonl(raw_text, config.text_key)
            invalid_jsonl_rows += invalid_rows
        else:
            units = extract_units_from_txt(raw_text)

        for unit in units:
            cleaned = apply_redactions(normalize_text(unit), config)
            reason = validate_sample(cleaned, config)
            if reason is not None:
                dropped_reasons[reason] += 1
                continue
            kept_samples.append(cleaned)

    before_dedupe = len(kept_samples)
    if config.dedupe:
        kept_samples = list(dict.fromkeys(kept_samples))

    random.Random(config.seed).shuffle(kept_samples)
    if config.max_samples is not None:
        kept_samples = kept_samples[: config.max_samples]

    if len(kept_samples) < 2:
        raise ValueError("Prepared corpus needs at least 2 usable samples")

    report = {
        "config": config.to_dict(),
        "files": len(files),
        "source_types": dict(source_type_counts),
        "invalid_jsonl_rows": invalid_jsonl_rows,
        "kept_before_dedupe": before_dedupe,
        "kept_after_dedupe": len(kept_samples) if config.dedupe else before_dedupe,
        "written_samples": len(kept_samples),
        "dropped_reasons": dict(dropped_reasons),
        "avg_chars": round(sum(len(sample) for sample in kept_samples) / len(kept_samples), 2),
        "preview": kept_samples[:5],
    }
    return kept_samples, report


def main() -> None:
    config = parse_args()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, report = collect_samples(config)
    output_path.write_text("\n".join(samples) + "\n", encoding="utf-8")

    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote prepared corpus to {output_path}")
    print(f"Wrote preparation report to {report_path}")


if __name__ == "__main__":
    main()
