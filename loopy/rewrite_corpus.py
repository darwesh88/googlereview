from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from loopy.concept_middleware import ConceptLexicon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a corpus with concept IDs.")
    parser.add_argument("--input", required=True, help="Input .txt file or directory")
    parser.add_argument("--output", required=True, help="Output .txt file or directory")
    parser.add_argument("--lexicon", default="loopy/concepts.sample.json")
    parser.add_argument("--decode", action="store_true", help="Decode concept IDs back to canonical text")
    parser.add_argument("--preview-lines", type=int, default=3)
    return parser.parse_args()


def iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.txt"))


def transform_text(lexicon: ConceptLexicon, text: str, decode: bool) -> tuple[str, Counter[str]]:
    if decode:
        return lexicon.decode_text(text), Counter()
    return lexicon.encode_text(text)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    lexicon = ConceptLexicon.load(args.lexicon)

    files = iter_text_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .txt files found under {input_path}")

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    total_counts: Counter[str] = Counter()
    preview_rows: list[dict[str, str]] = []

    for file_path in files:
        original = file_path.read_text(encoding="utf-8")
        transformed, counts = transform_text(lexicon, original, args.decode)
        total_counts.update(counts)

        if input_path.is_dir():
            relative = file_path.relative_to(input_path)
            target = output_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = output_path

        target.write_text(transformed, encoding="utf-8")

        if len(preview_rows) < args.preview_lines:
            preview_rows.append(
                {
                    "source": original.splitlines()[0] if original.splitlines() else "",
                    "rewritten": transformed.splitlines()[0] if transformed.splitlines() else "",
                }
            )

    report = {
        "mode": "decode" if args.decode else "encode",
        "files": len(files),
        "total_replacements": int(sum(total_counts.values())),
        "concepts_used": len(total_counts),
        "top_concepts": lexicon.summarize_counts(total_counts)[:10],
        "preview": preview_rows,
    }

    report_path = output_path.with_suffix(".report.json") if output_path.is_file() else output_path / "rewrite_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote transformed corpus to {output_path}")
    print(f"Wrote rewrite report to {report_path}")


if __name__ == "__main__":
    main()
