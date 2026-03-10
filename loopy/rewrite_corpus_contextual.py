from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from loopy.concept_middleware import ConceptLexicon
from loopy.concept_policy import ContextualConceptEncoder, ContextualRewritePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a corpus with context-gated concept IDs.")
    parser.add_argument("--input", required=True, help="Input .txt file or directory")
    parser.add_argument("--output", required=True, help="Output .txt file or directory")
    parser.add_argument("--lexicon", default="loopy/concepts.support.json")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--preview-lines", type=int, default=3)
    parser.add_argument("--trace-limit", type=int, default=20)
    return parser.parse_args()


def iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.txt"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    lexicon = ConceptLexicon.load(args.lexicon)
    policy = ContextualRewritePolicy.load(args.policy)
    encoder = ContextualConceptEncoder(lexicon, policy)

    files = iter_text_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .txt files found under {input_path}")

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    total_replacements: Counter[str] = Counter()
    total_skips: Counter[str] = Counter()
    trace_rows: list[dict[str, object]] = []
    preview_rows: list[dict[str, str]] = []

    for file_path in files:
        original = file_path.read_text(encoding="utf-8")
        rewritten, replacements, skips, trace = encoder.encode_text(original)
        total_replacements.update(replacements)
        total_skips.update(skips)
        if len(trace_rows) < args.trace_limit:
            remaining = args.trace_limit - len(trace_rows)
            trace_rows.extend(trace[:remaining])

        if input_path.is_dir():
            relative = file_path.relative_to(input_path)
            target = output_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = output_path

        target.write_text(rewritten, encoding="utf-8")

        if len(preview_rows) < args.preview_lines:
            preview_rows.append(
                {
                    "source": original.splitlines()[0] if original.splitlines() else "",
                    "rewritten": rewritten.splitlines()[0] if rewritten.splitlines() else "",
                }
            )

    report = {
        "mode": "contextual_encode",
        "files": len(files),
        "total_replacements": int(sum(total_replacements.values())),
        "concepts_used": len(total_replacements),
        "total_skips": int(sum(total_skips.values())),
        "top_concepts": lexicon.summarize_counts(total_replacements)[:10],
        "top_skip_reasons": [
            {"rule": key, "count": count}
            for key, count in total_skips.most_common(10)
        ],
        "preview": preview_rows,
        "trace": trace_rows,
    }

    report_path = output_path.with_suffix(".report.json") if output_path.is_file() else output_path / "rewrite_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote transformed corpus to {output_path}")
    print(f"Wrote rewrite report to {report_path}")


if __name__ == "__main__":
    main()
