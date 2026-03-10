from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopy.concept_middleware import ConceptLexicon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally decode a Loopy LM run.")
    parser.add_argument("--run-dir", required=True, help="Path to a train_token_lm output directory")
    parser.add_argument("--lexicon", help="Concept lexicon to decode symbolic outputs")
    parser.add_argument("--output", help="Optional path for a decoded text report")
    return parser.parse_args()


def maybe_decode(text: str, lexicon: ConceptLexicon | None) -> str:
    if lexicon is None:
        return text
    return lexicon.decode_text(text)


def extract_sections(sample_text: str) -> tuple[str, str]:
    prompt_marker = "Prompt:\n"
    generated_marker = "\n\nGenerated:\n"
    if prompt_marker not in sample_text or generated_marker not in sample_text:
        return "", sample_text.strip()
    prompt_part, generated_part = sample_text.split(generated_marker, 1)
    prompt = prompt_part.replace(prompt_marker, "", 1).strip()
    generated = generated_part.strip()
    return prompt, generated


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    sample_text = (run_dir / "sample_generation.txt").read_text(encoding="utf-8")
    lexicon = ConceptLexicon.load(args.lexicon) if args.lexicon else None

    prompt, generated = extract_sections(sample_text)
    decoded_prompt = maybe_decode(prompt, lexicon)
    decoded_generated = maybe_decode(generated, lexicon)

    report = {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "prompt": prompt,
        "generated": generated,
        "decoded_prompt": decoded_prompt,
        "decoded_generated": decoded_generated,
    }

    report_text = (
        "Metrics:\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        "Prompt:\n"
        f"{prompt}\n\n"
        "Generated:\n"
        f"{generated}\n\n"
        "Decoded Prompt:\n"
        f"{decoded_prompt}\n\n"
        "Decoded Generated:\n"
        f"{decoded_generated}\n"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
