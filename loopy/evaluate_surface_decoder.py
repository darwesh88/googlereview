from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from loopy.concept_middleware import ConceptLexicon
from loopy.dataset import load_text_samples, split_samples
from loopy.surface_decoder import ContextualAliasDecoder, extract_alias_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate contextual alias decoding against canonical decode.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--lexicon", default="loopy/concepts.support.json")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--min-alias-count", type=int, default=2)
    parser.add_argument("--base-switch-margin", type=float, default=0.6)
    parser.add_argument("--dominance-penalty", type=float, default=0.8)
    parser.add_argument("--copy-bonus", type=float, default=3.0)
    parser.add_argument("--output", default="loopy/runs/surface_decoder_eval.json")
    parser.add_argument("--model-output", default="loopy/runs/surface_decoder_model.json")
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args()


def evaluate_examples(
    examples,
    lexicon: ConceptLexicon,
    decoder: ContextualAliasDecoder,
    sample_limit: int,
) -> dict[str, object]:
    total = 0
    canonical_correct = 0
    contextual_correct = 0
    copy_used = 0
    fallback_to_canonical = 0
    by_concept = defaultdict(lambda: {"total": 0, "canonical_correct": 0, "contextual_correct": 0})
    samples = []

    for example in examples:
        total += 1
        canonical_alias = lexicon.id_to_entry[example.concept_id].canonical.lower()
        contextual = decoder.predict_alias(
            example.concept_id,
            example.context_tokens,
            memory_text=example.prefix_text,
        )
        canonical_hit = canonical_alias == example.true_alias
        contextual_hit = contextual.predicted_alias == example.true_alias

        if canonical_hit:
            canonical_correct += 1
        if contextual_hit:
            contextual_correct += 1
        if contextual.used_copy:
            copy_used += 1
        if contextual.predicted_alias == contextual.canonical_alias:
            fallback_to_canonical += 1

        row = by_concept[example.concept_id]
        row["total"] += 1
        row["canonical_correct"] += int(canonical_hit)
        row["contextual_correct"] += int(contextual_hit)

        if len(samples) < sample_limit and canonical_hit != contextual_hit:
            samples.append(
                {
                    "concept_id": example.concept_id,
                    "true_alias": example.true_alias,
                    "canonical_alias": canonical_alias,
                    "contextual_alias": contextual.predicted_alias,
                    "context_tokens": list(example.context_tokens),
                    "prefix_text": example.prefix_text[-80:],
                    "contextual_score": contextual.score,
                    "canonical_score": contextual.canonical_score,
                    "margin_over_canonical": contextual.margin_over_canonical,
                    "used_copy": contextual.used_copy,
                }
            )

    by_concept_report = []
    for concept_id, row in sorted(by_concept.items()):
        total_count = row["total"]
        by_concept_report.append(
            {
                "concept_id": concept_id,
                "canonical": lexicon.id_to_entry[concept_id].canonical,
                "total": total_count,
                "canonical_accuracy": row["canonical_correct"] / total_count if total_count else 0.0,
                "contextual_accuracy": row["contextual_correct"] / total_count if total_count else 0.0,
            }
        )

    return {
        "examples": total,
        "canonical_accuracy": canonical_correct / total if total else 0.0,
        "contextual_accuracy": contextual_correct / total if total else 0.0,
        "copy_used": copy_used,
        "fallback_to_canonical": fallback_to_canonical,
        "samples": samples,
        "by_concept": by_concept_report,
    }


def main() -> None:
    args = parse_args()
    lexicon = ConceptLexicon.load(args.lexicon)
    samples = load_text_samples(args.data_path, dedupe=False)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    decoder = ContextualAliasDecoder(
        lexicon,
        window=args.window,
        min_alias_count=args.min_alias_count,
        base_switch_margin=args.base_switch_margin,
        dominance_penalty=args.dominance_penalty,
        copy_bonus=args.copy_bonus,
    )
    decoder.fit(train_samples)
    val_examples = []
    for text in val_samples:
        val_examples.extend(extract_alias_examples(text, lexicon, args.window))

    metrics = evaluate_examples(val_examples, lexicon, decoder, args.sample_limit)
    report = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "window": args.window,
        "min_alias_count": args.min_alias_count,
        "base_switch_margin": args.base_switch_margin,
        "dominance_penalty": args.dominance_penalty,
        "copy_bonus": args.copy_bonus,
        **metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    decoder.save(args.model_output)

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_path}")
    print(f"Saved decoder model to {args.model_output}")


if __name__ == "__main__":
    main()
