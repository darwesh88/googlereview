# Ambiguity Baseline

This is the first build step after the research memo.

## Goal

Stop forcing every alias match into a concept code.

This baseline adds a simple context gate:

- ambiguous aliases may only rewrite if nearby words support the intended concept
- otherwise the original word is left alone

## What this tests

It does not solve output wording yet.
It only tests whether safer input rewriting improves the concept path.

## Files

- [concept_policy.py](C:/Users/adarw/Desktop/googlereview/loopy/concept_policy.py)
- [rewrite_corpus_contextual.py](C:/Users/adarw/Desktop/googlereview/loopy/rewrite_corpus_contextual.py)
- [concept_policy.support.json](C:/Users/adarw/Desktop/googlereview/loopy/concept_policy.support.json)

## Commands

```powershell
python -m loopy.rewrite_corpus_contextual --input loopy/noisy_support_corpus.txt --output loopy/runs/noisy_support_concept_gated.txt --lexicon loopy/concepts.support.json --policy loopy/concept_policy.support.json
python -m loopy.train_token_lm --data-path loopy/runs/noisy_support_concept.txt --output-dir loopy/runs/noisy_concept_canonical --epochs 45 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
python -m loopy.train_token_lm --data-path loopy/runs/noisy_support_concept_gated.txt --output-dir loopy/runs/noisy_concept_gated --epochs 45 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
Get-Content loopy\runs\noisy_support_concept_gated.report.json -Raw
Get-Content loopy\runs\noisy_concept_canonical\best_metrics.json -Raw
Get-Content loopy\runs\noisy_concept_gated\best_metrics.json -Raw
```

## What counts as a good sign

- the gated rewrite keeps or improves the concept-model metrics
- the rewrite report shows some ambiguous matches were intentionally skipped
- decoded outputs stay readable later

## What would be a bad sign

- the gate removes too many good rewrites
- metrics get worse than canonical concept rewriting
- the policy becomes too manual to maintain
