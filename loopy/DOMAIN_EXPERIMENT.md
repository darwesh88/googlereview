# Domain Experiment

This is the first fair test for concept middleware.

## Why this dataset

The toy corpus was too small and too generic.
It barely used the concept layer.

This domain corpus is different:

- repeated entities
- lots of synonym variation
- multi-word aliases
- narrow support-style language

That is where reversible concept IDs have a real chance to help.

## Run order

1. Generate the domain corpus.
2. Rewrite it with concept IDs.
3. Train the same token LM on both versions.
4. Compare validation loss and perplexity.

## Commands

```powershell
python -m loopy.make_domain_corpus --output loopy/domain_support_corpus.txt --samples 320 --seed 7
python -m loopy.rewrite_corpus --input loopy/domain_support_corpus.txt --output loopy/runs/domain_support_concept.txt --lexicon loopy/concepts.support.json
python -m loopy.train_token_lm --data-path loopy/domain_support_corpus.txt --output-dir loopy/runs/domain_plain --epochs 40 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
python -m loopy.train_token_lm --data-path loopy/runs/domain_support_concept.txt --output-dir loopy/runs/domain_concept --epochs 40 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
Get-Content loopy\runs\domain_plain\best_metrics.json -Raw
Get-Content loopy\runs\domain_concept\best_metrics.json -Raw
```

## What counts as a good sign

- concept corpus gets a clearly lower validation loss
- concept corpus gets a clearly lower perplexity
- rewritten text stays readable
- generation stays coherent after decoding concept IDs back to canonical text

## What would be a bad sign

- no real difference between plain and concept runs
- concept run gets worse
- rewritten corpus becomes unnatural or brittle

## Important note

The LM trainer preserves duplicate lines on purpose.
That matters here because concept rewriting can turn several different surface sentences into the same normalized sentence.
If duplicates were removed, the concept corpus would become artificially smaller and the comparison would be invalid.

## Observed result

This experiment succeeded after the dedupe bug was fixed.

Final fair result:

- plain loss: `0.5442`
- concept loss: `0.2409`
- plain perplexity: `1.7233`
- concept perplexity: `1.2723`

Interpretation:

The concept layer clearly helped on the narrow support-domain corpus.
