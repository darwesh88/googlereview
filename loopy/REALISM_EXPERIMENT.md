# Realism Experiment

This is the harder follow-up test.

## Why this exists

The first support-domain run was favorable to the concept lexicon.
That was good. It proved the middle layer can help.

Now we need a messier test:

- some aliases are covered by the lexicon
- some are not
- wording is less controlled
- names and extra clauses add noise

## Goal

Check whether the concept layer still helps when coverage is only partial.

## Commands

```powershell
python -m loopy.make_noisy_support_corpus --output loopy/noisy_support_corpus.txt --samples 360 --seed 11 --uncovered-ratio 0.30
python -m loopy.rewrite_corpus --input loopy/noisy_support_corpus.txt --output loopy/runs/noisy_support_concept.txt --lexicon loopy/concepts.support.json
python -m loopy.train_token_lm --data-path loopy/noisy_support_corpus.txt --output-dir loopy/runs/noisy_plain --epochs 45 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
python -m loopy.train_token_lm --data-path loopy/runs/noisy_support_concept.txt --output-dir loopy/runs/noisy_concept --epochs 45 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
python -m loopy.inspect_run --run-dir loopy/runs/noisy_plain
python -m loopy.inspect_run --run-dir loopy/runs/noisy_concept --lexicon loopy/concepts.support.json --output loopy/runs/noisy_concept.decoded.txt
Get-Content loopy\runs\noisy_plain\best_metrics.json -Raw
Get-Content loopy\runs\noisy_concept\best_metrics.json -Raw
Get-Content loopy\runs\noisy_concept.decoded.txt -Raw
```

## What counts as a win

- the concept run still beats plain text
- decoded generations stay readable
- the gain survives even though some wording is outside the lexicon

## What would downgrade the idea

- the concept win disappears on partial coverage
- decoded outputs become awkward
- the model depends too heavily on a hand-crafted closed world

## Observed result

This harder test still favored the concept layer.

Final fair result:

- plain loss: `0.6243`
- concept loss: `0.5076`
- plain perplexity: `1.8669`
- concept perplexity: `1.6614`

Interpretation:

The win became smaller, but it survived partial coverage and noisier wording.
That makes the direction more believable.
