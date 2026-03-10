# Real Corpus Runbook

This is the next Loopy stage.

The goal is to test the symbolic middle layer on a real domain corpus instead of synthetic text.

## 1. Put raw files in one place

Recommended location:

- `loopy/data/raw/<dataset-name>/`

Supported input formats right now:

- `.txt`
- `.jsonl` with a text field

## 2. Prepare the corpus

Run the prep step first.
This cleans whitespace, filters junk lines, optionally redacts sensitive patterns, and writes one sample per line.

Example with `.txt` files:

```powershell
python -m loopy.prepare_real_corpus --input loopy/data/raw/support_october --output loopy/data/real/support_october.txt --min-chars 24 --max-chars 280 --min-tokens 4 --dedupe --redact-emails --redact-urls --redact-long-numbers
```

Example with `.jsonl` files:

```powershell
python -m loopy.prepare_real_corpus --input loopy/data/raw/support_jsonl --output loopy/data/real/support_jsonl.txt --text-key body --min-chars 24 --max-chars 280 --min-tokens 4 --dedupe --redact-emails --redact-urls --redact-long-numbers
```

What you get:

- cleaned corpus: `loopy/data/real/<dataset-name>.txt`
- prep report: `loopy/data/real/<dataset-name>.report.json`

## 3. Create the domain lexicon

Start from:

- [concepts.real.template.json](C:/Users/adarw/Desktop/googlereview/loopy/concepts.real.template.json)

Copy it to a domain-specific file, for example:

- `loopy/concepts.support.real.json`

Good first target:

- `50-100` clear concepts
- high-frequency domain entities first
- only merge terms that really share meaning

## 4. Rewrite the prepared corpus

```powershell
python -m loopy.rewrite_corpus --input loopy/data/real/support_october.txt --output loopy/runs/support_october_concept.txt --lexicon loopy/concepts.support.real.json
Get-Content loopy\runs\support_october_concept.report.json -Raw
```

Use the report to sanity-check:

- total replacements
- concepts used
- first rewritten examples

## 5. Train the plain baseline

```powershell
python -m loopy.train_token_lm --data-path loopy/data/real/support_october.txt --output-dir loopy/runs/support_october_plain --epochs 45 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
```

## 6. Train the concept baseline

```powershell
python -m loopy.train_token_lm --data-path loopy/runs/support_october_concept.txt --output-dir loopy/runs/support_october_concept --epochs 45 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --weight-decay 0.01
```

## 7. Inspect both runs

```powershell
python -m loopy.inspect_run --run-dir loopy/runs/support_october_plain
python -m loopy.inspect_run --run-dir loopy/runs/support_october_concept --lexicon loopy/concepts.support.real.json --output loopy/runs/support_october_concept.decoded.txt
Get-Content loopy\runs\support_october_plain\best_metrics.json -Raw
Get-Content loopy\runs\support_october_concept\best_metrics.json -Raw
Get-Content loopy\runs\support_october_concept.decoded.txt -Raw
```

## 8. Decision rule

This stage is a success if:

- the concept run still beats the plain run on validation loss or perplexity
- decoded outputs stay readable
- the rewritten corpus looks semantically sane

This stage fails if:

- the gain disappears entirely on real data
- the rewrite starts flattening too much meaning
- the lexicon becomes too brittle or too manual to maintain
