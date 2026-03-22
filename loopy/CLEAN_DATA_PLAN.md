# Clean Data Plan

The next Loopy phase should not rely only on the noisy Twitter support slice.

Use a cleaner benchmark first so architecture signal is easier to read.

## Why

The current Twitter support corpus is useful as a stress test, but it is:

- heterogeneous
- noisy
- full of local detail drift
- broad enough to hide whether a latent architecture is fundamentally improving

So the next benchmark should be:

- clean
- easy to reproduce
- line-based like the current corpora
- large enough for the same train/validation setup

## First clean benchmark

Start with TinyStories.

Why TinyStories first:

- clean natural language
- easy to fetch reproducibly
- narrow enough that architecture differences should be easier to see
- cheap enough to benchmark quickly

## Prepare the corpus

After installing `loopy/requirements.txt`, run:

```powershell
python -m loopy.prepare_hf_corpus --dataset tinystories --output loopy/data/real/tinystories_5k.txt --max-samples 5000 --min-chars 40 --max-chars 600 --min-tokens 6 --dedupe
```

This writes:

- `loopy/data/real/tinystories_5k.txt`
- `loopy/data/real/tinystories_5k.report.json`

## Benchmark plan

Use the harness plan:

- [experiment_plans/clean_tinystories_compare.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/clean_tinystories_compare.json)

Prepare it:

```powershell
python -m loopy.experiment_runner prepare --plan-file loopy/experiment_plans/clean_tinystories_compare.json
```

This compares:

1. raw patch prior baseline
2. best downstream `v3` reference
3. best balanced `v4.2` reference
4. best masked-predictive `v4.2` reference

## First result

TinyStories has now been run.

Observed downstream results:

- raw patch prior:
  - `bpb = 1.4022`
  - `accuracy = 0.7193`
- best downstream `v3`:
  - `bpb = 1.7467`
  - `accuracy = 0.6539`
- best balanced `v4.2`:
  - `bpb = 1.9336`
  - `accuracy = 0.6381`
- masked-predictive `v4.2`:
  - `bpb = 2.0513`
  - `accuracy = 0.6114`

Interpretation:

- clean data did not close the raw gap
- this means the current main bottleneck is architecture, not just data heterogeneity
- keep TinyStories as a standing clean benchmark for future branches

## Decision rule

If learned streams still lose badly to raw on TinyStories:

- the next bottleneck is mostly architecture
- move to a bigger `v5` hypothesis

If the gap shrinks a lot on TinyStories:

- data heterogeneity is a major part of the current problem
- keep a clean benchmark and a noisy benchmark side by side

## Second clean benchmark

If TinyStories is useful, follow it with:

```powershell
python -m loopy.prepare_hf_corpus --dataset wikitext103 --output loopy/data/real/wikitext103_5k.txt --max-samples 5000 --min-chars 40 --max-chars 600 --min-tokens 6 --dedupe
```

This gives a second clean benchmark with a different style:

- more factual
- less story-like
- still cleaner than the Twitter support stress test
