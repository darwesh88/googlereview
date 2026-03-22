# Next

## Current snapshot

The corrected benchmark to beat is now:

- raw patch prior, `patch_size=2`, `20` epochs
  - `bpb = 2.5258`
  - `accuracy = 0.5233`

Best learned results so far:

- best downstream branch:
  - `v3`, `5.0 bpb`, grouped prior, `20` epochs
  - `bpb = 2.8497`
- best reconstruction branch:
  - `v4.2`, `6.0 bpb`
  - `byte_accuracy = 0.9907`
- best balanced `v4.2` masked-predictive point:
  - `byte_accuracy = 0.9912`
  - downstream grouped prior `bpb = 3.1301`

Interpretation:

- `v3` still wins on downstream usefulness
- `v4.2` wins on reconstruction quality
- current `v42` masked-predictive sweeps are too far from raw to justify more local tuning
- the first clean TinyStories benchmark says the architecture gap is real even on cleaner text

## Immediate priorities

1. Freeze the current `v42_masked_grid_10` sweep after any run already in flight.
2. Stop treating "maybe the noisy data is the whole problem" as an open question.
3. Keep clean and noisy benchmarks together from now on.
4. Keep the harness, but use it on a stronger hypothesis, not on more nearby `v4.2` settings.

The first clean-data scaffold now exists in:

- [prepare_hf_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/prepare_hf_corpus.py)
- [CLEAN_DATA_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/CLEAN_DATA_PLAN.md)
- [experiment_plans/clean_tinystories_compare.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/clean_tinystories_compare.json)

## Clean benchmark result

TinyStories was run as the first clean benchmark.

Observed downstream results:

- raw patch prior:
  - `bpb = 1.4022`
  - `accuracy = 0.7193`
- best downstream `v3` reference:
  - `bpb = 1.7467`
  - `accuracy = 0.6539`
- best balanced `v4.2` reference:
  - `bpb = 1.9336`
  - `accuracy = 0.6381`
- masked-predictive `v4.2` reference:
  - `bpb = 2.0513`
  - `accuracy = 0.6114`

Interpretation:

- raw still wins clearly on clean data
- `v3` is still the best learned downstream branch
- `v4.2` is still the best reconstruction branch
- masked predictive did not help the clean downstream benchmark
- so the main bottleneck is not just noisy customer-support data

## Next branch

The next branch should be:

- `v5 = prior-aware codec`

Meaning:

- keep the best codec structure pieces from `v4.2`
- train the codec more directly for downstream predictability instead of reconstruction alone
- keep the clean TinyStories benchmark and the noisy Twitter support benchmark side by side while testing it

## Harness use

Keep the harness as the default workflow.

Do not throw it away.

But use it for:

- clean benchmark comparisons
- architecture ablations
- new branch validation

Not for:

- more small `v4.2` mask/probability sweeps in the current weak neighborhood
