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

## Immediate priorities

1. Freeze the current `v42_masked_grid_10` sweep after any run already in flight.
2. Stop using the noisy Twitter support slice as the only proving ground.
3. Introduce a cleaner data benchmark before the next architecture branch.
4. Keep the harness, but use it on a stronger hypothesis, not on more nearby `v4.2` settings.

The first clean-data scaffold now exists in:

- [prepare_hf_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/prepare_hf_corpus.py)
- [CLEAN_DATA_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/CLEAN_DATA_PLAN.md)
- [experiment_plans/clean_tinystories_compare.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/clean_tinystories_compare.json)

## Data decision

Do not continue only on the mixed Twitter support corpus.

Use a two-track data regime:

1. A cleaner, narrower dataset to measure architecture signal.
2. The current noisy support corpus as a robustness check.

Why:

- the current corpus is broad, messy, and heterogeneous
- that makes it useful as a stress test
- but it is poor as the only benchmark when we are still trying to detect whether a latent architecture is fundamentally good

## Recommended next experiment block

Before building `v5`, run one controlled clean-data comparison:

1. Choose one cleaner corpus.
   Good first candidates:
   - a small clean natural-language corpus such as TinyStories
   - a clean WikiText slice
   - or a curated single-domain support dataset with consistent language
2. Match it roughly to the same order of scale as the current working corpus.
3. Re-run:
   - raw patch prior baseline
   - best downstream `v3`
   - best balanced `v4.2`
4. Compare the gap against raw.

Decision rule:

- if learned streams still lose badly on clean data, the next bottleneck is mainly architecture
- if the gap shrinks sharply on clean data, data heterogeneity is a major part of the current problem

## After the data check

If clean-data results are still weak, the next branch should be:

- `v5 = prior-aware codec`

Meaning:

- keep the best codec structure pieces from `v4.2`
- train the codec more directly for downstream predictability instead of reconstruction alone

## Harness use

Keep the harness as the default workflow.

Do not throw it away.

But use it for:

- clean benchmark comparisons
- architecture ablations
- new branch validation

Not for:

- more small `v4.2` mask/probability sweeps in the current weak neighborhood
