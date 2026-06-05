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
4. Run the first `v6` dynamic-patching benchmark before any more fixed-patch tuning.
5. Keep the harness, but use it on a stronger hypothesis, not on more nearby `v4.2` settings.

The first clean-data scaffold now exists in:

- [prepare_hf_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/prepare_hf_corpus.py)
- [CLEAN_DATA_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/CLEAN_DATA_PLAN.md)
- [experiment_plans/clean_tinystories_compare.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/clean_tinystories_compare.json)
- [experiment_plans/v6_dynamic_tinystories.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/v6_dynamic_tinystories.json)

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
- first `v5` reference:
  - codec byte accuracy: `0.9966`
  - grouped prior `bpb = 1.9610`
  - grouped prior accuracy: `0.6309`

Interpretation:

- raw still wins clearly on clean data
- `v3` is still the best learned downstream branch
- `v4.2` is still the best reconstruction branch
- masked predictive did not help the clean downstream benchmark
- `v5` preserved strong reconstruction, but its internal prior loss still did not transfer into a better external grouped-prior result
- so the main bottleneck is not just noisy customer-support data

## Active next branch

`v5` and `v5.1` are no longer the main active line.

`v5.1` is implemented and smoke-tested, but the project is moving directly to:

- `v6 = dynamic byte patching`

Working plan:

- [V6_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/V6_PLAN.md)

Meaning:

- stop assuming every symbol should represent exactly 2 bytes
- let patches have variable byte lengths
- test whether boundary-aware dynamic patches improve downstream bpb
- keep the same grouped-prior evaluation so results remain comparable

Current `v5` status:

- scaffold exists in:
  - [v5_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v5_config.py)
  - [symbolic_codec_v5.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v5.py)
  - [train_symbolic_codec_v5.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v5.py)
  - [train_patch_prior_v5.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v5.py)
- local smoke passed:
  - codec checkpoint saves
  - `prior_match_loss` is nonzero
  - grouped prior loads the `v5` checkpoint and trains
- first TinyStories pass completed:
  - codec byte accuracy: `0.9966`
  - grouped prior `bpb = 1.9610`
  - result is better than masked-predictive `v4.2`, but still worse than plain `v4.2` and `v3`

Current `v5.1` status:

- scaffold exists in:
  - [grouped_prior_core.py](C:/Users/adarw/Desktop/googlereview/loopy/grouped_prior_core.py)
  - [v51_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v51_config.py)
  - [symbolic_codec_v51.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v51.py)
  - [train_symbolic_codec_v51.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v51.py)
  - [train_patch_prior_v51.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v51.py)
- local smoke passed:
  - codec checkpoint saves
  - `prior_match_loss` is nonzero
  - external grouped prior loads the `v5.1` checkpoint and trains
- parked before full TinyStories benchmark

Immediate next run order:

1. TinyStories `v6` dynamic codec benchmark
2. TinyStories `v6` grouped prior benchmark
3. `space` vs `boundary` vs `fixed` ablation only if the first v6 run is not obviously dead
4. Twitter support `v6` robustness check only if TinyStories clearly improves over `v5`

Current `v6` status:

- scaffold exists in:
  - [dynamic_patching.py](C:/Users/adarw/Desktop/googlereview/loopy/dynamic_patching.py)
  - [v6_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v6_config.py)
  - [symbolic_codec_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v6.py)
  - [train_symbolic_codec_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v6.py)
  - [train_patch_prior_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v6.py)
- local smoke passed:
  - codec checkpoint saves
  - grouped prior loads the v6 checkpoint and trains
  - dynamic patch encoding produces variable patch lengths

## Harness use

Keep the harness as the default workflow.

Do not throw it away.

But use it for:

- clean benchmark comparisons
- architecture ablations
- new branch validation

Not for:

- more small `v4.2` mask/probability sweeps in the current weak neighborhood
