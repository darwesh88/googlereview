# Next

`v3` is now the active branch.

## Current status

What we know:

- `v2` grouped independent bits are good enough as a reference architecture
- `v2` is not producing a downstream stream that beats raw patches
- grouped downstream targets helped, but not enough
- so the next meaningful branch is a stronger latent redesign

What `v3` has shown so far:

- old single-symbol `v3` failed badly at `patch_size=4` and `patch_size=2`
- old single-symbol `v3` at `patch_size=1` only reached about `0.75` byte accuracy and plateaued
- product-codebook `v3` is now the active branch
- soft assignments plus usage regularization were the first real `v3` breakthrough
- best `patch_size=1` toy result so far:
  - byte accuracy: `0.9265`
  - codebook perplexity: `250.97`
  - output is mostly readable
- the key `patch_size=2` milestone has now been hit:
  - `num_codebooks=4`, `40` epochs on toy data
  - byte accuracy: `0.9746`
  - exact toy reconstruction
- the same high-capacity `patch_size=2` setup also passed a short real-corpus smoke:
  - byte accuracy: `0.9892`
  - very small local errors only
- first Colab capacity-reduction runs are now in:
  - `num_codebooks=3`, `sub_codebook_size=256`, `raw_capacity_bpb=12.0`
    - byte accuracy: `0.9973`
  - `num_codebooks=4`, `sub_codebook_size=128`, `raw_capacity_bpb=14.0`
    - byte accuracy: `0.9988`
  - `num_codebooks=2`, `sub_codebook_size=256`, `raw_capacity_bpb=8.0`
    - byte accuracy: `0.9892`
  - `num_codebooks=2`, `sub_codebook_size=128`, `raw_capacity_bpb=7.0`
    - byte accuracy: `0.9889`
  - `num_codebooks=2`, `sub_codebook_size=64`, `raw_capacity_bpb=6.0`
    - byte accuracy: `0.9794`
  - same `6.0 bpb` setting at `20` epochs
    - byte accuracy: `0.9846`
  - `num_codebooks=2`, `sub_codebook_size=32`, `raw_capacity_bpb=5.0`
    - byte accuracy: `0.9464`

## What this means

The direction is still right:

- structured symbols seem better than independent bits
- but the current `v3` training path is still too weak
- we are still in architecture-debugging mode

The next task is no longer “make `patch_size=2` work at all.”
That part now works.

The next task is now narrower:

- keep `patch_size=2` fidelity
- hold `7.0 bpb` as the safer `v3` baseline
- hold `5.0 bpb` as the best downstream `v3` point
- stop trying to fix `5.0 bpb` with more training alone
- switch to `v4`: cross-patch context around the quantizer

## Immediate next step

Best next hypothesis:

- keep soft assignments
- keep explicit usage pressure
- keep `patch_size=2`
- keep the existing codebook family
- keep cross-patch transformer context before quantization
- keep cross-patch transformer context after quantization
- add a small residual-detail side channel for exact local fixes

## Decision rule

Move `v4` forward only if:

- it improves the current `5.0 bpb` reconstruction quality
- while keeping downstream usefulness near the current `5.0 bpb` `v3` point
- and codebook perplexity stays healthy

## Immediate next run

`v4` is now implemented in:

- [train_symbolic_codec_v4.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v4.py)
- [symbolic_codec_v4.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v4.py)
- [v4_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v4_config.py)

Immediate next run:

- first `v4` real comparison is done
- first `v4.2` toy comparison is encouraging
- first real `v4.2` comparison is now a win
- next run should be a small `v4.2` tuning pass:
  - same `5.0 bpb` semantic capacity
  - tune only the residual branch controls

Main metrics:

- reconstruction quality first
- downstream `bpb` second

## Latest downstream result

Observed:

- raw patch prior:
  - `bpb = 2.9473`
- `v3` grouped prior on `7.0 bpb`:
  - `bpb = 3.8102`
- `v3` grouped prior on `6.0 bpb`:
  - `bpb = 3.4601`
  - longer prior: `3.1844`
- `v3` grouped prior on `5.0 bpb`:
  - `bpb = 3.0444`
  - `10` epochs: `2.9174`
  - `20` epochs: `2.8497`

Interpretation:

- lower-capacity `v3` symbols are becoming easier to predict downstream
- `6.0 bpb` is better downstream than `7.0 bpb`
- `5.0 bpb` has now beaten the raw downstream baseline
- longer prior training improved `6.0 bpb`, but not enough to catch `5.0 bpb`
- a longer codec run at `5.0 bpb` did not materially improve reconstruction
- the next step should now be an architecture change, not more training on the same design
- `v4` is now that architecture branch
- first `v4` real result was effectively a tie with `v3`
- `v4.2` is now the winning branch at `5.0 bpb`
- the next step is tuning, not redesign

## Do not do next

- do not move `v3` to real text yet
- do not rent H100s yet
- do not return to tiny `v2` rate sweeps unless there is a brand new hypothesis
- do not jump straight to H100s while Colab is still enough for these frontier tests
- do not push `v3` below `5.0 bpb` until downstream usefulness is tested
