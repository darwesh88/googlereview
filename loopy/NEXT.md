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
- first small tuning pass is now done
- next run should combine the two helpful changes:
  - `residual_usage_weight = 0.005`
  - `residual_gate_bias = -1.5`
- combined run is now done and was worse
- next run should return to single-variable tuning from the best point:
  - keep `residual_usage_weight = 0.005`
  - keep `residual_gate_bias = -2.0`
  - test one smaller change at a time
- lower residual pressure (`0.003`) is now tested and was worse
- next step should stop this tuning sweep and move back to downstream usefulness testing

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
- the current best tuned `v4.2` point is `0.96058` byte accuracy
- the combined tuning test regressed, so interaction effects matter
- the lower residual-pressure test also regressed, so this tuning sweep is likely done
- first `v4.2` downstream grouped-prior result is now in:
  - `bpb = 2.9551`
  - this is very close to raw, but still slightly worse than the raw `patch_size=2` prior (`2.9473`)
  - it is also worse than the best downstream `v3` point (`2.8497`)

## Do not do next

- do not move `v3` to real text yet
- do not rent H100s yet
- do not return to tiny `v2` rate sweeps unless there is a brand new hypothesis
- do not jump straight to H100s while Colab is still enough for these frontier tests
- do not push `v3` below `5.0 bpb` until downstream usefulness is tested
- do not keep micro-tuning `v4.2` residual controls unless there is a very specific new hypothesis

## Latest downstream result

Observed:

- raw patch prior at `patch_size=2`:
  - `bpb = 2.9473`
- best downstream `v3` point at `5.0 bpb`, `20` epochs:
  - `bpb = 2.8497`
- first downstream `v4.2` grouped prior at the best tuned `5.0 bpb` checkpoint:
  - `bpb = 2.9551`

Interpretation:

- `v4.2` clearly improved reconstruction over `v3` and `v4`
- but it has not yet improved downstream usefulness over raw
- and it has not matched the best downstream `v3` point

Decision:

- `v4.2` is currently the best reconstruction architecture at `5.0 bpb`
- `v3` is still the best downstream architecture at `5.0 bpb`
- `v4.2` at `6.0 bpb` is now the best `6.0 bpb` reconstruction point:
  - `byte_accuracy = 0.9907`
- but its first downstream grouped prior at `6.0 bpb` is:
  - `bpb = 3.2052`
  - this is better than old `v3` at `6.0 bpb` (`3.4601` / `3.1844` longer)
  - but still worse than raw (`2.9473`) and worse than best downstream `v3` at `5.0 bpb` (`2.8497`)
- the next branch should not be more residual tuning
- the next branch should test whether a small predictive objective on top of the best `v4.2` `6.0 bpb` point can recover the downstream gap
