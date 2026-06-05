# V5.1 Plan

## Goal

Build the next `v5` iteration by removing the mismatch between:

- the internal prior used during codec training
- the external grouped prior used for downstream evaluation

Simple target:

- make the training-time prior look much more like the thing we actually score later

## Status

`v5.1` scaffold is now implemented.

Local smoke status:

- codec checkpoint saves
- `prior_match_loss` is nonzero
- external grouped prior loads the `v5.1` checkpoint and trains

The branch is now parked before a full TinyStories benchmark.

Reason:

- the broader research review pointed to fixed patching as the more likely bottleneck
- v6 dynamic patching is now the active branch
- v5.1 remains available if we need to isolate the prior-alignment question later

## Why `v5.1` is needed

The first `v5` pass was useful, but not good enough.

Observed TinyStories results:

- raw: `1.4022`
- `v3`: `1.7467`
- `v4.2`: `1.9336`
- `v4.2 + masked predictive`: `2.0513`
- `v5`: `1.9610`

Codec side stayed strong:

- `v5` byte accuracy: `0.9966`

So the problem is not that prior-aware training destroyed the codec.

The problem is that the first `v5` prior-aware objective did not transfer well into the final grouped-prior metric.

## Main hypothesis

Two things are still wrong in `v5`:

1. the internal `CausalGroupedPriorHead` is only similar to the external grouped prior, not tightly aligned to it
2. the internal prior consumes previous **hard detached** symbol IDs, so the codec does not get sequence-level gradient through earlier symbol choices

That means `v5` is still optimizing an approximation of the real downstream task.

## Core idea

`v5.1` should keep the same codec backbone, but make the prior-aware path:

- architecture-aligned with grouped-prior evaluation
- differentiable through previous soft symbol assignments

In plain English:

- do not only tell the codec "make this patch easy to predict"
- also let it feel how earlier symbol choices affect later predictability

## What stays from `v5`

- local patch encoder
- pre-context transformer
- product quantizer
- post-context transformer
- byte decoder
- residual-detail side channel
- reconstruction and codebook losses
- grouped codebook format

## What changes in `v5.1`

### 1. Replace hard previous IDs with soft expected embeddings

For each codebook at position `t`, use the quantizer soft assignment distribution:

- `q[t, k, i]`

and convert it into an expected embedding:

- `e[t, k] = sum_i q[t, k, i] * E[k, i]`

Then shift those grouped expected embeddings right and feed them into the internal causal prior.

This matters because gradients can now flow through earlier symbol choices.

### 2. Align the internal prior architecture to the external grouped prior

The internal prior should use the same basic family as the external evaluator:

- grouped embeddings
- one recurrent causal core
- one head per codebook

Best next move:

- factor out a reusable grouped prior core module
- use it inside both:
  - `symbolic_codec_v51.py`
  - `train_patch_prior_v51.py`

That removes avoidable architecture mismatch.

### 3. Keep KL as the training loss, but on the aligned soft path

Recommended first loss:

- `prior_match_loss = KL(q_t || p_t)`

Where:

- `q_t` is the quantizer soft assignment at the current position
- `p_t` is the aligned causal prior prediction from previous soft grouped embeddings only

Still log:

- hard grouped CE
- hard grouped `bpb`

But keep the differentiable KL path as the thing that actually shapes the codec.

## Proposed `v5.1` loss

```text
total_loss =
    recon_loss
  + commitment_loss
  + codebook_loss
  + usage_loss
  + residual_usage_loss
  + prior_weight * prior_match_loss
```

Same overall structure as `v5`.

The difference is not a new scalar loss.

The difference is:

- better internal prior architecture
- differentiable sequence-level conditioning

## Architecture sketch

```text
bytes
  -> local patch encoder
  -> pre-context transformer
  -> product quantizer
  -> post-context transformer
  -> byte decoder + residual detail head

soft grouped assignments
  -> grouped expected embeddings
  -> shift-right
  -> causal grouped prior core
  -> next-symbol distributions
  -> KL(q_t || p_t)
```

## File plan

New files:

- [v51_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v51_config.py)
- [symbolic_codec_v51.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v51.py)
- [train_symbolic_codec_v51.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v51.py)
- [train_patch_prior_v51.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v51.py)

Recommended shared utility:

- `grouped_prior_core.py`

Purpose:

- reusable grouped causal prior core shared by training-time and evaluation-time paths

## First implementation order

1. copy `v5_config.py` into `v51_config.py`
2. extract a reusable grouped prior core from the existing grouped-prior path
3. copy `symbolic_codec_v5.py` into `symbolic_codec_v51.py`
4. replace hard detached prior inputs with shifted expected grouped embeddings
5. keep logging:
   - `prior_match_loss`
   - `prior_ce_loss`
   - `prior_bpb`
6. copy the training scripts into `v51` versions

## First run order

### Run 1. Local smoke

Success conditions:

- checkpoint saves
- `prior_match_loss > 0`
- `prior_bpb` logs
- reconstruction not collapsed

### Run 2. TinyStories clean benchmark

This is the gate.

`v5.1` must beat:

- `v5`: `1.9610`

And ideally also beat:

- `v4.2`: `1.9336`

If it cannot clear at least the first bar, the `v5` family is probably not worth pushing much further.

### Run 3. Twitter support robustness check

Only do this if TinyStories clearly improves.

## Success rule

`v5.1` is worth continuing only if it shows one of these:

1. TinyStories grouped prior `bpb` clearly below `1.9610`
2. TinyStories grouped prior at or below `1.9336`
3. noisy-data downstream improves too without losing the strong codec behavior

## Failure rule

If `v5.1` still cannot beat plain `v4.2` on TinyStories, then:

- the current grouped-codebook family is likely not enough
- the next move should probably shift back toward the `v3` side of the tradeoff or a larger symbol-design change
