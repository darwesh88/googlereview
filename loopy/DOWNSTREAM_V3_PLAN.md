# Downstream v3 Plan

`v3` has now reached a real compression frontier:

- stable baseline: `7.0 bpb`
- low-capacity frontier: `6.0 bpb`
- too-far point: `5.0 bpb`

The next question is no longer just reconstruction.

The next question is:

- are the `v3` patch symbols a better downstream prediction target than raw byte patches?

## Why this matters

If `v3` is going to matter for later language-model training, the symbol stream should become:

- shorter
- structured
- easier for the next model to predict

Reconstruction alone is not enough.

## Planned comparison

Compare three patch-level priors at `patch_size=2`:

1. raw patch prior
2. `v3` grouped-symbol prior on the `7.0 bpb` checkpoint
3. `v3` grouped-symbol prior on the `6.0 bpb` checkpoint

Main metric:

- validation `bpb`

Lower is better.

## Decision rule

- If `v3` grouped priors get close to raw, the `v3` branch is becoming useful for downstream LM training.
- If `v3` grouped priors stay much worse than raw, then `v3` is still mainly a reconstruction codec, not yet a better training representation.
