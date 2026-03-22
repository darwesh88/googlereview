# Breakpoint

Resume from here.

## Stable facts

- corrected raw downstream baseline:
  - raw patch prior, `patch_size=2`, `20` epochs
  - `bpb = 2.5258`
  - `accuracy = 0.5233`
- best learned downstream result:
  - `v3`, `5.0 bpb`, grouped prior, `20` epochs
  - `bpb = 2.8497`
- best reconstruction result:
  - `v4.2`, `6.0 bpb`
  - `byte_accuracy = 0.9907`
- best masked-predictive `v4.2` result:
  - `byte_accuracy = 0.9912`
  - downstream grouped prior `bpb = 3.1301`
- clean TinyStories downstream benchmark:
  - raw: `1.4022`
  - `v3`: `1.7467`
  - `v4.2`: `1.9336`
  - `v4.2 + masked predictive`: `2.0513`

## Current interpretation

- `v3` is still the best branch for downstream predictability
- `v4.2` is still the best branch for reconstruction
- the current `v42` harness neighborhood is too weak to close the corrected raw gap by local tuning alone
- clean data did not remove the gap, so the next bottleneck is mainly architecture

## Harness state

The harness is real and should stay.

It now supports:

- `prepare`
- `status`
- `run`
- `collect`
- `bundle`
- `restore`
- `ingest`

Use it as:

- local repo = control plane
- Colab or another GPU box = execution worker
- batch artifacts zip = persistence layer

## Do not resume with

- more `v42_masked_grid_10` local sweeps
- more tiny residual-control tuning
- more small masked-predictive parameter nudges on the same Twitter support corpus

## Resume with

1. keep the TinyStories clean benchmark as a standing benchmark
2. keep the noisy Twitter support corpus as the robustness benchmark
3. move to the next larger architecture branch

## Likely next architecture

If clean-data results still do not move the gap enough, the next branch should be:

- `v5 = prior-aware codec`

See:

- [V5_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/V5_PLAN.md)

That branch should optimize for:

- reconstruction
- codebook health
- residual sparsity
- downstream predictability during codec training
