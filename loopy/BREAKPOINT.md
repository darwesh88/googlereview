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

## Current interpretation

- `v3` is still the best branch for downstream predictability
- `v4.2` is still the best branch for reconstruction
- the current `v42` harness neighborhood is too weak to close the corrected raw gap by local tuning alone

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

1. a cleaner benchmark dataset
2. one controlled raw vs `v3` vs `v4.2` comparison on that dataset
3. then the next larger architecture branch if needed

## Likely next architecture

If clean-data results still do not move the gap enough, the next branch should be:

- `v5 = prior-aware codec`

That branch should optimize for:

- reconstruction
- codebook health
- residual sparsity
- downstream predictability during codec training
