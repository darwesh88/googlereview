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
- first `v5` TinyStories result:
  - codec byte accuracy: `0.9966`
  - grouped prior `bpb = 1.9610`
  - grouped prior accuracy: `0.6309`

## Current interpretation

- `v3` is still the best branch for downstream predictability
- `v4.2` is still the best branch for reconstruction
- the current `v42` harness neighborhood is too weak to close the corrected raw gap by local tuning alone
- clean data did not remove the gap, so the next bottleneck is mainly architecture
- `v5` kept strong reconstruction, but the internal prior head still did not transfer into a downstream win
- `v5.1` was implemented and smoke-tested, but is parked before full TinyStories benchmarking
- the active hypothesis is now that fixed 2-byte patching is the bigger bottleneck
- `v6` dynamic patching is the next branch

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
3. run the first `v6` dynamic-patching benchmark before resuming any broad sweep

## Likely next architecture

The next branch is now:

- `v6 = dynamic byte patching`

See:

- [V6_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/V6_PLAN.md)
- [dynamic_patching.py](C:/Users/adarw/Desktop/googlereview/loopy/dynamic_patching.py)
- [symbolic_codec_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v6.py)
- [train_symbolic_codec_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v6.py)
- [train_patch_prior_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v6.py)

That branch should optimize for:

- variable byte-patch boundaries
- reconstruction
- codebook health
- downstream grouped-prior bpb

Current status:

- `v5` scaffold is implemented
- local smoke passed on `example_corpus.txt`
- first TinyStories `v5` benchmark is complete and did not clear the bar
- `v5.1` scaffold is now implemented
- local `v5.1` smoke passed
- `v6` scaffold is now implemented
- local `v6` codec and prior smokes passed
- next decision should come from the first TinyStories `v6` benchmark, not from more `v4.2` sweeps or more `v5` micro-tuning
