# Loopy

Loopy is a research repo for learned text latents.

The core question is simple:

- can a learned patch-symbol stream become a better training target than raw byte patches
- while still reconstructing the original text well

## Current answer

Not yet.

What is true now:

- raw downstream baseline at `patch_size=2`, `20` epochs:
  - `bpb = 2.5258`
  - `accuracy = 0.5233`
- best learned downstream result so far:
  - `v3`, `5.0 bpb`, grouped prior, `20` epochs
  - `bpb = 2.8497`
  - `accuracy = 0.4472`
- best balanced reconstruction result so far:
  - `v4.2`, `6.0 bpb`
  - `byte_accuracy = 0.9907`
- best `v4.2` masked-predictive result so far:
  - `v4.2`, `6.0 bpb`, masked predictive
  - `byte_accuracy = 0.9912`
  - downstream grouped prior `bpb = 3.1301`

So the project has split cleanly:

- `v3` is still the best downstream branch
- `v4.2` is the best reconstruction branch
- neither branch beats raw on the corrected downstream benchmark

## Branch summary

### v1

- symbolic middleware probe
- useful for early intuition
- archived

### v2

- grouped independent-bit codec
- proved reconstruction/compression tradeoffs are real
- did not produce a downstream stream that beats raw
- archived as a reference branch

### v3

- product-codebook symbolic codec
- first branch to get close downstream
- best learned downstream point is still here

### v4 / v4.2

- adds cross-patch context
- `v4.2` adds a residual-detail side channel
- best reconstruction quality now lives here
- masked predictive objective helped a bit, but not enough

## Current recommendation

Do not keep sweeping the current `v42_masked_grid_10` neighborhood.

The gap to the corrected raw baseline is too large for more tiny parameter sweeps to be the right next move.

The next useful work is:

1. keep the harness as infrastructure
2. move to a cleaner data regime so architecture signal is easier to read
3. then test the next larger hypothesis shift

## Docs to read first

- [NEXT.md](C:/Users/adarw/Desktop/googlereview/loopy/NEXT.md): immediate next steps
- [BREAKPOINT.md](C:/Users/adarw/Desktop/googlereview/loopy/BREAKPOINT.md): resume-from-here snapshot
- [HARNESS.md](C:/Users/adarw/Desktop/googlereview/loopy/HARNESS.md): experiment control-plane workflow
- [CLEAN_DATA_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/CLEAN_DATA_PLAN.md): clean benchmark setup
- [RESEARCH_LOG.md](C:/Users/adarw/Desktop/googlereview/loopy/RESEARCH_LOG.md): full chronological history

## Harness status

The experiment harness is now the default workflow for multi-run comparisons:

- prepare a batch locally
- run exact commands on Colab or another GPU worker
- collect and bundle artifacts so sessions are not lost
- restore and ingest results back into the local ledger

Main files:

- [experiment_runner.py](C:/Users/adarw/Desktop/googlereview/loopy/experiment_runner.py)
- [experiment_baselines.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_baselines.json)
- [HARNESS.md](C:/Users/adarw/Desktop/googlereview/loopy/HARNESS.md)

## Practical state

Loopy is past the "does anything work at all?" stage.

It is now at the harder stage:

- choose better data
- choose the next architecture hypothesis carefully
- only then resume larger remote sweeps
