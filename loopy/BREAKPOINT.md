# Breakpoint

Loopy has pivoted again.

## Current state

- v1 symbolic middleware is archived
- v2 grouped independent-bit codec is archived as the current reference branch
- v3 codebook / patch-symbol work is now the active direction
- product-codebook `v3` changes are the current live workspace state

## What v2 settled

- reconstruction on real noisy text works
- there is a real compression vs fidelity tradeoff
- grouped downstream symbols help more than plain bitwise learned targets
- but the learned downstream stream still loses clearly to raw patches

So the next bottleneck is not:

- more tiny rate sweeps
- more grouped packing tweaks
- more light predictive-loss tuning

It is:

- a stronger latent redesign

## Current v3 status

Single-symbol `v3` was too weak:

- `patch_size=4`: blank output
- `patch_size=2`: still failed badly
- `patch_size=1`: plateaued around `0.75` byte accuracy

Product-codebook `v3` is now in the workspace:

- multiple sub-symbols per patch instead of one symbol
- first toy smoke result:
  - byte accuracy: `0.7589`
  - codebook perplexity: `6.90`
  - output is partially readable, but still weak
- soft assignments plus usage regularization changed the picture:
  - `patch_size=1` now reaches `0.9265` byte accuracy
  - codebook perplexity is about `250.97`
  - output is mostly readable
- `patch_size=2` is now viable at high capacity:
  - toy run with `num_codebooks=4`, `40` epochs
  - byte accuracy: `0.9746`
  - exact toy reconstruction
- short real-corpus smoke with the same high-capacity setup:
  - byte accuracy: `0.9892`
  - only tiny detail errors in the sample
- important caveat:
  - this setup has `raw_capacity_bpb = 16.0`
  - so it proves viability, not efficiency
- the first capacity-reduction sweep changed that:
  - `num_codebooks=2`, `sub_codebook_size=256`, `raw_capacity_bpb=8.0`
  - byte accuracy: `0.9892`
  - still near-exact real-text reconstruction
  - this became the first best efficiency-oriented `v3` baseline
- the next sweep pushed lower:
  - `num_codebooks=2`, `sub_codebook_size=128`, `raw_capacity_bpb=7.0`
  - byte accuracy: `0.9889`
  - this is now the best efficiency-oriented `v3` baseline
  - `num_codebooks=2`, `sub_codebook_size=64`, `raw_capacity_bpb=6.0`
  - byte accuracy: `0.9794`
  - this is the current cliff test: still alive, but clearly degraded
  - the same `6.0 bpb` run at `20` epochs improved to:
    - byte accuracy: `0.9846`
    - this means `6.0 bpb` is now a real frontier point, not just a failure case
  - the next lower test at `5.0 bpb` gave:
    - byte accuracy: `0.9464`
    - this is now clearly below the current quality bar

## Next resume step

Next resume step:

- use the `7.0 bpb` run as the new `v3` reference point
- keep the `8.0 bpb` run as the safer reference
- treat the new `6.0 bpb` `20`-epoch run as the current low-capacity frontier
- stop lowering capacity for now
- move to downstream usefulness testing for `v3`
- use [train_patch_prior_v3.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v3.py) for grouped-symbol priors on `v3`
- downstream comparison now says:
  - raw patch prior: `2.9473 bpb`
  - `v3` grouped at `7.0 bpb`: `3.8102 bpb`
  - `v3` grouped at `6.0 bpb`: `3.4601 bpb`
  - `v3` grouped at `5.0 bpb`: `3.0444 bpb`
- next resume step:
  - run a longer grouped prior on the `5.0 bpb` checkpoint
