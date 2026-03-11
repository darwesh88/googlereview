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

## What this means

The direction is still right:

- structured symbols seem better than independent bits
- but the current `v3` training path is still too weak
- we are still in architecture-debugging mode

The next task is no longer “make `patch_size=2` work at all.”
That part now works.

The next task is:

- keep `patch_size=2` fidelity
- lower capacity from the current high-capacity setup
- do this on Colab GPU, not local CPU

## Immediate next step

Best next hypothesis:

- keep soft assignments
- keep explicit usage pressure
- start from the working `patch_size=2`, `num_codebooks=3`, `sub_codebook_size=256` setup
- reduce capacity gradually and compare fidelity

## Decision rule

Move `v3` to the next stage only if:

- `patch_size=2` keeps high fidelity on real text
- capacity can start coming down without collapse
- codebook perplexity stays healthy

The next stage is now a Colab GPU sweep below the new `12.0 bpb` point.

## Do not do next

- do not move `v3` to real text yet
- do not rent H100s yet
- do not return to tiny `v2` rate sweeps unless there is a brand new hypothesis
