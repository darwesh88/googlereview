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
- first `patch_size=2` toy result with the same idea is still weak:
  - byte accuracy: `0.5652`
  - codebook perplexity: `169.11`
  - output is not readable enough yet
- a higher-capacity `patch_size=2` diagnostic helped, but not enough:
  - `num_codebooks=3`
  - byte accuracy: `0.6298`
  - output still not readable enough

## What this means

The direction is still right:

- structured symbols seem better than independent bits
- but the current `v3` training path is still too weak
- we are still in architecture-debugging mode

The next task is not a real-corpus run yet.
It is to make the soft-assignment `v3` scale from `patch_size=1` to `patch_size=2`.
The latest diagnostic says that capacity matters, but it is not the only issue.

## Immediate next step

Run the next `patch_size=2` stabilization experiment, not another `patch_size=1` repeat.
Best next hypothesis:

- keep soft assignments
- keep explicit usage pressure
- improve patch-level modeling quality, not just capacity

## Decision rule

Move `v3` to the real corpus only if:

- `patch_size=2` toy reconstruction becomes clearly readable
- `patch_size=2` byte accuracy improves meaningfully over the current `0.5652`
- codebook perplexity stays healthy

If `patch_size=2` still stalls, the next move is another `v3` architecture fix, not more training scale.

## Do not do next

- do not move `v3` to real text yet
- do not rent H100s yet
- do not return to tiny `v2` rate sweeps unless there is a brand new hypothesis
