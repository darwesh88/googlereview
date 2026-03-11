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
- but the first `patch_size=2` run is still weak:
  - byte accuracy: `0.5652`
  - codebook perplexity: `169.11`
  - output still breaks badly
- a higher-capacity `patch_size=2` diagnostic improves things a bit:
  - `num_codebooks=3`
  - byte accuracy: `0.6298`
  - output is still not readable enough
- so capacity matters, but capacity alone is not enough

## Next resume step

Do not move `v3` to real text yet.

Next resume step:

- use the soft-assignment `patch_size=1` run as the new `v3` reference point
- focus all next work on making `patch_size=2` readable
- treat `patch_size=2` as both a capacity and modeling problem
- if `patch_size=2` still stalls, fix `v3` again before any scaling
