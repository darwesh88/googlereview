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

## Next resume step

Do not move `v3` to real text yet.

Next resume step:

- commit the product-codebook `v3` branch
- use that committed version as the new reference point
- run one clean toy sanity test on it
- if toy reconstruction still stalls, fix `v3` again before any scaling
