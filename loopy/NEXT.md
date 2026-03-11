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
- new product-codebook `v3` is now in the workspace
- first product-codebook toy smoke result:
  - byte accuracy: `0.7589`
  - codebook perplexity: `6.90`
  - output is partially readable, but still not strong enough

## What this means

The direction is still right:

- structured symbols seem better than independent bits
- but the current `v3` training path is still too weak
- we are still in architecture-debugging mode

The next task is not a real-corpus run yet.
It is a clean product-codebook toy sanity run on the committed branch.

## Immediate next step

Run:

`python -m loopy.train_symbolic_codec_v3 --data-path loopy/example_corpus.txt --output-dir loopy/runs/v3_product_smoke --epochs 20 --batch-size 4 --max-seq-len 128 --patch-size 1 --embed-dim 128 --latent-dim 128 --encoder-layers 2 --decoder-layers 2 --num-heads 4 --dropout 0.0 --weight-decay 0.0 --num-codebooks 2 --sub-codebook-size 256 --commitment-weight 0.05 --codebook-weight 0.25 --predictive-weight 0.0 --overfit-all`

Then inspect:

- `best_metrics.json`
- `sample_reconstruction.txt`

## Decision rule

Move `v3` to the real corpus only if:

- toy reconstruction becomes clearly readable
- byte accuracy improves meaningfully over the current partial result
- codebook perplexity stays healthy

If the product-codebook toy result still stalls, the next move is another `v3` architecture fix, not more training scale.

## Do not do next

- do not move `v3` to real text yet
- do not rent H100s yet
- do not return to tiny `v2` rate sweeps unless there is a brand new hypothesis
