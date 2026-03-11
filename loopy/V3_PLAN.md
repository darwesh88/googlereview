# Loopy v3 Plan

Loopy v3 is the next branch after the grouped independent-bit family stopped improving downstream predictability enough.

## Why v3 exists

v2 taught us:

- reconstruction can work well on real text
- grouped independent bits can create a meaningful latent
- structured downstream targets help more than plain bitwise targets
- but the current latent is still not a better prediction target than raw patches

So v3 moves to **discrete patch symbols** instead of grouped independent bits.

## Core idea

- raw byte patches in
- encoder produces a latent per patch
- product quantization maps each patch to multiple learned sub-symbols
- decoder reconstructs the original bytes from the quantized latent
- optional predictive loss can encourage the patch symbols to be easier to model downstream

## First goal

Do not chase compression yet.

First prove:

1. the v3 codec trains end to end
2. codebook perplexity stays healthy
3. reconstruction quality becomes readable on small tests
4. the symbol stream is more structured than the v2 grouped-bit stream

## Immediate local test

Run a smoke test on the toy corpus first.

Suggested command:

`python -m loopy.train_symbolic_codec_v3 --data-path loopy/example_corpus.txt --output-dir loopy/runs/v3_product_smoke --epochs 20 --batch-size 4 --max-seq-len 128 --patch-size 1 --embed-dim 128 --latent-dim 128 --encoder-layers 2 --decoder-layers 2 --num-heads 4 --dropout 0.0 --weight-decay 0.0 --num-codebooks 2 --sub-codebook-size 256 --commitment-weight 0.05 --codebook-weight 0.25 --predictive-weight 0.0 --overfit-all`

## Current result

Single-symbol `v3` did not work well enough:

- `patch_size=4` collapsed into blank output
- `patch_size=2` still failed badly
- `patch_size=1` learned only partial text and plateaued around `0.75` byte accuracy

The active branch is now a **product-codebook** `v3`:

- multiple sub-symbols per patch instead of one codebook ID
- first toy smoke result:
  - byte accuracy: `0.7589`
  - codebook perplexity: `6.90`
  - output is partially readable, but still not good enough

## Decision rule

If the product-codebook `v3` cannot beat the partial `patch_size=1` toy result cleanly, stop and fix the architecture before any real-corpus run.

If `v3` starts reconstructing toy text clearly and codebook perplexity stays healthy, move to the real Twitter corpus next.
