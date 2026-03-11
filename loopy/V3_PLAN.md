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
- vector quantizer maps each patch to a single learned codebook symbol
- decoder reconstructs the original bytes from the quantized latent
- optional predictive loss encourages the patch symbols to be easier to model downstream

## First goal

Do not chase compression yet.

First prove:

1. the v3 codec trains end to end
2. codebook perplexity stays healthy
3. reconstruction quality is competitive with v2 on small tests
4. the exported symbol stream is easier to model than the v2 bitwise stream

## Immediate local test

Run a smoke test on the toy corpus first.

Suggested command:

`python -m loopy.train_symbolic_codec_v3 --data-path loopy/example_corpus.txt --output-dir loopy/runs/v3_smoke --epochs 5 --batch-size 4 --max-seq-len 128 --patch-size 4 --embed-dim 96 --latent-dim 96 --encoder-layers 2 --decoder-layers 2 --num-heads 4 --codebook-size 256 --commitment-weight 0.25 --codebook-weight 1.0 --predictive-weight 0.01 --overfit-all`

## Decision rule

If v3 cannot reconstruct the toy corpus cleanly, stop and fix the codebook path before any real-corpus run.

If v3 reconstructs well and codebook perplexity is healthy, move to the real Twitter corpus next.
