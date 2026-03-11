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
- first product-codebook toy smoke result:
  - byte accuracy: `0.7589`
  - codebook perplexity: `6.90`
  - output is partially readable, but still not good enough
- soft assignments plus a usage loss then improved the same toy setup sharply:
  - `patch_size=1`
  - byte accuracy: `0.9265`
  - codebook perplexity: `250.97`
  - output became mostly readable
- but the first `patch_size=2` run is still weak:
  - byte accuracy: `0.5652`
  - codebook perplexity: `169.11`
  - output is not yet good enough
- increasing `patch_size=2` capacity and training longer changed the picture:
  - toy run with `num_codebooks=4`, `40` epochs:
    - byte accuracy: `0.9746`
    - output became exact on the toy sentence
  - short real-corpus smoke with the same high-capacity setup:
    - byte accuracy: `0.9892`
    - output stayed very close, with only tiny detail errors

## First capacity-reduction results

Two Colab GPU runs tested whether `v3` can lower capacity while keeping the new fidelity.

### Run 1

- `patch_size=2`
- `num_codebooks=3`
- `sub_codebook_size=256`
- `raw_capacity_bpb=12.0`
- byte accuracy: `0.9973`
- codebook perplexity: `252.72`

Sample reconstruction:

- `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri2`

### Run 2

- `patch_size=2`
- `num_codebooks=4`
- `sub_codebook_size=128`
- `raw_capacity_bpb=14.0`
- byte accuracy: `0.9988`
- codebook perplexity: `127.21`

Sample reconstruction:

- `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri3`

Interpretation:

- both runs are successful
- `12.0 bpb` is currently the best efficiency-oriented `v3` point
- `14.0 bpb` is the best pure-fidelity point
- the next work should continue reducing capacity from the new `12.0 bpb` baseline

## Decision rule

If the soft-assignment product-codebook `v3` cannot keep `patch_size=2` fidelity while lowering capacity, stop and fix the architecture before any larger-scale run.

If `v3` starts reconstructing toy text clearly and codebook perplexity stays healthy, move to the real Twitter corpus next.
