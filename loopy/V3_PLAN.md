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
- `8.0 bpb` becomes the first strong efficiency-oriented `v3` point
- `14.0 bpb` is the best pure-fidelity point
- the next work should continue reducing capacity from the new `8.0 bpb` baseline

### Run 3

- `patch_size=2`
- `num_codebooks=2`
- `sub_codebook_size=256`
- `raw_capacity_bpb=8.0`
- byte accuracy: `0.9892`
- codebook perplexity: `246.69`

Sample reconstruction:

- `Customer: delivery slot of 7m. Now 93. and still waiting.... Agent: Sorry Sam, did you receive your order? Cerij`

## Decision rule

If the soft-assignment product-codebook `v3` cannot keep `patch_size=2` fidelity while lowering capacity, stop and fix the architecture before any larger-scale run.

If `v3` starts reconstructing toy text clearly and codebook perplexity stays healthy, move to the real Twitter corpus next.

## Second capacity-reduction results

### Run 4

- `patch_size=2`
- `num_codebooks=2`
- `sub_codebook_size=128`
- `raw_capacity_bpb=7.0`
- byte accuracy: `0.9889`
- codebook perplexity: `124.67`

Sample reconstruction:

- `Customer: delivery slot of 7g. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? CeriM`

Interpretation:

- this is another strong success
- capacity dropped from `8.0` to `7.0`
- fidelity barely moved
- `7.0 bpb` is now the best efficiency-oriented `v3` point

### Run 5

- `patch_size=2`
- `num_codebooks=2`
- `sub_codebook_size=64`
- `raw_capacity_bpb=6.0`
- byte accuracy: `0.9794`
- codebook perplexity: `61.46`

Sample reconstruction:

- `Customer: delivery slot of 70. Now !30 and still waiting.... Agent: Sorry Samm did you receive your order? Cerix`

Interpretation:

- `6.0 bpb` is still viable
- but it is the first clear degradation point in the `v3` sweep
- this is now the cliff test, not the baseline

### Run 6

- same `6.0 bpb` setting
- `20` epochs instead of `10`
- byte accuracy: `0.9846`
- codebook perplexity: `61.95`

Sample reconstruction:

- `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam. did you receive your order? Cerix`

Interpretation:

- longer training helped a lot
- `6.0 bpb` is no longer just a broken cliff point
- it is still weaker than `7.0 bpb`, but now clearly alive
- this justified one lower-capacity probe

### Run 7

- `patch_size=2`
- `num_codebooks=2`
- `sub_codebook_size=32`
- `raw_capacity_bpb=5.0`
- `20` epochs
- byte accuracy: `0.9464`
- codebook perplexity: `31.28`

Sample reconstruction:

- `Customer: delivery slot of 5a. Now 4t? and still wamting.... Agent: S rry Sho, d d you receive your orderi Ceris`

Interpretation:

- `5.0 bpb` is still alive mathematically
- but quality is now clearly below the current bar
- this is too degraded to become the next baseline
- the capacity sweep should stop here for now

## Updated v3 decision

- best pure fidelity:
  - `14.0 bpb`
  - byte accuracy: `0.9988`
- best efficiency-oriented balance:
  - `7.0 bpb`
  - byte accuracy: `0.9889`
- safest low-capacity reference:
  - `8.0 bpb`
  - byte accuracy: `0.9892`
- current cliff test:
  - `6.0 bpb` at `10` epochs
  - byte accuracy: `0.9794`
- recovered low-capacity point:
  - `6.0 bpb` at `20` epochs
  - byte accuracy: `0.9846`
- too-far compression point:
  - `5.0 bpb` at `20` epochs
  - byte accuracy: `0.9464`

Next step:

- keep `7.0 bpb` as the stable baseline
- keep `6.0 bpb` as the new low-capacity frontier
- stop lowering capacity
- test whether `v3` symbols are better downstream targets than raw patches

## Downstream milestone

The first downstream grouped-prior tests changed the picture again.

Observed results:

- raw patch prior:
  - `bpb = 2.9473`
- `v3` grouped at `7.0 bpb`:
  - `bpb = 3.8102`
- `v3` grouped at `6.0 bpb`:
  - `bpb = 3.4601`
- `v3` grouped at `5.0 bpb`:
  - `5` epochs: `3.0444`
  - `10` epochs: `2.9174`
  - `20` epochs: `2.8497`

Interpretation:

- `5.0 bpb` is too degraded as a reconstruction baseline
- but it is now the best downstream `v3` point
- after longer prior training, it has beaten the raw patch baseline

Updated next step:

- keep `7.0 bpb` as the safer reconstruction baseline
- keep `5.0 bpb` as the current best downstream point
- note that longer grouped-prior training helped `6.0 bpb`, but not enough
- run a longer codec training on the `5.0 bpb` checkpoint next
