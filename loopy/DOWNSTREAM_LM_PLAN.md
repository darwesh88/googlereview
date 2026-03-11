# Downstream LM Plan

This is the next branch after the local rate sweep.

## Goal

Test whether the learned Loopy v2 stream is easier to model than a simple raw-byte token stream.

This is not trying to prove better compression.
It is trying to prove downstream usefulness for language-model training.

## Export step

Use [export_stream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/export_stream_v2.py) to write:

- `group_stream.txt`
- `raw_byte_stream.txt`

Both are emitted as angle-bracket tokens so [train_token_lm.py](C:/Users/adarw/Desktop/googlereview/loopy/train_token_lm.py) can consume them directly.

Example grouped stream tokens:

- `<p> <g0:12> <g1:7> <g2:55> <g3:2>`

Example raw-byte stream tokens:

- `<p> <b:67> <b:117> <b:115> <b:116>`

## First experiment

1. export the streams from the best fidelity baseline checkpoint
2. train the same tiny LM on:
   - grouped stream
   - raw-byte stream
3. compare:
   - validation loss
   - perplexity
   - training speed
   - vocab size

## Decision rule

Move forward with the downstream-LM branch only if the grouped stream:

- trains at least as easily as the raw-byte stream
or
- reaches clearly better validation quality per unit compute

If it does not, return to the packing/entropy branch.

## Refinement

The exported group-token LM is only a tooling smoke test.

It expands each patch back into four tokens, which removes the sequence-length advantage that matters most for Loopy v2.

So the stronger downstream test is now:

- use [train_patch_prior_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v2.py)
- compare:
  - `--mode learned`
  - `--mode raw`
- evaluate both in **bits per byte** on the same patch-level prediction task

## Smoke result

One-epoch CPU smoke test:

- learned patch prior:
  - val bpb: `5.4109`
  - val accuracy: `0.5550`
  - epoch seconds: `10.86`
- raw patch prior:
  - val bpb: `4.3833`
  - val accuracy: `0.2246`
  - epoch seconds: `16.57`

Interpretation:

- the patch-level downstream path works
- after 1 epoch, the raw patch baseline is still better in bpb
- this is only a smoke test, not the real comparison
- the next proper experiment should be a longer learned-vs-raw patch prior run on Colab GPU
