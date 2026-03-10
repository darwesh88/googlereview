# Loopy Research Plan

## Active thesis

Build a semantic binary codec for language:

- raw UTF-8 bytes in
- learned grouped binary patches internally
- exact reconstruction out
- later, train a prior model on the binary patch stream

## Why this is the active path

Loopy v1 showed that shallow symbolic normalization can help in narrow synthetic domains, but that result did not hold on the first broad real corpus.

So the new hypothesis is:

- the internal representation has to be deeper than word-level concept replacement
- the binary structure itself must matter

## Phase 1: local codec smoke test

- train the v2 codec on `example_corpus.txt`
- use fixed-size patches first
- validate end-to-end training and saved artifacts
- inspect exact reconstruction quality

## Phase 2: local overfit test

- run `--overfit-all` on the toy corpus
- verify that reconstruction becomes near-exact
- check that grouped bits do not collapse immediately

## Phase 3: first real local test

- run the codec on `twitter_support_5k.txt`
- validate stability, reconstruction behavior, and bit metrics on real noisy text
- extend the baseline to a 20-epoch reference run

## Phase 4: rate-aware local training

- turn on small rate pressure
- compare reconstruction vs estimated bits-per-byte
- confirm that mild compression pressure does not destabilize the codec

## Phase 5: Google Colab validation

- reproduce the best CPU baseline on a GPU
- run a small `rate_weight` sweep
- measure actual packed bitstream size with [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py)
- test one capacity ablation without changing too many variables at once

## Phase 6: prior model over binary stream

- after the codec is stable on Colab
- train a small model over the binary patch sequence
- compare against byte and token baselines

## Phase 7: H100 expansion

Only after Colab-scale tests are stable:

- move larger codec runs to H100 rentals
- test bigger latent sizes and longer sequences
- explore dynamic patching and stronger entropy models

## What we are not doing yet

- more Loopy v1 lexicon tuning
- Mercury-style generation
- low-bit weight experiments
- multi-node work
- large H100 campaigns before Colab validation and bitstream measurement
