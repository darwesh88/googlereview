# Loopy Next Work Order

## Current state

Loopy v2 is now validated enough to stop asking whether the architecture works.

What is already true:

- the codec reconstructs real noisy text with high fidelity
- the learned binary stream is useful for modeling
- mild to moderate rate pressure creates a real compression/fidelity tradeoff
- the best tested compromise is `rate_weight=0.003`
- the best packed-bitstream result so far is `rate_weight=0.01`
- the current packed bitstream is still worse than raw-text `zlib/gzip`

This means the next work should not be another tiny rate sweep.

## Main decision

The next branch should be:

1. **packing / entropy coding improvement first**
2. **downstream LM usefulness second**

Do not jump to H100 rentals yet.

## Why this is the right order

The current research already answers the local rate question well enough.

- `0.003` is the best compromise
- `0.01` is the best packed-bitstream point
- `0.002`, `0.0025`, and `0.005` all lost

So the next bottleneck is not rate tuning.
It is that model-side predictability is not yet turning into a strong real stored bitstream.

## Immediate work package

### Work package A: better packing

Goal:

- improve real packed learned-bitstream size without changing the codec architecture too much

Tasks:

1. inspect the current bitstream measurement path in [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py)
2. separate patch-group streams instead of packing one flat bitstream
3. measure whether per-group compression beats the current flat packing
4. test simple delta / run-length style preprocessing only if it is mathematically justified by the bit patterns

Success condition:

- packed learned-bitstream `zlib` bpb drops below the current `0.003` result of `4.3997`
- fidelity stays close to the current `0.003` run

### Work package B: downstream LM usefulness

Goal:

- test whether the learned binary stream helps later modeling even if raw compression is not state of the art

Tasks:

1. export the learned patch/binary stream for `twitter_support_5k`
2. train a tiny prior model on that stream
3. compare against a byte-level or token-level baseline of similar scale
4. judge whether the latent stream is easier to model than raw text

Success condition:

- the downstream model on the learned stream is clearly easier to train or reaches better predictive quality per unit compute

## Exact next coding task

Start with **Work package A**.

Concrete first implementation:

- add grouped bitstream measurement to [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py)
- report:
  - flat packed bitstream size
  - per-group packed bitstream size
  - per-group `zlib/gzip` results
  - whether groupwise packing improves over flat packing

Reason:

- this is the smallest next change that directly attacks the current bottleneck
- it does not require retraining first
- it tells us whether the packing path is weak because we are mixing unlike bit groups together

Status:

- grouped bitstream measurement has now been added and tested
- grouped packing was worse than flat packing on the current best checkpoints
- so Work package A did not unlock the next gain
- a first downstream export path now exists via [export_stream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/export_stream_v2.py)
- the simple group-token LM smoke test is runnable, but it expands each patch back into four tokens, so it does not preserve the sequence-length advantage we really want to test

## Updated immediate task

Move now to **Work package B**.

Concrete first implementation:

- use [export_stream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/export_stream_v2.py)
- export:
  - `group_stream.txt`
  - `raw_byte_stream.txt`
- train the same tiny LM on both streams with [train_token_lm.py](C:/Users/adarw/Desktop/googlereview/loopy/train_token_lm.py)
- compare validation quality and training speed

Current interpretation:

- this is a useful smoke test only
- it validates the tooling path
- it does not yet answer the real downstream question

## Refined next task

Build a **patch-level prior model**, not just a token LM over expanded group tokens.

Reason:

- the group-token export uses 4 tokens for each 4-byte patch
- that removes the potential sequence reduction benefit of the learned patch representation
- the correct downstream test should operate on one learned code step per patch

## What not to do next

- do not do another tiny `rate_weight` sweep
- do not move to H100 yet
- do not frame the project as "beating gzip"
- do not return to Loopy v1
- do not add Mercury-style generation yet

## Decision after the next work package

If better packing materially improves stored bitstream size:

- continue improving the codec/packing path on Colab

If better packing does not help much:

- move immediately to downstream LM usefulness

That is the clean next fork.


