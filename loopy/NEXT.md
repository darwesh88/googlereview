# Next

Loopy v2 now has two important real-corpus results:

- a strong modeling baseline with high fidelity
- a first modest packed-bitstream improvement under rate pressure

## Current status

Best fidelity-oriented baseline:

- byte accuracy: `0.9876`
- estimated bpb: `1.5684`
- zlib-compressed learned bitstream bpb: `4.4418`
- zlib-compressed raw text bpb: `3.0611`

Best packed-bitstream result so far:

- run: `v2_twitter_rate_med`
- `rate_weight=0.01`
- byte accuracy: `0.9799`
- estimated bpb: `2.1823`
- zlib-compressed learned bitstream bpb: `4.3861`

Best middle tradeoff so far:

- run: `v2_twitter_rate_003`
- `rate_weight=0.003`
- byte accuracy: `0.9810`
- estimated bpb: `2.2935`
- zlib-compressed learned bitstream bpb: `4.3997`

Failed middle point:

- run: `v2_colab_rate_002`
- `rate_weight=0.002`
- byte accuracy: `0.9803`
- zlib-compressed learned bitstream bpb: `4.5546`
- run: `v2_colab_rate_0025`
- `rate_weight=0.0025`
- byte accuracy: `0.9840`
- zlib-compressed learned bitstream bpb: `4.5508`
- run: `v2_colab_rate_005`
- `rate_weight=0.005`
- byte accuracy: `0.9805`
- zlib-compressed learned bitstream bpb: `4.5231`

## What this means

The architecture clearly works.

The low-rate sweep is now informative enough:

- `0.003` is the best tested compromise
- `0.01` is the best packed-bitstream point
- the points around `0.002` to `0.005` did not beat `0.003`

So the next clean task is no longer another small rate sweep.
It is to attack the next bottleneck directly.

## Immediate next step

Next two serious directions are:

1. improve packing / entropy coding
2. test whether downstream modeling on the learned stream gives a real LLM-training benefit

Detailed execution order is captured in [WORK_ORDER.md](C:/Users/adarw/Desktop/googlereview/loopy/WORK_ORDER.md).

Important refinement:

- the first group-token export path is only a tooling smoke test
- the real downstream test should be patch-level, not 4 expanded group tokens per patch
- [train_patch_prior_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v2.py) is now the active downstream comparison tool

Immediate experiment:

- run `train_patch_prior_v2.py` in both modes:
  - `learned`
  - `raw`
- use Colab GPU
- run longer than the 1-epoch CPU smoke test

Observed 5-epoch result:

- learned patch prior:
  - `bpb = 5.1364`
  - `accuracy = 0.5777`
- raw patch prior:
  - `bpb = 3.6991`
  - `accuracy = 0.3240`

Interpretation:

- the current learned stream is still worse than the raw patch baseline on the core downstream metric (`bpb`)
- this is a negative result for the current codec objective
- scaling this exact setup further is unlikely to be the best next move

## Decision rule

Move forward only if the next step either:

- improves packed learned-bitstream bpb below `4.3997` without large fidelity loss
or
- shows a downstream LM benefit that justifies the representation even if raw compression remains worse than `zlib/gzip`

Current next step:

- redesign the codec objective so the learned patches are optimized for predictability, not only reconstruction

That redesign is now started:

- [train_binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_binary_codec_v2.py) now supports `--predictive-weight`
- the codec now includes a next-patch bit prediction auxiliary loss

Immediate experiment:

- retrain the codec on the real corpus with a light predictive weight
- then rerun the learned-vs-raw patch prior comparison

## Do not do next

- do not move to H100 yet
- do not claim the codec beats standard text compression yet
- do not keep doing tiny nearby rate sweeps unless there is a clear new hypothesis
