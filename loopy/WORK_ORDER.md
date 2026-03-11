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

The next branch should now be:

1. **downstream-aware codec redesign first**
2. **retest downstream patch prediction**

Do not jump to H100 rentals yet.

## Why this is the right order

The current research already answers the local rate question well enough.

- `0.003` is the best compromise
- `0.01` is the best packed-bitstream point
- `0.002`, `0.0025`, and `0.005` all lost

So the next bottleneck is not rate tuning.
And it is not grouped packing either.

The deeper issue is:

- the codec learns a stream that reconstructs well
- but that stream is not yet a better prediction target than raw patches

## Immediate work package

### Work package A: downstream-aware codec redesign

Goal:

- make the learned patch stream easier to predict downstream without destroying reconstruction

Tasks:

1. add an auxiliary next-patch predictability objective during codec training
2. keep the current reconstruction path intact
3. retrain on the real corpus baseline setting
4. rerun the patch-prior comparison against the raw patch baseline

Success condition:

- learned patch prior `bpb` moves materially toward or below the raw patch baseline

### Work package B: downstream LM usefulness retest

Goal:

- verify whether the redesigned codec actually produces a better downstream patch stream

Tasks:

1. train the codec with the new downstream-aware objective
2. rerun [train_patch_prior_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v2.py) in `learned` and `raw` modes
3. compare `bpb`, `accuracy`, and training speed

## Exact next coding task

Start with **Work package A**.

Concrete first implementation:

- add a lightweight predictive auxiliary loss to [train_binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_binary_codec_v2.py)
- simplest form:
  - predict next patch bits from current patch latents
  - weight the new loss lightly so reconstruction remains stable

Reason:

- this attacks the newly exposed bottleneck directly
- the current codec is reconstructive but not predictive enough
- we need the learned stream itself to become a better modeling target

Status:

- predictive auxiliary loss is now implemented
- the codec training path now supports `--predictive-weight`
- smoke run passed and logs `predictive_loss`

Immediate experiment:

1. train a new codec run on the real corpus with:
   - `predictive_weight=0.01`
   - current best baseline settings otherwise unchanged
2. rerun the patch prior comparison on that new checkpoint
3. compare learned-patch prior `bpb` against the previous learned result of `5.1364`

Status:

- grouped packing was tested and rejected
- the patch-level downstream prior path is now implemented
- the first 5-epoch learned-vs-raw patch prior comparison was negative for the learned stream

Observed downstream result:

- learned patch prior: `5.1364 bpb`
- raw patch prior: `3.6991 bpb`

Interpretation:

- the current codec objective is not producing a better downstream stream than raw patches
- this is the bottleneck to fix next

Observed predictive-branch result:

- predictive codec run:
  - byte accuracy: `0.9814`
  - estimated bpb: `2.5241`
  - predictive loss: `0.00594`
- learned patch prior on predictive checkpoint:
  - `bpb = 5.0593`
  - `accuracy = 0.5711`

Interpretation:

- the predictive auxiliary loss helped only slightly:
  - learned `bpb` improved from `5.1364` to `5.0593`
- this is not enough to close the large gap to raw patches (`3.6991`)
- the next redesign should therefore be stronger than a light auxiliary loss

Follow-up now implemented:

- [train_patch_prior_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v2.py) now also supports `--mode grouped`
- grouped mode predicts grouped categorical patch symbols instead of independent bits
- 1-epoch smoke result on the baseline codec:
  - grouped patch prior: `5.1985 bpb`

Interpretation:

- grouped-symbol modeling is the next clean structured-latent test
- if it beats the bitwise learned prior materially, then the problem is partly in the downstream target representation
- if it still loses badly to raw patches, then the next move is a true codebook-style codec redesign

## What not to do next

- do not do another tiny `rate_weight` sweep
- do not move to H100 yet
- do not frame the project as "beating gzip"
- do not return to Loopy v1
- do not add Mercury-style generation yet

## Decision after the next work package

The predictive auxiliary loss did not improve downstream learned-patch `bpb` materially enough.

So the next branch should be:

- first run a real grouped-symbol downstream comparison
- then, if needed, consider a stronger latent redesign such as patch symbols / codebooks instead of independent bit prediction


