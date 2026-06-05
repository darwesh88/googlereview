# V6 Plan

## Goal

Move Loopy away from fixed 2-byte patches.

The v6 hypothesis is:

- fixed patch boundaries are the wrong bottleneck for text
- useful byte latents need variable-length patches
- predictable text should use longer patches
- locally ambiguous/high-detail regions should use shorter patches

This follows the same broad lesson as BLT/MEGABYTE-style byte modeling:

- local byte processing
- global patch-level modeling
- patch boundaries matter

## Status

The first v6 dynamic-patch scaffold is implemented.

Files:

- [dynamic_patching.py](C:/Users/adarw/Desktop/googlereview/loopy/dynamic_patching.py)
- [v6_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v6_config.py)
- [symbolic_codec_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v6.py)
- [train_symbolic_codec_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v6.py)
- [train_patch_prior_v6.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v6.py)
- [experiment_plans/v6_dynamic_tinystories.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/v6_dynamic_tinystories.json)

Local smoke status:

- codec checkpoint saves as `v6_codec.pt`
- grouped prior loads the v6 checkpoint and trains
- dynamic patch encoding produces variable patch lengths

Smoke numbers are not research results.

## What v6 is now

This first version is deliberately simple.

It uses deterministic dynamic patching:

- `boundary`: end patches at spaces/punctuation when allowed
- `space`: end patches at spaces when allowed
- `fixed`: only end at `max_patch_size`, useful as an ablation

The first benchmark should use:

- `patching_mode = boundary`
- `max_patch_size = 8`
- `min_patch_size = 2`
- `max_patches = 64`
- `max_seq_len = 128`

This makes the effective codec capacity depend on actual patch length.

Example:

- `4` codebooks x `64` entries = `24` bits per patch
- if average patch length is `4` bytes, effective capacity is about `6 bpb`
- if average patch length is `6` bytes, effective capacity is about `4 bpb`

So v6 is not directly "6 bpb" like v4.2.

It logs:

- `avg_patch_bytes`
- `empirical_capacity_bpb`
- `code_bits_per_patch`

## Why this before learned entropy patching

A learned patcher adds another moving part.

The first question is simpler:

- do variable patch boundaries help at all under the same downstream prior metric?

If deterministic variable patches do not help, then a learned patcher may still help, but the next design should be more BLT-like end-to-end.

If deterministic variable patches do help, then v6.1 should replace the heuristic boundary rule with a learned entropy/uncertainty patcher.

## First Colab Run

Run this on TinyStories after cells that set up the repo and prepare:

- `loopy/data/real/tinystories_5k.txt`

### 1. Train v6 codec

```python
!python -m loopy.train_symbolic_codec_v6 --data-path loopy/data/real/tinystories_5k.txt --output-dir loopy/runs/auto_clean/v6_dynamic_boundary_clean/codec --epochs 20 --batch-size 8 --max-seq-len 128 --max-patches 64 --max-patch-size 8 --min-patch-size 2 --patching-mode boundary --embed-dim 128 --latent-dim 128 --encoder-layers 2 --decoder-layers 2 --pre-context-layers 1 --post-context-layers 1 --num-heads 4 --dropout 0.0 --weight-decay 0.0 --num-codebooks 4 --sub-codebook-size 64 --assignment-temp 0.5 --commitment-weight 0.05 --codebook-weight 0.25 --usage-weight 0.05 --use-residual-detail --residual-usage-weight 0.005 --residual-gate-bias -2.0
```

### 2. Inspect codec

```python
!cat loopy/runs/auto_clean/v6_dynamic_boundary_clean/codec/best_metrics.json
print()
!cat loopy/runs/auto_clean/v6_dynamic_boundary_clean/codec/sample_reconstruction.txt
```

### 3. Train v6 grouped prior

```python
!python -m loopy.train_patch_prior_v6 --data-path loopy/data/real/tinystories_5k.txt --codec-run-dir loopy/runs/auto_clean/v6_dynamic_boundary_clean/codec --output-dir loopy/runs/auto_clean/v6_dynamic_boundary_clean/prior --epochs 20 --batch-size 16 --hidden-size 128 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.01 --group-embed-dim 16 --batch-encode-size 32
```

### 4. Inspect prior

```python
!cat loopy/runs/auto_clean/v6_dynamic_boundary_clean/prior/best_metrics.json
```

## Decision Rule

Compare the v6 grouped prior `bpb` against:

- raw TinyStories: `1.4022`
- v3 TinyStories: `1.7467`
- v4.2 TinyStories: `1.9336`
- v5 TinyStories: `1.9610`

Interpretation:

- worse than `1.9610`: dynamic boundary MVP failed
- between `1.9336` and `1.9610`: slight signal, but not enough
- below `1.9336`: variable patching is useful
- near/below `1.7467`: v6 becomes the main branch

## If v6 Works

Next steps:

1. add `space` vs `boundary` vs `fixed` ablation
2. tune target capacity through `sub_codebook_size`, `max_patch_size`, and `min_patch_size`
3. add a learned entropy/uncertainty patcher
4. keep raw fixed-patch and v3 baselines in every run

## If v6 Fails

Do not return to v5 micro-tuning.

The next serious pivot would be:

- continuous BLT-like patch latents first
- discrete symbols later
- or a learned variable-length tokenizer closer to GQ-VAE than the current product-VQ design
