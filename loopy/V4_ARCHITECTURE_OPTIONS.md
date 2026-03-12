# V4 Architecture Options

The current `v3` result suggests a split:

- `7.0 bpb` is better for reconstruction
- `5.0 bpb` is better for downstream prediction

That means one latent stream is trying to do two jobs:

1. carry **semantic structure**
2. carry **fragile exact detail** like names, digits, punctuation, spelling

## Current architectural bottleneck

In `v3`, each patch is encoded locally and decoded locally.

- the patch encoder sees bytes **inside one patch**
- the quantizer acts on **one patch latent**
- the patch decoder reconstructs **one patch at a time**

So at lower capacity, each patch has to describe itself mostly alone.

That is likely the real reason `5.0 bpb` helps downstream but hurts exact reconstruction:

- low-capacity symbols are becoming more predictable
- but the decoder cannot use nearby patches to recover missing detail

## Best next architecture idea

Add **cross-patch context** around the quantizer.

### Proposed v4

```text
[local patch encoder] -> [cross-patch transformer] -> [quantizer] -> [cross-patch transformer] -> [local patch decoder]
```

Meaning:

- local patch encoder:
  - still models bytes inside each patch
- encoder-side cross-patch transformer:
  - lets patches see neighbors before quantization
  - helps decide what to keep vs. what can be inferred later
- decoder-side cross-patch transformer:
  - lets quantized patches help reconstruct each other
  - should especially help names, numbers, punctuation, and local spelling detail

This is the single most likely `v4` change to break the current wall.

## Why this is better than "just train longer"

Current evidence says:

- lower-capacity `v3` gets better downstream
- longer training at `5.0 bpb` does not fix reconstruction
- so the problem is likely **missing context**, not undertraining

## Residual detail channel

A separate detail-residual channel is still a valid follow-up idea.

But it should be treated as:

- **v4.2**

not the first change.

Why:

- cross-patch context is the simpler and more direct explanation for the current failure mode
- if context alone fixes much of the reconstruction gap, the architecture stays cleaner

## First recommended v4 experiment

At `patch_size=2` and the current best downstream setting:

1. keep the current `5.0 bpb` symbol capacity
2. insert a small cross-patch transformer before quantization
3. insert a small cross-patch transformer after quantization
4. keep the existing losses at first
5. compare against current `v3` on:
   - reconstruction quality
   - downstream grouped-prior `bpb`

## Decision rule

- if reconstruction improves materially while downstream `bpb` stays near the current `5.0 bpb` win, this is the right `v4`
- if downstream predictability collapses, then the added context is interfering too much
- if reconstruction barely changes, then the next branch is a small residual-detail side channel
