# V4 Architecture Options

The current `v3` result suggests a split:

- `7.0 bpb` is better for reconstruction
- `5.0 bpb` is better for downstream prediction

That means one latent stream is doing two jobs at once:

1. carry **semantic structure**
2. carry **fragile exact detail** like names, digits, punctuation, spelling

That is likely the wrong architecture.

## Best next architecture idea

Use **two channels** instead of one.

### Channel A: semantic symbols

- low-capacity patch symbols
- optimized for predictability
- this is the part we want the next model to learn on

### Channel B: detail residual

- tiny side channel for fragile surface details
- names
- numbers
- punctuation
- rare character-level corrections

Then reconstruct text from:

- semantic symbol stream
- plus small residual detail stream

## Why this is the best next step

Current evidence says:

- the lower-capacity stream is becoming better downstream
- but exact text quality breaks on small details
- longer training does not fix that

So the most likely missing piece is not "more training".
It is a separate path for exact surface recovery.

## Simple mental model

Think of it like:

- **main idea channel**
- **fine-detail correction channel**

Instead of forcing one code stream to carry both.

## First recommended v4 experiment

At `patch_size=2`:

1. keep the current `5.0 bpb` symbol stream as the semantic channel
2. add a small residual head that predicts:
   - byte-difference mask
   - corrected bytes only where needed
3. keep reconstruction and usage losses
4. compare:
   - reconstruction quality
   - downstream grouped-prior `bpb`

## Decision rule

- if reconstruction improves while downstream `bpb` stays near the current `5.0 bpb` win, this is the right branch
- if downstream predictability collapses, then the residual path is interfering too much
