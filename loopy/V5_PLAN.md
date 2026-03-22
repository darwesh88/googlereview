# V5 Plan

## Goal

Build the first Loopy branch that trains the codec **directly for downstream predictability**, not only for reconstruction.

The target is no longer:

- "can we reconstruct well?"

The target is:

- "can the learned grouped symbol stream become easier for a causal prior to model than raw patches?"

## Why `v5` is needed

The current branches have split:

- `v3` is best downstream
- `v4.2` is best reconstruction

The clean TinyStories benchmark made the problem clearer:

- raw patch prior: `1.4022 bpb`
- `v3`: `1.7467 bpb`
- `v4.2`: `1.9336 bpb`
- `v4.2 + masked predictive`: `2.0513 bpb`

So:

- cleaner data did not fix the gap
- masked predictive did not solve the real downstream objective
- the next bottleneck is architectural alignment

## Core idea

`v5` should keep the best codec structure from `v4.2`, but replace the current weak predictive auxiliary path with a **true causal prior-aware loss**.

Simple version:

- `v4.2` learns to reconstruct bytes well
- `v5` should learn symbol assignments that a causal grouped prior can predict well

## What stays from `v4.2`

- local patch encoder
- pre-quantization cross-patch context
- product-codebook quantizer
- post-quantization cross-patch context
- local byte decoder
- residual-detail side channel
- current grouped symbol format

This is important.

`v5` is **not** a total rewrite.
It is a new training objective on top of the best `v4.2` structure.

## What changes in `v5`

### 1. Expose quantizer assignment probabilities

The quantizer should return:

- hard `symbol_ids`
- quantized embeddings
- soft assignment probabilities `q_tk` for each patch position `t` and codebook `k`

Right now the model only uses hard targets or masked targets.
`v5` needs the soft assignment distribution.

### 2. Add a causal grouped prior head inside codec training

Add a new module:

- `CausalGroupedPriorHead`

Input:

- previous patch symbols only

Output:

- per-codebook logits for the next patch symbol

Important constraint:

- this prior must be **strictly causal**
- no bidirectional leakage
- no masked objective using future context

Implementation should start simple:

- grouped symbol embeddings
- shift-right input
- `GRU` or explicitly causal transformer
- one prediction head per codebook

### 3. Match current symbol assignments to the causal prior

The new loss should align codec symbol assignments with what the causal prior can model.

Recommended first loss:

- `prior_match_loss = mean KL(q_t || p_t)`

Where:

- `q_t` = current quantizer soft assignment distribution
- `p_t` = prior-predicted distribution from previous symbols only

Why this is better than the current masked predictive loss:

- it matches the real downstream objective
- it does not rely on future context
- it pushes the codec toward symbol choices that are easier for the next model to predict

### 4. Keep reconstruction and detail losses

`v5` is not allowed to win downstream by destroying reconstruction.

So keep:

- reconstruction loss
- commitment loss
- codebook loss
- usage loss
- residual usage loss

And add:

- `prior_match_loss`

## Proposed `v5` loss

First implementation:

```text
total_loss =
    recon_loss
  + commitment_loss
  + codebook_loss
  + usage_loss
  + residual_usage_loss
  + prior_weight * prior_match_loss
```

Optional logging metrics:

- hard next-symbol CE under the internal prior head
- prior-head `bpb` estimate
- codebook perplexity
- residual gate usage

## First architecture shape

```text
bytes
  -> local patch encoder
  -> pre-context transformer
  -> product quantizer
  -> post-context transformer
  -> byte decoder + residual detail head

hard / soft symbols
  -> shift-right grouped symbol embeddings
  -> causal grouped prior head
  -> prior_match_loss against current quantizer assignments
```

## Why this is the right next hypothesis

The current downstream prior is trained **after** the codec.

That means the codec is free to choose symbol assignments that are:

- good for reconstruction
- but awkward for the next model

`v5` changes that.

It makes downstream predictability part of codec training itself.

That is the cleanest next hypothesis because it directly attacks the exact metric raw is still winning on.

## File plan

New files:

- [v5_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v5_config.py)
- [symbolic_codec_v5.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v5.py)
- [train_symbolic_codec_v5.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v5.py)
- [train_patch_prior_v5.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v5.py)

Expected reuse:

- `v4.2` encoder / decoder / residual structure
- `v3` quantizer core
- grouped prior utilities from `train_patch_prior_v2.py`

## First implementation order

### Phase 1. Minimal `v5`

1. Copy `v4` config into `v5_config.py`
2. Add:
   - `prior_weight`
   - `prior_hidden_size`
   - `prior_num_layers`
   - `prior_dropout`
3. Copy `symbolic_codec_v4.py` into `symbolic_codec_v5.py`
4. Add:
   - quantizer soft assignment outputs
   - grouped symbol embedding path
   - causal prior head
   - `prior_match_loss`
5. Copy `train_symbolic_codec_v4.py` into `train_symbolic_codec_v5.py`
6. Save:
   - `prior_match_loss`
   - optional internal prior `bpb` metric

### Phase 2. Downstream evaluation

1. Add `train_patch_prior_v5.py`
2. Keep the same grouped-prior evaluation protocol
3. Compare against:
   - TinyStories raw
   - TinyStories `v3`
   - TinyStories `v4.2`
   - Twitter raw
   - Twitter `v3`
   - Twitter `v4.2`

## First run order

### Run 1. Local toy smoke

Purpose:

- ensure `prior_match_loss` is nonzero
- ensure the branch trains end to end

Success conditions:

- `prior_match_loss > 0`
- codebook perplexity healthy
- reconstruction not collapsed

### Run 2. TinyStories clean benchmark

Purpose:

- first real read on whether prior-aware training helps the exact downstream metric

Decision threshold:

- must improve materially over current clean `v4.2` downstream `1.9336`

### Run 3. Twitter support robustness check

Purpose:

- verify the gain is not limited to clean story text

## Success gates

`v5` is worth continuing only if it shows at least one of these:

1. TinyStories downstream `bpb` moves materially below current `v3` clean result `1.7467`
2. Twitter downstream `bpb` moves materially below current `v4.2` best `3.1301`
3. Reconstruction stays near current `v4.2` levels while downstream gap clearly shrinks

## Failure rule

If `v5` still fails to narrow the clean TinyStories gap meaningfully, then:

- the current grouped-symbol family may not be enough
- next work should move toward a stronger latent supervision or a different symbol design

## Not the next step

Do not do these first:

- more `v42` parameter sweeps
- more residual gate tuning
- more masked-predictive probability sweeps
- more data-only experiments before `v5`

The clean benchmark already answered the data question enough.
