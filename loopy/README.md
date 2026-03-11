# Loopy

Loopy is now a two-stage research repo.

## v1

Loopy v1 tested a symbolic middleware layer:

- rewrite selected concepts into reversible IDs like `<n2>`
- train on rewritten text
- decode back to normal English

v1 result:

- worked on narrow synthetic support corpora
- did not beat plain text on the first broad real Twitter corpus

Conclusion:

- useful research probe
- not the active architecture direction anymore

## v2

Loopy v2 is the completed active branch for grouped-bit latents.

The new goal is to build a **semantic binary codec**:

- raw bytes in
- learned grouped binary patches internally
- exact reconstruction back to text
- later, train models on the binary stream itself

This is the boundary-pushing direction.

## Why v2 exists

v1 taught us something important:

- shallow word-level normalization is too weak for broad real text
- if this idea is going to matter, the internal representation has to be deeper and more mathematical

So v2 focuses on:

- byte-level input
- grouped binary quantization
- compression-aware training
- exact recoverability

## Current active files

- [V2_ARCHITECTURE.md](C:/Users/adarw/Desktop/googlereview/loopy/V2_ARCHITECTURE.md): the v2 architecture
- [V2_LOCAL_RUNBOOK.md](C:/Users/adarw/Desktop/googlereview/loopy/V2_LOCAL_RUNBOOK.md): local-machine execution path
- [COLAB_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/COLAB_PLAN.md): the Colab scale-up plan
- [Loopy_v2_Colab_Baseline.ipynb](C:/Users/adarw/Desktop/googlereview/loopy/Loopy_v2_Colab_Baseline.ipynb): ready Colab entrypoint
- [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py): prototype true bitstream measurement
- [v2_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v2_config.py): v2 config
- [binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/binary_codec_v2.py): grouped binary codec model
- [train_binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_binary_codec_v2.py): training loop for the codec
- [V3_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/V3_PLAN.md): product-codebook branch plan
- [v3_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v3_config.py): v3 config
- [symbolic_codec_v3.py](C:/Users/adarw/Desktop/googlereview/loopy/symbolic_codec_v3.py): product-codebook codec
- [train_symbolic_codec_v3.py](C:/Users/adarw/Desktop/googlereview/loopy/train_symbolic_codec_v3.py): v3 training loop

## Current best understanding in v2

The strongest real-corpus modeling baseline so far is now:

- byte accuracy: `0.9876`
- estimated bpb: `1.5684`
- reconstruction stayed highly faithful on real noisy text

The strongest packed-bitstream result so far came from adding moderate rate pressure locally:

- `rate_weight=0.01`
- byte accuracy: `0.9799`
- zlib-compressed learned bitstream bpb: `4.3861`
- zlib-compressed raw text bpb: `3.0611`

The best middle tradeoff tested so far is:

- `rate_weight=0.003`
- byte accuracy: `0.9810`
- zlib-compressed learned bitstream bpb: `4.3997`

And the `rate_weight=0.005` Colab GPU run was clearly worse than the baseline:

- `rate_weight=0.002`
  - byte accuracy: `0.9803`
  - zlib-compressed learned bitstream bpb: `4.5546`
- `rate_weight=0.0025`
  - byte accuracy: `0.9840`
  - zlib-compressed learned bitstream bpb: `4.5508`
- `rate_weight=0.005`
  - byte accuracy: `0.9805`
  - zlib-compressed learned bitstream bpb: `4.5231`

Interpretation:

- the learned representation is strong for modeling
- moderate rate pressure can improve the packed learned bitstream
- `0.003` is currently the best compromise among the tested rate points
- `0.01` is still the best packed-bitstream point so far
- the nearby `0.002`, `0.0025`, and `0.005` points are all worse than `0.003`
- but the current codec is still not yet competitive with standard raw-text compression
- the active problem is no longer just fidelity/compression
- the learned stream is also not yet beating a raw patch baseline in downstream patch prediction
- a light predictive auxiliary loss helped only slightly downstream:
  - learned patch prior `bpb` moved from `5.1364` to `5.0593`
  - raw patch prior is still much better at `3.6991`
- a grouped-symbol downstream target helped more than bitwise learned prediction:
  - grouped patch prior `bpb = 4.9839`
  - still worse than raw `3.6991`
- grouped symbols on the predictive codec did not improve further:
  - grouped predictive patch prior `bpb = 5.1005`

## Best next move from here

Do not rent H100s yet.

Do this next:

1. stop the simple local rate sweep around the current working point
2. stop the grouped packing branch
3. treat the current downstream patch-prior result as a negative result for the existing codec objective
4. treat the first predictive auxiliary-loss test as only a weak partial improvement
5. treat grouped-symbol priors as evidence that structured targets help, but not enough inside this codec family
6. move to a stronger latent redesign so the learned stream is easier to predict, not just easier to reconstruct
7. `v3` codebook scaffold is now started
8. independent-bit `v2` is now good enough as a reference point, but not the main architecture to scale further

## Current active branch

`v3` is now the active architecture branch.

Current understanding:

- single-symbol `v3` was too weak
- product-codebook `v3` is the correct next attempt
- the first product-codebook toy smoke test is partially readable but still not strong enough
- the next task is a clean toy sanity run on the committed product-codebook path, before any real-corpus `v3` run
