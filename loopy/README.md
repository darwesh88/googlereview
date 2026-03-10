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

Loopy v2 is the active path.

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

Interpretation:

- the learned representation is strong for modeling
- moderate rate pressure can improve the packed learned bitstream
- but the current codec is still not yet competitive with standard raw-text compression
- the active problem is now the fidelity/compression tradeoff, not whether the architecture works at all

## Best next move from here

Do not rent H100s yet.

Do this next:

1. use [COLAB_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/COLAB_PLAN.md)
2. do a controlled real-corpus rate sweep around the current working point
3. measure packed bitstream size again with [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py)
4. map the fidelity/compression frontier before deciding whether H100 rentals are justified
