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
- [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py): prototype true bitstream measurement
- [v2_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v2_config.py): v2 config
- [binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/binary_codec_v2.py): grouped binary codec model
- [train_binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_binary_codec_v2.py): training loop for the codec

## Archived but important v1 files

- [RESEARCH_LOG.md](C:/Users/adarw/Desktop/googlereview/loopy/RESEARCH_LOG.md)
- [AMBIGUITY_RESEARCH.md](C:/Users/adarw/Desktop/googlereview/loopy/AMBIGUITY_RESEARCH.md)
- [DECODER_BASELINE.md](C:/Users/adarw/Desktop/googlereview/loopy/DECODER_BASELINE.md)
- [REAL_CORPUS_RUNBOOK.md](C:/Users/adarw/Desktop/googlereview/loopy/REAL_CORPUS_RUNBOOK.md)

## Current best result in v2

The best current CPU baseline is the 20-epoch real-corpus run on `twitter_support_5k`.

- byte accuracy: `0.9849`
- estimated bpb: `1.8326`
- raw capacity bpb: `6.0`
- reconstruction stayed highly faithful on real noisy text

Interpretation:

- the binary codec architecture is stable on real text
- fragile details like names and numbers are still the main error mode
- CPU validation is strong enough to move to medium-scale GPU validation

## Colab entrypoint

The fastest way to start the GPU phase is the ready notebook:

- [Loopy_v2_Colab_Baseline.ipynb](C:/Users/adarw/Desktop/googlereview/loopy/Loopy_v2_Colab_Baseline.ipynb)

GitHub Colab URL:

`https://colab.research.google.com/github/darwesh88/googlereview/blob/main/loopy/Loopy_v2_Colab_Baseline.ipynb`
## Best next move from here

Do not rent H100s yet.

Do this next:

1. use [COLAB_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/COLAB_PLAN.md)
2. reproduce the CPU baseline on Colab GPU
3. run a small `rate_weight` sweep
4. measure packed bitstream size with [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py)
5. only then decide whether H100 rentals are justified

