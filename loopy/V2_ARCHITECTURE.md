# Loopy v2 Architecture

Loopy v2 is a pivot.

v1 tried to normalize a small set of words and phrases before language-model training.
That helped on narrow synthetic data, but it did not win on the first broad real corpus.

v2 changes the goal.

## New thesis

Do not replace a few words with concept tags.
Build a **semantic binary codec** that compresses raw text into structured binary patches, then train models on that binary stream.

The important claim is not:

- new token IDs

The important claim is:

- a better internal binary representation of text
- shorter effective sequences
- exact recoverability
- better predictability than raw text or plain tokenization

## High-level design

### 1. Raw bytes in

- input starts as UTF-8 bytes
- no normal tokenizer is the main representation
- local smoke tests use fixed byte patches first
- later versions can move to dynamic patching

### 2. Patch encoder

- split text into byte patches
- encode each patch into a latent vector
- local version uses a small transformer over bytes within each patch

### 3. Grouped binary quantizer

- convert each patch latent into grouped bits
- bits are learned, not hand-assigned
- groups are meant to later support different roles:
  - route bits
  - semantic bits
  - state/action bits
  - detail bits

The current local scaffold uses straight-through binary quantization.

Useful adjacent work:

- [BLT](https://arxiv.org/abs/2412.09871)
- [Large Concept Models](https://arxiv.org/abs/2412.08821)
- [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
- [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)
- [Binary Spherical Quantization](https://arxiv.org/abs/2406.07548)

### 4. Exact decoder

- decode the bit representation back into the original bytes
- exact reconstruction is required for the codec path to matter
- if exact reconstruction fails, v2 is not ready

### 5. Rate-aware objective

The training target is not only reconstruction.
It is:

- reconstruction quality
- bit-rate / entropy pressure
- stable bit usage

The local scaffold starts with reconstruction first and a light rate proxy.

### 6. Prior model later

Once the codec itself works, train a model over the binary patch stream.
That is the stage where we can compare:

- byte baseline
- token baseline
- binary-codec prior model

Only after that should we think about Mercury-style refinement on the binary stream.

## What makes v2 different from v1

v1:

- human-defined concept replacements
- shallow normalization
- useful as a probe

v2:

- raw bytes in
- learned grouped bits
- mathematical compression pressure
- exact reconstruction requirement
- possible path to a genuinely new representation layer

## Local-first strategy

Do not rent H100s yet.

First validate on this machine:

1. the codec trains end to end
2. exact reconstruction gets visibly good on toy data
3. bit density and estimated bits-per-byte are stable
4. overfit tests pass on small corpora

Only then move to GPU rentals.

## Success criteria for local stage

- end-to-end training works
- toy corpus overfit works
- reconstruction becomes close to exact
- binary groups do not collapse immediately
- metrics and saved artifacts are stable enough to compare runs

## Failure criteria for local stage

- reconstruction stays poor even on tiny corpora
- bit groups collapse into trivial patterns
- the system is slower and less stable than a plain byte autoencoder
- there is no sign of useful compression pressure

## What not to do yet

- do not try to prove novelty with the current v1 middleware
- do not jump to large real corpora immediately
- do not spend H100 money before local overfit and reconstruction checks pass
- do not bring Mercury-style generation in before the codec works
