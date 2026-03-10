# Breakpoint

Loopy has pivoted.

## Current state

- v1 symbolic middleware is archived as a completed research branch
- v2 semantic binary codec is the active direction
- CPU validation is complete enough
- Colab validation has started
- the first Colab GPU baseline reproduced high fidelity and large speed gains
- but the first measured packed bitstream is still worse than standard raw-text compression

## What v2 is currently proving

- bytes in
- grouped binary patches internally
- exact or near-exact reconstruction on real noisy text
- strong learned predictability of the latent bits

## Most recent result

The first Colab baseline showed:

- byte accuracy: `0.9861`
- estimated bpb: `1.6132`
- avg epoch seconds: `4.66`
- zlib-compressed learned bitstream bpb: `4.5060`
- zlib-compressed raw text bpb: `3.0611`

Interpretation:

- Loopy v2 is strong as a learned representation and modeling direction
- Loopy v2 is not yet competitive as a practical compressor with the current hard-bit packing path
- the next work is to see whether small rate pressure improves the real packed bitstream result

## Next resume step

Run the Colab `rate_weight=0.001` comparison and measure its packed bitstream size.
