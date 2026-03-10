# Breakpoint

Loopy has pivoted.

## Current state

- v1 symbolic middleware is archived as a completed research branch
- v2 semantic binary codec is the active direction
- CPU validation is complete enough
- Colab validation has started
- strong real-corpus modeling quality is now reproduced across local and Colab runs
- moderate rate pressure gives the first real packed-bitstream improvement
- but the measured packed bitstream is still worse than standard raw-text compression

## What v2 is currently proving

- bytes in
- grouped binary patches internally
- exact or near-exact reconstruction on real noisy text
- strong learned predictability of the latent bits

## Most recent result

Best fidelity-oriented real-corpus result:

- byte accuracy: `0.9876`
- estimated bpb: `1.5684`
- zlib-compressed learned bitstream bpb: `4.4418`
- zlib-compressed raw text bpb: `3.0611`

Best packed-bitstream result so far:

- `rate_weight=0.01`
- byte accuracy: `0.9799`
- estimated bpb: `2.1823`
- zlib-compressed learned bitstream bpb: `4.3861`

Interpretation:

- Loopy v2 is strong as a learned representation and modeling direction
- small to moderate rate pressure can improve the packed learned bitstream
- the active problem is now the compression/fidelity frontier
- Loopy v2 is still not yet competitive as a practical compressor with the current packing path

## Next resume step

Run one or two intermediate real-corpus rate points such as `0.003` and `0.005`, then compare packed bitstream size against fidelity.
