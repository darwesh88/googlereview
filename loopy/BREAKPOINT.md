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

Best middle tradeoff result so far:

- `rate_weight=0.003`
- byte accuracy: `0.9810`
- estimated bpb: `2.2935`
- zlib-compressed learned bitstream bpb: `4.3997`

Rejected middle point:

- `rate_weight=0.002`
  - byte accuracy: `0.9803`
  - estimated bpb: `2.5042`
  - zlib-compressed learned bitstream bpb: `4.5546`
- `rate_weight=0.0025`
  - byte accuracy: `0.9840`
  - estimated bpb: `2.3094`
  - zlib-compressed learned bitstream bpb: `4.5508`
- `rate_weight=0.005`
  - byte accuracy: `0.9805`
  - estimated bpb: `2.3675`
  - zlib-compressed learned bitstream bpb: `4.5231`

Interpretation:

- Loopy v2 is strong as a learned representation and modeling direction
- small to moderate rate pressure can improve the packed learned bitstream
- `0.003` is currently the best tested compromise between compression pressure and fidelity
- `0.01` remains the best packed-bitstream point so far
- grouped packing did not help
- the learned stream also lost to a raw patch baseline in downstream patch prediction
- a light predictive auxiliary-loss branch improved learned downstream `bpb` only slightly:
  - old learned patch prior: `5.1364`
  - predictive learned patch prior: `5.0593`
  - raw patch prior: `3.6991`
- grouped-symbol 5-epoch downstream result on the baseline codec:
  - grouped patch prior: `4.9839`
  - better than bitwise learned, still worse than raw
- the active problem is now the codec objective itself
- Loopy v2 is still not yet competitive as a practical compressor with the current packing path

## Next resume step

Stop the tiny local rate sweep.

Grouped packing has now been tested and rejected for the current bit layout.

The next resume step is:

- stop tweaking the light predictive auxiliary-loss branch
- test grouped-symbol prior on the predictive codec
- if that still loses clearly to raw, move to a stronger latent redesign so learned patches are easier to predict downstream
