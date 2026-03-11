# Next

Loopy v2 now has two important real-corpus results:

- a strong modeling baseline with high fidelity
- a first modest packed-bitstream improvement under rate pressure

## Current status

Best fidelity-oriented baseline:

- byte accuracy: `0.9876`
- estimated bpb: `1.5684`
- zlib-compressed learned bitstream bpb: `4.4418`
- zlib-compressed raw text bpb: `3.0611`

Best packed-bitstream result so far:

- run: `v2_twitter_rate_med`
- `rate_weight=0.01`
- byte accuracy: `0.9799`
- estimated bpb: `2.1823`
- zlib-compressed learned bitstream bpb: `4.3861`

Best middle tradeoff so far:

- run: `v2_twitter_rate_003`
- `rate_weight=0.003`
- byte accuracy: `0.9810`
- estimated bpb: `2.2935`
- zlib-compressed learned bitstream bpb: `4.3997`

Failed middle point:

- run: `v2_colab_rate_005`
- `rate_weight=0.005`
- byte accuracy: `0.9805`
- zlib-compressed learned bitstream bpb: `4.5231`

## What this means

The architecture clearly works.

Rate pressure can improve the real packed bitstream, but it currently costs some fidelity.
So the next clean task is to map the tradeoff curve, not to jump to bigger hardware.

## Immediate next step

Run one or two tighter rate points on the same real corpus:

- `rate_weight=0.002`
- `rate_weight=0.0025`

Keep everything else fixed:

- `patch_size=4`
- `bit-groups 6,6,6,6`
- `epochs 10`
- `batch_size 8`

For each run, record:

- `best_metrics.json`
- `sample_reconstruction.txt`
- `bitstream_summary.json`

## Decision rule

Move forward only if one of the intermediate rate settings:

- improves packed learned-bitstream bpb relative to `4.3997`
- keeps byte accuracy close to or above `0.981`
- keeps reconstruction errors local rather than structural

## Do not do next

- do not move to H100 yet
- do not claim the codec beats standard text compression yet
- do not keep extending epochs blindly before the rate frontier is mapped
