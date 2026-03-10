# Next

Loopy v2 is now in the Colab validation phase.

## Current status

The first Colab GPU baseline passed for modeling quality and speed, but not yet for practical compression.

Observed result:

- byte accuracy: `0.9861`
- estimated bpb: `1.6132`
- average epoch seconds: `4.66`
- zlib-compressed learned bitstream bpb: `4.5060`
- zlib-compressed raw text bpb: `3.0611`

## What this means

The learned representation is strong and GPU scaling works.
But the first true bitstream measurement shows that the current packed hard-bitstream is still worse than standard compression on raw text.

So the next clean step is:

- keep the same baseline
- add a small explicit rate penalty on Colab
- rerun the bitstream measurement
- see whether real packed size improves without damaging fidelity too much

## Immediate next step

Run the Colab `rate_weight=0.001` comparison, then measure it.

Training command:

```bash
!python -m loopy.train_binary_codec_v2 \
  --data-path loopy/data/real/twitter_support_5k.txt \
  --output-dir loopy/runs/v2_colab_rate_small \
  --epochs 20 \
  --batch-size 16 \
  --max-seq-len 128 \
  --patch-size 4 \
  --embed-dim 128 \
  --latent-dim 128 \
  --encoder-layers 2 \
  --decoder-layers 2 \
  --num-heads 4 \
  --dropout 0.0 \
  --weight-decay 0.0 \
  --bit-groups 6,6,6,6 \
  --rate-weight 0.001 \
  --balance-weight 0.001 \
  --align-weight 0.05
```

Measurement command:

```bash
!python -m loopy.measure_bitstream_v2 \
  --run-dir loopy/runs/v2_colab_rate_small \
  --data-path loopy/data/real/twitter_support_5k.txt \
  --output loopy/runs/v2_colab_rate_small/bitstream_summary.json
```

## Decision rule

Move forward only if:

- fidelity remains high
- packed bitstream size improves materially
- raw-text zlib/gzip is at least approached more closely than the current baseline

## Do not do next

- do not move to H100 yet
- do not make public compression claims yet
- do not discard the modeling result just because the first bitstream packing method is weak
