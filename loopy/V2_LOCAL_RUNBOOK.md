# Loopy v2 Local Runbook

This is the first execution path for the semantic binary codec.

The goal on this machine is simple:

- prove the codec trains
- prove it can overfit tiny data
- inspect exact reconstruction behavior
- inspect bit usage metrics

## Important staging rule

Do not start by compressing hard.

The first goal is to prove the architecture can reconstruct at all.
That means the earliest tests should be easier than real compression.

## Files

- [v2_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v2_config.py)
- [binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/binary_codec_v2.py)
- [train_binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_binary_codec_v2.py)

## Stage 0: single-byte sanity test

This stage passed.

## Stage 1: revised 2-byte patch diagnostic

This stage passed.

## Stage 2: revised 4-byte high-capacity patch test

This stage passed.

## Stage 3: 4-byte moderate-capacity test

This stage stayed strong under a tighter budget.

Observed result:

- byte accuracy: `0.9645`
- estimated bpb: `3.6890`
- reconstruction stayed almost exact, with only a small localized typo in the sample
- raw capacity bpb: `6.0`

Conclusion:

- the representation remained strong after reducing the budget to `24` bits per `4`-byte patch
- that justified the first real local corpus run

## Stage 4: first real local test

This stage passed strongly on `twitter_support_5k`.

Observed result:

- byte accuracy: `0.9821`
- estimated bpb: `2.2744`
- bit density: `0.4752`
- reconstruction stayed highly faithful on unseen validation text

Important observed weakness:

- names, numbers, and similar fragile details are still the main reconstruction errors

Interpretation:

- the architecture is stable on real noisy text
- the next clean step is not a huge scale jump yet
- the next clean step is to add a small rate penalty and see if compression can improve without breaking readability

## Stage 5: small rate-aware local follow-up

Run the same real corpus with a small rate penalty:

```powershell
python -m loopy.train_binary_codec_v2 --data-path loopy/data/real/twitter_support_5k.txt --output-dir loopy/runs/v2_twitter_rate_small --epochs 10 --batch-size 8 --max-seq-len 128 --patch-size 4 --embed-dim 128 --latent-dim 128 --encoder-layers 2 --decoder-layers 2 --num-heads 4 --dropout 0.0 --weight-decay 0.0 --bit-groups 6,6,6,6 --rate-weight 0.001 --balance-weight 0.001 --align-weight 0.05
```

## What to look for

- reconstruction quality
- byte accuracy
- estimated bits per byte
- whether local errors stay small and localized
- whether bit density remains healthy

## Decision rule before GPU rental

Move to an H100 only if:

- the real local run stays strong
- the small rate-aware run does not destroy fidelity
- the representation still looks compressible and stable

If those are not true yet, keep iterating locally.
