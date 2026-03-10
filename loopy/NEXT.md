# Next

Loopy v2 is now the active path.

## Current status

The real-corpus baseline improved further with a longer CPU run.

Best current CPU baseline on `twitter_support_5k`:

- epochs: `20`
- byte accuracy: `0.9849`
- estimated bpb: `1.8326`
- bit density: `0.4840`
- reconstruction stayed highly faithful on real noisy text

## What this means

CPU validation is now strong enough.
The next clean step is no longer a bigger CPU run. The next clean step is a medium-scale GPU validation phase on Google Colab.

## Immediate next step

Prepare the first Colab-scale experiment setup.

Use the current best baseline as the starting point:

- `patch_size=4`
- `bit-groups 6,6,6,6`
- `rate_weight=0.0` as the main baseline
- `rate_weight=0.001` as the follow-up comparison

## Decision rule

Move beyond Colab only if:

- the baseline stays strong on GPU
- medium-scale runs remain stable
- real compressed-size measurement looks strong enough to justify larger spend

## Do not do next

- do not keep stretching CPU runs further
- do not jump straight to H100 rentals
- do not claim full compression superiority before true compressed-size accounting exists
