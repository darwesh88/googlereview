# Loopy Colab Plan

This is the next scale-up phase after the CPU baseline.

## Why Colab is next

Colab is the right bridge between local CPU tests and rented H100s.

Use it to:

- verify the current baseline on a GPU
- run longer horizons without waiting on CPU
- do a small ablation matrix
- measure true bitstream size, not just the entropy proxy

Do not use Colab yet to chase huge model sizes or many architecture changes at once.

## Fastest entrypoint

If you want the quickest start, open the ready notebook in Colab:

- [Loopy_v2_Colab_Baseline.ipynb](C:/Users/adarw/Desktop/googlereview/loopy/Loopy_v2_Colab_Baseline.ipynb)

When the repo is on GitHub, the Colab URL is:

`https://colab.research.google.com/github/darwesh88/googlereview/blob/main/loopy/Loopy_v2_Colab_Baseline.ipynb`

The notebook:

- clones the repo
- uses a normal public GitHub clone
- installs dependencies only if the requirements file exists
- runs the current best baseline
- measures the baseline bitstream
- runs the `rate_weight=0.001` comparison
- measures the comparison bitstream

If Colab still asks for a GitHub token, you are almost certainly running an older cached notebook version. Reopen the notebook from the GitHub URL after pushing the latest changes.

## Current best CPU baseline

Use this as the starting point:

- data: `loopy/data/real/twitter_support_5k.txt`
- `patch_size=4`
- `bit-groups 6,6,6,6`
- `embed_dim=128`
- `latent_dim=128`
- `encoder_layers=2`
- `decoder_layers=2`
- `num_heads=4`
- `dropout=0.0`
- `weight_decay=0.0`
- `rate_weight=0.0`
- `balance_weight=0.001`
- `align_weight=0.05`
- best CPU reference: `20` epochs, byte accuracy `0.9849`, estimated bpb `1.8326`

## Colab setup steps

1. Open a GPU runtime in Colab.
2. Open the ready notebook or clone the repo manually.
3. Install minimal dependencies.
4. Verify GPU visibility in PyTorch.
5. Confirm the prepared dataset file exists in the repo clone.
6. Run the baseline first before any variants.

Suggested setup cells if you are not using the notebook:

```bash
!git clone <your-repo-url>
%cd googlereview
!python -m pip install -U pip
!python -m pip install -r loopy/requirements.txt
```

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## Phase C1: GPU baseline reproduction

Goal:

- reproduce the best CPU baseline on a Colab GPU
- confirm the run is stable
- collect speed and fidelity numbers

Command:

```bash
!python -m loopy.train_binary_codec_v2 \
  --data-path loopy/data/real/twitter_support_5k.txt \
  --output-dir loopy/runs/v2_colab_baseline \
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
  --rate-weight 0.0 \
  --balance-weight 0.001 \
  --align-weight 0.05
```

Success bar:

- byte accuracy roughly in the CPU range or better
- reconstruction remains highly faithful
- run is clearly faster than CPU

## Phase C2: small rate sweep

Goal:

- see whether explicit compression pressure helps or hurts once the run is off CPU

Run three values:

- `rate_weight=0.0`
- `rate_weight=0.001`
- `rate_weight=0.003`

Keep everything else fixed.

Decision rule:

- keep the smallest value that lowers or preserves bitstream size without harming readability too much

## Phase C3: true bitstream measurement

Use the new measurement script after each completed run.

Command:

```bash
!python -m loopy.measure_bitstream_v2 \
  --run-dir loopy/runs/v2_colab_baseline \
  --data-path loopy/data/real/twitter_support_5k.txt \
  --output loopy/runs/v2_colab_baseline/bitstream_summary.json
```

This script reports:

- packed hard-bit size
- zlib-compressed bitstream size
- gzip-compressed bitstream size
- raw-text zlib/gzip baselines

Important:

- this is still a prototype measurement
- it is better than the entropy proxy alone, but not yet a production codec measurement

## Phase C4: one scale-up ablation

Only after C1 to C3 are stable.

Pick one scale-up axis, not several.

Recommended first ablation:

- keep `patch_size=4`
- increase model width to `embed_dim=192`, `latent_dim=192`
- keep depth the same

Reason:

- this tests whether capacity helps fragile details like names and numbers without changing the whole setup

## Phase C5: first prior-model preparation

Do not train the LLM/prior yet.

Instead, confirm:

- the codec is stable
- bitstream size is measurable
- reconstruction quality is good enough to justify a prior model over the codes

Only after that should we start building the next-stage language model on top of the learned binary stream.

## What to save from every Colab run

- `best_metrics.json`
- `sample_reconstruction.txt`
- `bitstream_summary.json`
- final command used
- Colab GPU type
- epoch time or total training time

## Stop conditions

Do not move to H100 rentals yet if any of these happen:

- fidelity drops sharply on GPU variants
- bitstream size does not improve meaningfully with rate-aware runs
- fragile details get much worse with scale
- results are unstable across repeated runs

## When H100 becomes justified

H100 becomes justified only when:

- the Colab baseline reproduces cleanly
- one rate-aware variant is clearly better or clearly safer
- bitstream measurement looks good enough to warrant larger runs
- we are blocked by Colab memory or training time, not by architectural uncertainty


