# Breakpoint

Loopy has pivoted again.

## Current state

- the harness is now the default workflow for multi-run sweeps:
  - prepare a batch
  - run exact commands on local, Colab, or future remote compute
  - ingest results into a local ledger
- the corrected raw downstream baseline is now:
  - `patch_size=2`, `20` epochs
  - `bpb = 2.5258`
  - `accuracy = 0.5233`

- v1 symbolic middleware is archived
- v2 grouped independent-bit codec is archived as the current reference branch
- v3 codebook / patch-symbol work is the current reference branch
- v4 contextual codebook work is now the active implementation branch
- first `v4` real-text result is now in, and it was effectively a tie with `v3`
- first `v4.2` toy result is now in, and it improved over plain `v4`
- first `v4.2` real-text result is now in, and it beat both `v3` and `v4`
- first `v4.2` tuning pass is now in, and both variants improved again
- combined `v4.2` tuning test is now in, and it regressed
- lower residual-pressure `v4.2` test is now in, and it also regressed
- first `v4.2` downstream grouped-prior result is now in, and it is close to raw but still slightly worse
- `v4.2` at `6.0 bpb` is now in:
  - reconstruction is very strong (`0.9907` byte accuracy)
  - downstream grouped prior is `3.2052`
  - that beats old `v3` `6.0 bpb`, but not raw and not best `v3` `5.0 bpb`
- the old predictive-on-`v4.2` path was a no-op
- `v4` now has a real masked predictive objective
- local smoke test confirms `predictive_loss` is active again
- masked predictive `v4.2` at `6.0 bpb` is now in:
  - reconstruction: `0.9912` byte accuracy
  - downstream grouped prior: `3.1301`
  - this is better than plain `v4.2` at `6.0 bpb` (`3.2052`)
  - but still worse than corrected raw (`2.5258`) and best learned downstream `v3` (`2.8497`)
- first harness runs are now in:
  - `v42_6bpb_base` prior: `3.1965`
  - `v42_pw005_mp010` prior: `3.1787`
  - `v42_pw005_mp015` prior: `3.2130`
  - none are close to the corrected raw baseline yet

## What v2 settled

- reconstruction on real noisy text works
- there is a real compression vs fidelity tradeoff
- grouped downstream symbols help more than plain bitwise learned targets
- but the learned downstream stream still loses clearly to raw patches

So the next bottleneck is not:

- more tiny rate sweeps
- more grouped packing tweaks
- more light predictive-loss tuning

It is:

- a stronger latent redesign

## Current v3 status

Single-symbol `v3` was too weak:

- `patch_size=4`: blank output
- `patch_size=2`: still failed badly
- `patch_size=1`: plateaued around `0.75` byte accuracy

Product-codebook `v3` is now in the workspace:

- multiple sub-symbols per patch instead of one symbol
- first toy smoke result:
  - byte accuracy: `0.7589`
  - codebook perplexity: `6.90`
  - output is partially readable, but still weak
- soft assignments plus usage regularization changed the picture:
  - `patch_size=1` now reaches `0.9265` byte accuracy
  - codebook perplexity is about `250.97`
  - output is mostly readable
- `patch_size=2` is now viable at high capacity:
  - toy run with `num_codebooks=4`, `40` epochs
  - byte accuracy: `0.9746`
  - exact toy reconstruction
- short real-corpus smoke with the same high-capacity setup:
  - byte accuracy: `0.9892`
  - only tiny detail errors in the sample
- important caveat:
  - this setup has `raw_capacity_bpb = 16.0`
  - so it proves viability, not efficiency
- the first capacity-reduction sweep changed that:
  - `num_codebooks=2`, `sub_codebook_size=256`, `raw_capacity_bpb=8.0`
  - byte accuracy: `0.9892`
  - still near-exact real-text reconstruction
  - this became the first best efficiency-oriented `v3` baseline
- the next sweep pushed lower:
  - `num_codebooks=2`, `sub_codebook_size=128`, `raw_capacity_bpb=7.0`
  - byte accuracy: `0.9889`
  - this is now the best efficiency-oriented `v3` baseline
  - `num_codebooks=2`, `sub_codebook_size=64`, `raw_capacity_bpb=6.0`
  - byte accuracy: `0.9794`
  - this is the current cliff test: still alive, but clearly degraded
  - the same `6.0 bpb` run at `20` epochs improved to:
    - byte accuracy: `0.9846`
    - this means `6.0 bpb` is now a real frontier point, not just a failure case
  - the next lower test at `5.0 bpb` gave:
    - byte accuracy: `0.9464`
    - this is now clearly below the current quality bar

## Next resume step

Next resume step:

- keep `v3` `7.0 bpb` as the safer reconstruction reference
- keep `v3` `5.0 bpb` as the best downstream reference
- note that longer codec training at `5.0 bpb` did not help materially
- use the new `v4` branch to test the missing hypothesis:
  - cross-patch context before quantization
  - cross-patch context after quantization
- `v4` has now been validated on toy and compared on real text
- next resume step:
  - keep the contextual `v4.2` path
  - keep `residual_usage_weight = 0.005` and `residual_gate_bias = -2.0` as the best point so far
  - stop this residual tuning sweep
  - note the first downstream `v4.2` grouped prior result:
    - `bpb = 2.9551`
    - raw baseline: `2.9473`
    - best downstream `v3`: `2.8497`
  - move to the next branch:
- keep the best `v4.2` checkpoint
- use the best `v4.2` `6.0 bpb` checkpoint as the new balanced reference
- next practical move:
  - stop manual branch-toggling
  - build the controlled experiment runner around the current stable branches
  - use it to compare:
    - `v3 5.0 bpb` downstream winner
    - `v4.2 6.0 bpb` balanced reconstruction winner
    - `v4.2 6.0 bpb + masked predictive` improved balanced point
