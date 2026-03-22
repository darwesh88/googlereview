# Experiment Harness

The harness is Loopy's control plane.

It is intentionally inspired by the fixed-run, results-ledger style of `autoresearch`, but it is not autonomous architecture search yet.

## What it does

- expands a plan into exact commands
- keeps one batch directory per sweep
- collects finished artifacts into the batch for persistence
- bundles remote results into a portable zip
- compares finished runs against named baselines
- appends a ledger row for each experiment

## Files

- [experiment_runner.py](C:/Users/adarw/Desktop/googlereview/loopy/experiment_runner.py)
- [experiment_baselines.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_baselines.json)
- [experiment_plans/v42_masked_grid_10.json](C:/Users/adarw/Desktop/googlereview/loopy/experiment_plans/v42_masked_grid_10.json)

## Workflow

1. Prepare a batch from a plan.
2. Run the generated commands on local, Colab, or a future remote worker.
3. Ingest the finished results back into the ledger.
4. Persist remote artifacts so Colab/Vast runs are not lost.

## Prepare a batch

```powershell
python -m loopy.experiment_runner prepare --plan-file loopy/experiment_plans/v42_masked_grid_10.json
```

This creates a batch directory under `loopy/experiment_batches/` and writes:

- `batch.json`
- `commands.md`
- `experiments/*.json`

`commands.md` includes both:

- local `python -m ...` commands
- Colab-ready `!python -m ...` commands

## Check status

```powershell
python -m loopy.experiment_runner status --batch-dir loopy/experiment_batches/<batch-name>
```

## Run locally

```powershell
python -m loopy.experiment_runner run --batch-dir loopy/experiment_batches/<batch-name> --limit 2
```

## Persist remote results

After experiments finish on a remote worker, collect them into the batch:

```powershell
python -m loopy.experiment_runner collect --batch-dir loopy/experiment_batches/<batch-name>
```

Then bundle them into a single zip you can download from Colab or another worker:

```powershell
python -m loopy.experiment_runner bundle --batch-dir loopy/experiment_batches/<batch-name>
```

This writes `artifacts.zip` inside the batch directory by default.

## Restore a downloaded bundle locally

If the remote machine only gives you a zip back, restore it into your local repo:

```powershell
python -m loopy.experiment_runner restore --batch-dir loopy/experiment_batches/<batch-name> --bundle-file loopy/experiment_batches/<batch-name>/artifacts.zip
```

## Ingest results

```powershell
python -m loopy.experiment_runner ingest --batch-dir loopy/experiment_batches/<batch-name>
```

This writes `results.json` and appends rows to `loopy/experiment_ledger.jsonl`.

## Current use

Use the harness for controlled sweeps and baseline comparisons.
It now separates:

- local repo as the control plane
- remote GPU sessions as execution workers
- batch zips as the persistence layer

This should be enough before adding any Vast.ai-specific launcher.

Current recommendation:

- keep using the harness
- but do not keep sweeping the current `v42_masked_grid_10` neighborhood
- the next batch should target a cleaner dataset or a larger hypothesis shift
