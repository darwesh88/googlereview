# Experiment Harness

The harness is Loopy's control plane.

It is intentionally inspired by the fixed-run, results-ledger style of `autoresearch`, but it is not autonomous architecture search yet.

## What it does

- expands a plan into exact commands
- keeps one batch directory per sweep
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

## Ingest results

```powershell
python -m loopy.experiment_runner ingest --batch-dir loopy/experiment_batches/<batch-name>
```

This writes `results.json` and appends rows to `loopy/experiment_ledger.jsonl`.

## Current use

Use the harness for controlled `v4.2` sweeps and baseline comparisons.
It keeps run bookkeeping stable before we add any Vast.ai or other remote launcher.
