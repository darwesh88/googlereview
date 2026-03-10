# Real Corpus Data

Use this folder for real-data experiments.

Suggested layout:

- `loopy/data/raw/<dataset-name>/`
- `loopy/data/real/<dataset-name>.txt`
- `loopy/data/real/<dataset-name>.report.json`

Guidelines:

- keep raw source files under `raw/`
- keep prepared one-sample-per-line corpora under `real/`
- avoid committing private or sensitive production data
- if the data includes support tickets or user text, prefer running preparation with redaction flags first
