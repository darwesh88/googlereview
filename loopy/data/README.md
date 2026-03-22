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

Clean benchmark helper:

- [prepare_hf_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/prepare_hf_corpus.py) can fetch and prepare a reproducible Hugging Face dataset slice such as TinyStories or WikiText into the same line-based format
