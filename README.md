# llm-engineering-fundamentals

Production-grade implementations of core LLM building blocks — from **tokenization** to **inference optimization** — with tests, benchmarks, and visualizations.

## Repo layout

- `01-tokenization/`: BPE tokenizer + token visualizer + embedding distance demos
- `02-positional-embeddings/` … `16-synthetic-data/`: planned modules (implemented step-by-step)
- `shared/`: shared utilities (plotting/benchmarks/datasets)

## Quick start (Project 01)

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"

# run tests
make test

# run token visualizer (local)
make token-viz
```

## Quality bar

- **Reproducible**: deterministic seeds where applicable
- **Tested**: `pytest`
- **Linted**: `ruff`
- **CI**: GitHub Actions runs lint + tests on push/PR


