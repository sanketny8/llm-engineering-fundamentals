.PHONY: install dev lint test fmt token-viz

install:
	python -m pip install -U pip
	pip install -e .

dev:
	python -m pip install -U pip
	pip install -e ".[dev]"

lint:
	ruff check .

fmt:
	ruff check --fix .

test:
	pytest

token-viz:
	streamlit run 01-tokenization/visualizer_app.py




