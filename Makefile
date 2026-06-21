.PHONY: help install ingest report all notebook test lint clean

help:
	@echo "macro-cpi targets:"
	@echo "  install   - create venv and install package (uv venv + uv pip install -e .[dev])"
	@echo "  ingest    - fetch FRED + yfinance + Treasury into SQLite"
	@echo "  report    - build features, walk-forward validate, write markdown report"
	@echo "  all       - ingest + report"
	@echo "  notebook  - execute the diagnostic notebook end-to-end"
	@echo "  test      - run pytest"
	@echo "  lint      - run ruff"
	@echo "  clean     - remove caches and the local database"

install:
	uv venv .venv
	. .venv/bin/activate && uv pip install -e ".[dev]"

ingest:
	. .venv/bin/activate && macro-cpi ingest

report:
	. .venv/bin/activate && macro-cpi report

all:
	. .venv/bin/activate && macro-cpi all

notebook:
	. .venv/bin/activate && python notebooks/_build_notebook.py

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info
	rm -f data/*.db reports/*.png
