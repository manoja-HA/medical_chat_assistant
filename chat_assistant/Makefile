# Makefile for Chat Assistant project

# Paths
PY_SRC := src/app
TESTS := tests
PY_FILES := $(PY_SRC) $(TESTS)

# Tools (managed by Poetry per pyproject.toml)
POETRY := poetry

.PHONY: help install format lint fix type test clean clean-all

help:
	@echo "Common targets:"
	@echo "  install   - Install dependencies with Poetry"
	@echo "  format    - Run isort and black to format the code"
	@echo "  lint      - Run ruff, black --check, isort --check, and mypy"
	@echo "  fix       - Run ruff --fix then format (isort + black)"
	@echo "  type      - Run mypy type checks"
	@echo "  test      - Run pytest"
	@echo "  clean     - Remove cache and temporary files"
	@echo "  clean-all - clean + remove build/dist artifacts"

install:
	$(POETRY) install --no-interaction --no-root

format:
	$(POETRY) run isort $(PY_FILES)
	$(POETRY) run black $(PY_FILES)

lint:
	$(POETRY) run ruff check $(PY_FILES)
	$(POETRY) run black --check $(PY_FILES)
	$(POETRY) run isort --check-only $(PY_FILES)
	$(POETRY) run mypy $(PY_SRC)

fix:
	$(POETRY) run ruff check --fix $(PY_FILES)
	$(MAKE) format

type:
	$(POETRY) run mypy $(PY_SRC)

test:
	$(POETRY) run pytest -q

clean:
	@echo "Cleaning caches and temporary files..."
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache .ipynb_checkpoints

clean-all: clean
	@echo "Removing build and dist artifacts..."
	rm -rf build dist *.egg-info .coverage coverage.xml htmlcov
