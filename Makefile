.PHONY: install lint format format-check typecheck test check precommit clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

format-check:
	ruff format --check src tests scripts

typecheck:
	mypy src

test:
	pytest

check: lint format-check typecheck test

precommit:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build *.egg-info
