.PHONY: help install test verify build clean train-smoke benchmark

help:
	@echo "Cerebro Zero 1.0 commands:"
	@echo "  make install      Install the project and development dependencies"
	@echo "  make test         Run the complete regression suite"
	@echo "  make verify       Run tests, package checks and 1.0 smoke gates"
	@echo "  make build        Build the Python distribution"
	@echo "  make train-smoke  Run the deterministic tiny training path"
	@echo "  make benchmark    Run the end-to-end benchmark"
	@echo "  make clean        Remove generated Python/test caches"

install:
	python -m pip install -e '.[dev,web]'

test:
	python -m pytest -q

verify:
	python -m pytest -q
	python -c 'import platform; print(platform.platform())' | grep -q Android && echo "pip check skipped on Android/Termux: platform wheel metadata is not reliable there" || python -m pip check
	python -m build --wheel
	python scripts/release_gate.py
	python -c 'from cerebro_zero import Cerebro; r=Cerebro().run("verify"); assert r.success; print(r.text)'

build:
	python -m build

train-smoke:
	python -m pytest -q tests/test_1_0_golden_path.py

benchmark:
	python cli.py benchmark

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache build
