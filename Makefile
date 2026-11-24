.PHONY: help clean test train generate

PYTHON ?= python

help:
	@echo Available targets:
	@echo   make clean     - Remove caches and build artifacts
	@echo   make test      - Run all pytest unit tests
	@echo   make train     - Train tiny Amharic GPT model
	@echo   make generate  - Generate text using checkpoint

clean:
	$(PYTHON) scripts/clean.py

test:
	pytest tests/ -v

train:
	$(PYTHON) -m src.train --seed 42 --config_type tiny --max_iters 200 --lr 0.001 --checkpoint_dir checkpoints

generate:
	$(PYTHON) -m src.generate --seed 42 --checkpoint_dir checkpoints --prompt "ሰላም" --max_new_tokens 60
