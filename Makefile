# ML Project Makefile - Practical Quiz 1
# Targets: preprocess, train, evaluate, all
# Usage: make setup (first time), then make preprocess | make train | make all

.PHONY: preprocess train evaluate all clean setup

# Use venv Python if available, otherwise system python3
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Run data preprocessing
preprocess:
	@echo "=== Running Data Preprocessing ==="
	$(PYTHON) src/preprocess.py

# Run model training
train: preprocess
	@echo "=== Running Model Training ==="
	$(PYTHON) src/train.py

# Run model evaluation
evaluate: train
	@echo "=== Running Model Evaluation ==="
	$(PYTHON) src/evaluate.py

# Full pipeline: preprocess -> train -> evaluate
all: evaluate
	@echo "=== Pipeline Complete ==="

# Setup: create venv and install dependencies
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

# Optional: clean generated files
clean:
	rm -f data/processed/*.csv models/*.pkl results/*.txt
