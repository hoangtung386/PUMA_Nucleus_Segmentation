.PHONY: help install dev test lint format clean preprocess train inference docker-build docker-run

PYTHON ?= python
IMAGE  ?= symbiopan-v9-cellpath
DOCKER := docker

INPUT_DIR  ?= input
OUTPUT_DIR ?= output

help:
	@echo "Targets:"
	@echo "  install      Install package + dev deps"
	@echo "  test         Run pytest with coverage"
	@echo "  lint         Run ruff check"
	@echo "  format       Run ruff format + autofix"
	@echo "  clean        Remove __pycache__ + .pyc + build artifacts"
	@echo "  preprocess   Run data preprocessing"
	@echo "  train        Run Stage 1 training"
	@echo "  inference    Run WSI inference (override INPUT_DIR / OUTPUT_DIR)"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container with mounted volumes"

install:
	pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=symbiopan --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check . --exclude notebooks

format:
	$(PYTHON) -m ruff check --fix . --exclude notebooks
	$(PYTHON) -m ruff format . --exclude notebooks

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .ruff_cache .pytest_cache .mypy_cache .coverage

preprocess:
	$(PYTHON) -m scripts.preprocess

train:
	$(PYTHON) -m scripts.train_stage1

inference:
	mkdir -p $(INPUT_DIR) $(OUTPUT_DIR)
	$(PYTHON) -m scripts.infer_wsi --input $(INPUT_DIR) --output $(OUTPUT_DIR)

docker-build:
	$(DOCKER) build -t $(IMAGE) .

docker-run:
	mkdir -p $(OUTPUT_DIR) $(INPUT_DIR)
	$(DOCKER) run --rm --shm-size=8g --memory=32g --platform=linux/amd64 \
		--network none --gpus all \
		-v "$(PWD)/$(INPUT_DIR)/:/input/images/melanoma-whole-slide-image/" \
		-v "$(PWD)/$(OUTPUT_DIR)/:/output/" \
		$(IMAGE)
