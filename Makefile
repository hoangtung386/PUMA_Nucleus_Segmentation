.PHONY: help install dev test lint format clean preprocess stage1 inference docker-build docker-run

PYTHON ?= python
IMAGE  ?= symbiopan-v9-cellpath
DOCKER := docker

help:
	@echo "Targets:"
	@echo "  install      Install package + dev deps"
	@echo "  test         Run pytest"
	@echo "  lint         Run ruff"
	@echo "  format       Run ruff --fix"
	@echo "  clean        Remove __pycache__ + .pyc + build artifacts"
	@echo "  preprocess   Run data preprocessing"
	@echo "  stage1       Run stage 1 training"
	@echo "  inference    Run WSI inference"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container with mounted volumes"

install:
	pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=symbiopan --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .ruff_cache .pytest_cache .mypy_cache

preprocess:
	$(PYTHON) -m scripts.preprocess

stage1:
	$(PYTHON) -m scripts.train_stage1

inference:
	mkdir -p test && $(PYTHON) -m scripts.infer_wsi --input test --output output

docker-build:
	$(DOCKER) build -t $(IMAGE) .

docker-run:
	mkdir -p output test && \
	$(DOCKER) run --rm --shm-size=8g --memory=32g --platform=linux/amd64 \
		--network none --gpus all \
		-v "$(PWD)/test/:/input/images/melanoma-whole-slide-image/" \
		-v "$(PWD)/output/:/output/" \
		$(IMAGE)
