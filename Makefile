.PHONY: preprocess stage1 inference docker-build docker-run lint clean

preprocess:
	python -m scripts.run_preprocess

stage1:
	python -m scripts.run_stage1

inference:
	python -m scripts.run_inference --input test --output output

docker-build:
	docker build -t symbiopan-v8-cellpath .

docker-run:
	mkdir -p output && \
	docker run --rm --shm-size=8g --memory=32g --platform=linux/amd64 \
		--network none --gpus all \
		-v "$(PWD)/test/:/input/images/melanoma-whole-slide-image/" \
		-v "$(PWD)/output/:/output/" \
		symbiopan-v8-cellpath

lint:
	python -m ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
