.PHONY: install lint test benchmark build serve data-v2 dry-run-v2

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .
	ruff format --check .

test:
	pytest --cov=evidenceagent_mm --cov-report=term-missing

benchmark:
	eamm make-benchmark benchmarks/eamm_bronze --sessions 12
	eamm --db /tmp/eamm-benchmark.db benchmark benchmarks/eamm_bronze --output benchmarks/results/cpu_bronze.json

build:
	python -m build

serve:
	eamm --db data/processed/evidence.db serve --host 127.0.0.1 --port 8000

data-v2:
	python scripts/build_v2_training_data.py

dry-run-v2: data-v2
	python scripts/train_v2.py --config configs/sft_4090.json --dry-run
	python scripts/train_v2.py --config configs/dpo_4090.json --dry-run
	python scripts/train_v2.py --config configs/grpo_4090.json --dry-run
