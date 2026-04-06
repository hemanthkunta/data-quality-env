.PHONY: install run health gen-test openenv-validate qa infer infer-high-grade chat rl-train rl-eval check-100k self-improve docker-build docker-run

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	uvicorn env.app:app --host 0.0.0.0 --port 7860

health:
	curl -s http://localhost:7860/health

gen-test:
	$(PYTHON) -c "from env.dataset_gen import generate_dataset; print(generate_dataset(1, 42)[1])"

openenv-validate:
	$(PYTHON) -m pip install openenv-core
	$(PYTHON) -m openenv validate

qa:
	$(PYTHON) scripts/local_qa.py

infer:
	$(PYTHON) inference.py

infer-high-grade:
	$(PYTHON) high_grade_agent.py

chat:
	$(PYTHON) chat_agent.py --task-id 1 --seed 42

rl-train:
	$(PYTHON) scripts/train_rl_agent.py train --episodes 300 --output outputs/rl_policy.json

rl-eval:
	$(PYTHON) scripts/train_rl_agent.py eval --policy outputs/rl_policy.json --episodes-per-task 5

check-100k:
	$(PYTHON) scripts/check_100k_algorithms.py

self-improve:
	$(PYTHON) scripts/self_improve_loop.py --cycles 3 --episodes-per-cycle 200

docker-build:
	docker build -t dqe .

docker-run:
	docker run --rm -p 7860:7860 dqe
