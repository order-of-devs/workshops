#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = oskar_as_a_service
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


.PHONY: pull_model
pull_model:
	ollama pull granite4


.PHONE: build_rag
build_rag:
	uv run packages/knowladge_base_rag/src/knowladge_base_rag/main.py

.PHONE: run_flow_php_agent
run_flow_php_agent:
	uv run packages/flow_php_agent/src/flow_php_agent/main.py

.PHONE: run_emmet_agent
run_emmet_agent:
	uv run packages/emmet_agent/src/emmet_agent/main.py

.PHONE: run_mlflow
run_mlflow:
	docker compose containers/docker-compose.yml

.PHONE: register_prompt
register_prompt:
	uv run packages/registry_prompt/src/registry_prompt/main.py

.PHONE: run_deepfabric
run_deepfabric:
	uv run deepfabric generate packages/dataloader/src/dataloader/flow_php_config.yaml --output-save-as flow-php-dataset.jsonl