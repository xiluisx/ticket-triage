# Project-pinned AWS profile/region and Cloudflare token come from .env and override the shell,
# so a profile exported for another project (e.g. AWS_PROFILE=velhera) is never used here.
-include .env
export AWS_PROFILE AWS_REGION CLOUDFLARE_API_TOKEN

TF_MAIN := infra/terraform
DATA_BUCKET = $(shell terraform -chdir=$(TF_MAIN) output -raw data_bucket)

.PHONY: whoami setup lint format test tf-bootstrap tf-init tf-plan tf-apply data data-upload notebooks smoke labelstudio golden-sample golden-subsample golden-export \
	priorities-v2 splits smoke-claude kappa-gate distill train eval-sets predict thresholds benchmark fairness

whoami:           ## Muestra profile, cuenta y ARN activos (revisar antes de cualquier apply)
	@echo "AWS_PROFILE=$(AWS_PROFILE)"
	@aws sts get-caller-identity --query '[Account,Arn]' --output text

setup:            ## Instala entorno y hooks
	uv sync
	uv run pre-commit install
	uv run python -m ipykernel install --sys-prefix --name ticket-triage --display-name "ticket-triage (.venv)"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

tf-bootstrap:     ## Crea el bucket de estado (una vez)
	terraform -chdir=infra/bootstrap init
	terraform -chdir=infra/bootstrap apply

tf-init:
	terraform -chdir=$(TF_MAIN) init -backend-config=backend.hcl

tf-plan:
	terraform -chdir=$(TF_MAIN) plan

tf-apply:
	terraform -chdir=$(TF_MAIN) apply

data:             ## Descarga archivo de narrativas CFPB y construye el parquet
	uv run python scripts/download_cfpb.py

data-upload:
	aws s3 cp data/cfpb/complaints.parquet s3://$(DATA_BUCKET)/raw/cfpb/complaints.parquet

notebooks:        ## Ejecuta notebooks y genera .ipynb con salidas
	uv run jupytext --to ipynb --execute notebooks/01_eda.py
	uv run jupytext --to ipynb --execute notebooks/02_baseline.py

smoke:
	uv run python scripts/smoke_jev.py
	uv run python scripts/smoke_bedrock.py

labelstudio:
	docker compose up -d labelstudio

golden-sample:
	uv run python labeling/sample.py

golden-subsample: ## Reduce el golden set a 100 conservando lo ya etiquetado
	uv run python labeling/subsample.py

golden-export:
	uv run python labeling/export.py

priorities-v2:    ## Golden set v2 (prioridad binaria)
	uv run python -m training.priority

splits:           ## Splits temporales train/val/test
	uv run python -m training.splits

smoke-claude:     ## [$] Smoke de Claude Haiku 4.5 vía Bedrock (~$0.0005)
	uv run python scripts/smoke_claude.py

kappa-gate:       ## [$] Claude vs golden (kappa), ~$0.08
	uv run python -m training.distill gate

distill:          ## [$] Claude etiqueta ~2k tickets de prioridad (~$1.50)
	uv run python -m training.distill run

train:            ## Entrena clásicos de categoría y prioridad
	uv run python -m training.train category
	uv run python -m training.train priority

eval-sets:        ## Marcos de evaluación val/test/golden
	uv run python -m training.evalsets

predict:          ## [$] Predicciones de clásico, Jev y Claude sobre los marcos (~$1.60)
	uv run python -m training.predict

thresholds:       ## Barrido de umbrales de la cascada
	uv run python -m training.thresholds category
	uv run python -m training.thresholds priority

benchmark:        ## Tabla de 4 modelos + cascada
	uv run python -m training.benchmark

fairness:         ## Métricas por grupo y fairness.json
	uv run python -m training.fairness
