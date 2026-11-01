# Project-pinned AWS profile/region and Cloudflare token come from .env and override the shell,
# so a profile exported for another project (e.g. AWS_PROFILE=velhera) is never used here.
-include .env
export AWS_PROFILE AWS_REGION CLOUDFLARE_API_TOKEN

TF_MAIN := infra/terraform
DATA_BUCKET = $(shell terraform -chdir=$(TF_MAIN) output -raw data_bucket)

.PHONY: whoami setup lint format test tf-bootstrap tf-init tf-plan tf-apply data data-upload notebooks smoke labelstudio golden-sample golden-export

whoami:           ## Muestra profile, cuenta y ARN activos (revisar antes de cualquier apply)
	@echo "AWS_PROFILE=$(AWS_PROFILE)"
	@aws sts get-caller-identity --query '[Account,Arn]' --output text

setup:            ## Instala entorno y hooks
	uv sync
	uv run pre-commit install

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

golden-export:
	uv run python labeling/export.py
