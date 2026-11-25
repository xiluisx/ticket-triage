#!/usr/bin/env bash
# [$] Deploy the inference API: ECR repo -> image push -> Lambda + Function URL (cents).
# Needs TRIAGE_API_KEY in .env (exported by the Makefile) and the Jev key in SSM.
set -euo pipefail
: "${TRIAGE_API_KEY:?add TRIAGE_API_KEY to .env}"

TF=infra/terraform
TAG="$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%S)"
KEY_SHA=$(printf '%s' "$TRIAGE_API_KEY" | sha256sum | cut -d' ' -f1)

terraform -chdir="$TF" apply -target=aws_ecr_repository.api -target=aws_ecr_lifecycle_policy.api
ECR=$(terraform -chdir="$TF" output -raw api_ecr_url)

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "${ECR%%/*}"
# Lambda rejects OCI manifests with attestations: build a plain Docker v2 image.
docker build --provenance=false -f docker/lambda.Dockerfile -t "$ECR:$TAG" .
docker push "$ECR:$TAG"

# Persist the deployed tag/hash so a later plain `make tf-apply` keeps the Lambda (gitignored).
cat > "$TF/api.auto.tfvars" <<VARS
api_image_tag  = "$TAG"
api_key_sha256 = "$KEY_SHA"
VARS
terraform -chdir="$TF" apply
echo "API URL: $(terraform -chdir="$TF" output -raw api_url)"
