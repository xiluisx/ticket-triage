# Lambda image for the inference API. Build from the repo root:
#   docker build -f docker/lambda.Dockerfile -t ticket-triage-api .
# models/ is gitignored: the image is built locally from the trained artifacts.
FROM public.ecr.aws/lambda/python:3.12

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY backend/triage_core backend/triage_core
COPY backend/services backend/services
COPY training/pyproject.toml training/pyproject.toml
RUN uv export --frozen --no-dev --no-emit-workspace --package triage-services --no-hashes \
        -o requirements.txt \
    && uv pip install --no-cache --python /var/lang/bin/python3 \
        --target "${LAMBDA_TASK_ROOT}" -r requirements.txt \
    && uv pip install --no-cache --no-deps --python /var/lang/bin/python3 \
        --target "${LAMBDA_TASK_ROOT}" ./backend/triage_core ./backend/services \
    && rm -rf /build

WORKDIR ${LAMBDA_TASK_ROOT}
COPY datasets/categories_v1.json datasets/priorities_v2.json ${LAMBDA_TASK_ROOT}/datasets/
COPY models/ ${LAMBDA_TASK_ROOT}/models/
ENV DATASETS_DIR=${LAMBDA_TASK_ROOT}/datasets \
    MODELS_DIR=${LAMBDA_TASK_ROOT}/models

CMD ["triage_services.handler.handler"]
