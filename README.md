# Ticket Triage

Multi-tenant SaaS that classifies support tickets (email + API) by **category** and **priority** using a cost-aware cascade: tenant classical model → Jev (TypeSafe) → Claude on Bedrock.

Class project (MLOps, AWS + ML).

- [Architecture](docs/architecture.md)

## Quickstart

```bash
cp .env.example .env   # fill in values
make setup
make test
```

## Data

CFPB Consumer Complaint Database **Narratives Archive** (public domain). CFPB stopped publishing narratives on 2026-08-14; the archive keeps previously published ones.

## Modelos (Subproyecto 1)

Targets marcados `[$]` llaman APIs pagadas (Claude Haiku 4.5 en Bedrock, Jev), imprimen el costo estimado y piden confirmación. Las predicciones se guardan en cache: repetir un target no vuelve a pagar.

```bash
make priorities-v2 splits          # datos
make smoke-claude kappa-gate       # [$] acceso a Claude y acuerdo con el golden set
make distill train                 # [$] etiquetas de prioridad (destilación), entrenamiento local
make eval-sets predict             # [$] predicciones de clásico, Jev y Claude
make thresholds benchmark fairness # análisis offline
```

Resultados: `reports/benchmark.md`, `reports/fairness.md`, `reports/cascade_*.json`, `reports/kappa_gate.json`; figuras en `docs/figures/`.
