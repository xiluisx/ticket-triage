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
