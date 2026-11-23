# Ticket Triage — reglas para agentes de código

Proyecto final de MLOps (AWS + ML). Repo público. Specs en `docs/superpowers/specs/`, planes en `docs/superpowers/plans/`.

## Tarea actual

Ejecutar `docs/superpowers/plans/2026-11-23-data-models.md` **una tarea a la vez, en orden**, siguiendo cada paso tal como está escrito (TDD: test que falla → implementación → test que pasa).

- Antes de empezar: confirmar que la rama actual es `feat/data-models` (`git branch --show-current`). Si no, detenerse y avisar.
- Al terminar cada tarea: correr `make lint && make test`. Ambos deben pasar antes de seguir.
- Si un test falla de forma inesperada o el plan contradice el código existente: **detenerse y avisar**; no improvisar cambios de diseño.
- No modificar archivos fuera de los listados en la tarea, salvo que el paso lo indique.
- Marcar el checkbox `- [ ]` → `- [x]` del plan al completar cada paso.

## Git

- **Nunca hacer `git commit`, `git push`, `git merge` ni `git rebase`.** En cada "Punto de commit (usuario)", detenerse, mostrar el comando sugerido y esperar a que el usuario commitee.
- Nunca trabajar en `main`.

## Costos (presupuesto total < $10 USD)

- **Nunca ejecutar comandos que llaman APIs pagadas.** Los corre el usuario:
  `make smoke-claude`, `make kappa-gate`, `make distill`, `make predict`, `make smoke`,
  `python -m training.distill ...`, `python -m training.predict ...`, `scripts/smoke_*.py`.
  Al llegar a esos pasos: detenerse, decir el costo estimado del plan y esperar.
- Nunca pasar `--yes` a un script que pida confirmación de costo.
- Prohibido crear o modificar recursos en AWS: nada de `terraform apply`, `aws ... create/put/delete`, endpoints de SageMaker, NAT Gateway.

## Seguridad

- Nunca leer, imprimir ni modificar `.env`. Nunca pedir ni mostrar `TYPESAFE_API_KEY`, tokens de Cloudflare ni credenciales AWS.
- Cero secretos en archivos versionados.
- No instalar dependencias fuera de las que indica el plan (`uv add --package training "anthropic[bedrock]"`).

## Estilo

- Conversación y documentación en español; código, nombres y commits en inglés.
- Python 3.12, `uv run` para todo, ruff (line-length 100).
- Nombres de archivos de test únicos en todo el repo.
