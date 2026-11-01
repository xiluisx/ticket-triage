# Ticket Triage — reglas para el agente

Proyecto final de MLOps (AWS + ML). Specs en `docs/superpowers/specs/` (empezar por el spec maestro).

## Flujo de trabajo

- **Planear antes de implementar.** No escribir código ni crear archivos de implementación hasta que el usuario apruebe esa pieza concreta.
- **Nunca hacer `git commit`.** Al terminar un avance, avisar que es buen punto de commit y sugerir:
  - rama: `<tipo>/<slug>` (nunca trabajar directo en `main`)
  - mensaje en Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `test:`, `refactor:`…)
  - El usuario hace el commit.
- **Avisar cuando haya que capturar pantalla** para la evidencia del reporte. En cada momento relevante decir:
  - qué capturar (ej. CI en verde, `terraform apply`, recurso en consola AWS, gráfica del EDA, tabla de métricas, Kanban, dashboard)
  - dónde guardarla: `docs/evidence/YYYY-MM-DD-<slug>.png`
  - `docs/evidence/*.png` está en `.gitignore` (puede mostrar el ID de cuenta): son para el reporte PDF, no para el repo. Gráficas sin datos de la cuenta van en `docs/figures/`.
  - para qué sección del reporte sirve
  - El avance del **4 de noviembre de 2026** requiere evidencia explícita en el reporte.

## Seguridad

- Repo público: cero secretos en git. Secretos locales en `.env` (gitignored); en AWS, SSM Parameter Store.
- Nunca pedir ni leer el valor de `TYPESAFE_API_KEY`, tokens de Cloudflare o credenciales AWS.

## Costos

Presupuesto total < $10 USD. Prohibido: endpoints real-time de SageMaker, Studio apps encendidas, NAT Gateway, MLflow administrado. Avisar antes de cualquier acción que genere costo.

## Idioma

Conversación y documentación del proyecto en español; código, nombres de recursos y commits en inglés.
