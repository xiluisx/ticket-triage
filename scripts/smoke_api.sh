#!/usr/bin/env bash
# [$ < 0.01] Post 3 fixed tickets to the deployed API and list them back.
set -euo pipefail
: "${TRIAGE_API_KEY:?add TRIAGE_API_KEY to .env}"
URL=$(terraform -chdir=infra/terraform output -raw api_url)
URL=${URL%/}

post() {
  curl -sS --fail-with-body -X POST "$URL/v1/tickets" \
    -H "X-Api-Key: $TRIAGE_API_KEY" -H "Content-Type: application/json" -d "$1" \
    | python3 -c 'import json,sys; t=json.load(sys.stdin); p=t["prediction"]; print(t["ticket_id"], {k: (v["label"], v["level"], round(v["confidence"], 2), v["latency_ms"]//1) for k, v in p.items()}, t["pii_counts"])'
}

post '{"subject": "Foreclosure next week", "body": "My mortgage servicer scheduled a foreclosure sale of my home for next Tuesday even though I sent the loan modification papers. Call me at (555) 123-4567."}'
post '{"subject": "Wrong late fee", "body": "My credit card statement shows a late fee of $35 but I paid on time. Please refund it."}'
post '{"body": "Someone opened accounts in my name and my credit report now shows debts that are not mine. My email is jane.roe@example.com."}'
curl -sS --fail-with-body "$URL/v1/tickets?limit=3" -H "X-Api-Key: $TRIAGE_API_KEY" \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["items"]), "tickets listed")'
