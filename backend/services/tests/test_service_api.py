import datetime as dt
import io
import json

from triage_services.emf import EmfLogger
from triage_services.handler import build_deps

PII_BODY = "Someone used card 4111 1111 1111 1111 and wrote from thief@evil.com"


def events(api):
    return [json.loads(x) for x in api.emf.getvalue().splitlines()]


def test_health_needs_no_key(api):
    assert api.client.get("/health").json() == {"status": "ok"}


def test_missing_or_wrong_key_is_401(api):
    for headers in ({}, {"X-Api-Key": "tt_cfpb-demo_wrong"}):
        resp = api.client.get("/v1/tickets", headers=headers)
        assert resp.status_code == 401
        assert resp.json() == {
            "error": {"code": "unauthorized", "message": "missing or invalid API key"}
        }


def test_create_classifies_redacts_and_stores(api):
    resp = api.client.post(
        "/v1/tickets", headers=api.headers, json={"subject": "Mortgage escrow", "body": PII_BODY}
    )
    assert resp.status_code == 201
    ticket = resp.json()
    assert ticket["tenant_id"] == "cfpb-demo"
    assert ticket["received_at"] == "2026-11-24T10:00:00.000Z"
    assert ticket["pii_counts"] == {"card": 1, "email": 1}
    assert "4111" not in ticket["body_redacted"] and "thief" not in ticket["body_redacted"]
    assert ticket["prediction"]["category"]["label"] == "mortgage"  # keyword fake
    assert ticket["prediction"]["category"]["level"] in ("jev", "claude")
    assert ticket["final_category"] == "mortgage"
    stored = api.client.get(f"/v1/tickets/{ticket['ticket_id']}", headers=api.headers).json()
    assert stored == ticket


def test_emf_event_has_metrics_and_no_text(api):
    api.client.post("/v1/tickets", headers=api.headers, json={"subject": "Hi", "body": PII_BODY})
    [event] = events(api)
    assert event["event"] == "ticket_classified"
    assert (event["TicketsProcessed"], event["PiiRedacted"], event["Unclassified"]) == (1, 2, 0)
    assert event["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [[]]
    raw = api.emf.getvalue()
    assert "4111" not in raw and "thief" not in raw and "Hi" not in raw


def test_list_is_newest_first_with_limit(api):
    ids = []
    for i in range(3):
        api.deps.clock = lambda i=i: dt.datetime(2026, 11, 24, 10, i, tzinfo=dt.UTC)
        r = api.client.post("/v1/tickets", headers=api.headers, json={"body": f"ticket {i}"})
        ids.append(r.json()["ticket_id"])
    items = api.client.get("/v1/tickets?limit=2", headers=api.headers).json()["items"]
    assert [t["ticket_id"] for t in items] == [ids[2], ids[1]]
    assert api.client.get("/v1/tickets?limit=0", headers=api.headers).status_code == 422


def test_validation_and_not_found_use_error_body(api):
    resp = api.client.post("/v1/tickets", headers=api.headers, json={"body": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    resp = api.client.get("/v1/tickets/nope", headers=api.headers)
    assert resp.json() == {"error": {"code": "not_found", "message": "ticket not found"}}


def test_unexpected_error_is_500_and_counted(api, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("dynamo down")

    monkeypatch.setattr(api.deps.store, "list_recent", boom)
    resp = api.client.get("/v1/tickets", headers=api.headers)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
    assert events(api)[-1]["Errors"] == 1


def test_all_levels_down_still_stores_unclassified(api):
    class Broken:
        name, cache_key = "broken", "broken"

        def predict(self, texts, labels):
            raise ConnectionError("down")

    for engine in api.deps.engines.values():
        engine.cascade.levels[:] = [(Broken(), None)]
    resp = api.client.post("/v1/tickets", headers=api.headers, json={"body": "x"})
    assert resp.status_code == 201
    t = resp.json()
    assert t["final_category"] is None and t["prediction"]["priority"]["level"] == "none"
    assert events(api)[-1]["Unclassified"] == 1


def test_mangum_function_url_event(api, settings, table):
    from mangum import Mangum

    from triage_services.app import create_app

    deps = build_deps(settings)
    deps.emf = EmfLogger(io.StringIO())
    handler = Mangum(create_app(lambda: deps), lifespan="off")
    event = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": "/v1/tickets",
        "rawQueryString": "",
        "headers": {
            "host": "x.lambda-url.us-east-1.on.aws",
            "x-api-key": "tt_cfpb-demo_test-secret",
        },
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/v1/tickets",
                "protocol": "HTTP/1.1",
                "sourceIp": "1.2.3.4",
                "userAgent": "pytest",
            },
            "requestId": "r",
            "routeKey": "$default",
            "stage": "$default",
            "domainName": "x.lambda-url.us-east-1.on.aws",
            "apiId": "x",
            "accountId": "anonymous",
            "time": "24/Nov/2026:10:00:00 +0000",
            "timeEpoch": 0,
        },
        "isBase64Encoded": False,
    }
    resp = handler(event, None)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"]) == {"items": []}


def test_index_page_is_served_without_secrets(api):
    resp = api.client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text
    assert "Ticket Triage" in html
    assert "__LABELS__" not in html
    assert '"mortgage": "Mortgage' in html or '"mortgage":' in html
    assert "tt_cfpb-demo_test-secret" not in html  # the key is typed by the user, never embedded
