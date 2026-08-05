"""The optional HTTP service exposes the same core over HTTP. Skips if the api extra isn't installed."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient          # noqa: E402

from governor.api import app                        # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_decide_generic():
    assert client.post("/decide", json={"signals": {"x": 0.9}}).json()["decision"] == "escalate"
    assert client.post("/decide", json={"signals": {}}).json()["decision"] == "auto_send"
    # load-shed in the escalate band when the human is saturated
    r = client.post("/decide", json={"signals": {"x": 0.6}, "human_queue_depth": 10})
    assert r.json()["decision"] == "hold"


def test_govern_recruiting_end_to_end():
    # competitor company -> escalate; clean non-competitor low-risk -> auto_send
    esc = client.post("/govern", json={
        "candidate": {"name": "A", "current_company": "Anthropic", "seniority": "senior",
                      "match_confidence": 0.9},
        "subject": "hi", "body": "Would you be open to a quick chat about a role?",
        "draft_confidence": 0.9}).json()
    assert esc["decision"] == "escalate"
    ok = client.post("/govern", json={
        "candidate": {"name": "B", "current_company": "Stripe", "seniority": "mid",
                      "match_confidence": 0.85},
        "subject": "hi", "body": "Would you be open to a quick chat about a role?",
        "draft_confidence": 0.9}).json()
    assert ok["decision"] == "auto_send"


def test_scoreboard_endpoint():
    j = client.get("/scoreboard").json()
    assert j["dangerous_auto_sends"] == 0 and j["recall"] == 1.0
