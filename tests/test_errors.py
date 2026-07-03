import json
import pytest

from app import app  # noqa: E402

@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

def test_intent_predict_invalid_json(client):
    resp = client.post("/api/intent/predict", data="not json", content_type="application/json")
    # Flask returns 400 Bad Request for invalid JSON by default
    assert resp.status_code in (400, 415, 422)
