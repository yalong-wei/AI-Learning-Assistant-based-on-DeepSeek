import os
import pytest

os.environ.setdefault("FLASK_ENV", "testing")

from app import app  # noqa: E402

@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "deepseek_configured" in data
