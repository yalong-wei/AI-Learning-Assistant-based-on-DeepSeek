import json
import types
import pytest

from app import app, get_intent_model  # noqa: E402

class DummyModel:
    classes_ = ["示例代码", "概念解释"]
    def predict(self, X):
        return ["示例代码"]
    def predict_proba(self, X):
        return [[0.8, 0.2]]

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "dummy://model")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "dummy://mlflow")
    monkeypatch.setattr("app.mlflow.sklearn.load_model", lambda uri: DummyModel())

@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

def test_intent_predict_ok(client):
    resp = client.post("/api/intent/predict", data=json.dumps({"text": "给个sklearn逻辑回归示例"}), content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["label"] == "示例代码"
    assert "topk" in data and isinstance(data["topk"], list)

def test_intent_predict_missing_text(client):
    resp = client.post("/api/intent/predict", data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 400


def test_intent_predict_long_text(client):
    long_text = "a" * 5000
    resp = client.post("/api/intent/predict", data=json.dumps({"text": long_text}), content_type="application/json")
    assert resp.status_code == 200
