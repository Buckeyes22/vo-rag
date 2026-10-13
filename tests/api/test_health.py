from fastapi.testclient import TestClient

from vo_rag.api.main import app


def test_health_ok():
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "vo-rag"}
