"""End-to-end tests for the FastAPI application."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a test client with isolated data files."""
    tmp = tmp_path_factory.mktemp("nexus_e2e")

    # Patch default store paths before importing app
    import nexus.app.registry as reg_mod
    import nexus.app.telemetry as tel_mod

    reg_mod._DEFAULT_STORE = tmp / "tools.json"
    tel_mod._DEFAULT_LOG = tmp / "telemetry.jsonl"

    from nexus.app.main import app

    with TestClient(app) as c:
        yield c


class TestAPI:
    def test_health(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_register_tool(self, client: TestClient):
        r = client.post("/tools/register", json={
            "name": "test_api_tool",
            "description": "A tool for testing",
            "latency_ms": 100,
            "cost": 0.01,
            "reliability": 0.9,
            "tags": ["test"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "test_api_tool"

    def test_list_tools(self, client: TestClient):
        r = client.get("/tools")
        assert r.status_code == 200
        tools = r.json()
        assert any(t["name"] == "test_api_tool" for t in tools)

    def test_get_tool(self, client: TestClient):
        r = client.get("/tools/test_api_tool")
        assert r.status_code == 200
        assert r.json()["name"] == "test_api_tool"

    def test_get_tool_404(self, client: TestClient):
        r = client.get("/tools/nonexistent")
        assert r.status_code == 404

    def test_route_request(self, client: TestClient):
        # Register a couple of tools first
        client.post("/tools/register", json={
            "name": "data_fetcher",
            "description": "Fetch data from database",
            "latency_ms": 200,
            "cost": 0.01,
            "reliability": 0.95,
            "tags": ["data"],
        })
        client.post("/tools/register", json={
            "name": "chart_maker",
            "description": "Create charts and visualizations",
            "latency_ms": 300,
            "cost": 0.02,
            "reliability": 0.9,
            "tags": ["visualization"],
        })

        r = client.post("/route", json={"query": "fetch revenue data"})
        assert r.status_code == 200
        data = r.json()
        assert data["selected_tool"] != "none"
        assert 0.0 <= data["confidence"] <= 1.0
        assert "request_id" in data

    def test_route_and_feedback_loop(self, client: TestClient):
        r = client.post("/route", json={"query": "get data quickly"})
        assert r.status_code == 200
        request_id = r.json()["request_id"]

        fb = client.post("/feedback", json={
            "request_id": request_id,
            "success": True,
            "latency_ms": 45.0,
            "user_satisfaction": 0.9,
        })
        assert fb.status_code == 200
        assert fb.json()["status"] == "recorded"

    def test_metrics(self, client: TestClient):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "total_routes" in data
        assert "tools" in data

    def test_top_risks_endpoint(self, client: TestClient):
        # Create telemetry for at least one tool
        route = client.post("/route", json={"query": "fetch revenue data"})
        assert route.status_code == 200
        req_id = route.json()["request_id"]
        client.post("/feedback", json={
            "request_id": req_id,
            "success": False,
            "latency_ms": 280.0,
        })

        r = client.get("/metrics/top-risks?limit=3&window=5")
        assert r.status_code == 200
        data = r.json()
        assert data["limit"] == 3
        assert data["window"] == 5
        assert "risks" in data
        assert isinstance(data["risks"], list)

    def test_delete_tool(self, client: TestClient):
        client.post("/tools/register", json={
            "name": "to_delete",
            "description": "will be deleted",
        })
        r = client.delete("/tools/to_delete")
        assert r.status_code == 200
        assert client.get("/tools/to_delete").status_code == 404

    def test_recalculate(self, client: TestClient):
        r = client.post("/admin/recalculate")
        assert r.status_code == 200
        assert "reputations" in r.json()

    def test_self_check_endpoint(self, client: TestClient):
        r = client.get("/admin/self-check")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "issues" in data

    def test_route_respects_security_clearance(self, client: TestClient):
        client.post("/tools/register", json={
            "name": "restricted_ops_tool",
            "description": "Runs restricted operations",
            "security_level": "restricted",
            "latency_ms": 10,
            "cost": 0.001,
            "reliability": 0.99,
            "tags": ["ops"],
        })
        r = client.post("/route", json={
            "query": "run operations",
            "security_clearance": "internal",
            "allowed_tools": ["restricted_ops_tool"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["selected_tool"] == "none"
        assert data["policy_trace"]["filtered_by_clearance"] >= 1
