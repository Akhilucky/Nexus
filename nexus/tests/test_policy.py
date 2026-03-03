"""Tests for policy guardrails."""

from nexus.app.policy import PolicyGuardrails
from nexus.models.schemas import RouteRequest, SecurityLevel, Tool


def _tool(name: str, *, security: SecurityLevel, tags: list[str], reliability: float = 0.9) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        input_schema={},
        latency_ms=100,
        cost=0.01,
        reliability=reliability,
        security_level=security,
        tags=tags,
    )


class TestPolicyGuardrails:
    def test_security_clearance_filters_higher_tools(self):
        guardrails = PolicyGuardrails()
        tools = [
            _tool("pub", security=SecurityLevel.public, tags=[]),
            _tool("conf", security=SecurityLevel.confidential, tags=[]),
        ]
        req = RouteRequest(query="x", security_clearance=SecurityLevel.internal)
        result = guardrails.filter_tools(tools, req)

        names = [t.name for t in result.allowed_tools]
        assert "pub" in names
        assert "conf" not in names

    def test_blocked_tags_and_reliability(self):
        guardrails = PolicyGuardrails()
        tools = [
            _tool("safe", security=SecurityLevel.internal, tags=["data"], reliability=0.9),
            _tool("blocked", security=SecurityLevel.internal, tags=["unsafe"], reliability=0.9),
            _tool("lowrel", security=SecurityLevel.internal, tags=["data"], reliability=0.3),
        ]
        req = RouteRequest(
            query="x",
            blocked_tags=["unsafe"],
            min_reliability=0.8,
        )
        result = guardrails.filter_tools(tools, req)

        names = [t.name for t in result.allowed_tools]
        assert names == ["safe"]
        assert result.trace["filtered_by_blocked_tags"] == 1
        assert result.trace["filtered_by_reliability"] == 1

    def test_allowed_tools_whitelist(self):
        guardrails = PolicyGuardrails()
        tools = [
            _tool("a", security=SecurityLevel.internal, tags=[]),
            _tool("b", security=SecurityLevel.internal, tags=[]),
        ]
        req = RouteRequest(query="x", allowed_tools=["b"])
        result = guardrails.filter_tools(tools, req)

        names = [t.name for t in result.allowed_tools]
        assert names == ["b"]
