"""Policy guardrails for safe and compliant tool routing."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.models.schemas import RouteRequest, SecurityLevel, Tool

_SECURITY_ORDER = {
    SecurityLevel.public: 0,
    SecurityLevel.internal: 1,
    SecurityLevel.confidential: 2,
    SecurityLevel.restricted: 3,
}


@dataclass
class PolicyResult:
    allowed_tools: list[Tool]
    trace: dict[str, int | list[str] | str]


class PolicyGuardrails:
    """Applies deterministic policy checks before scoring tools."""

    def filter_tools(self, tools: list[Tool], request: RouteRequest) -> PolicyResult:
        allowed = tools
        trace: dict[str, int | list[str] | str] = {
            "initial_candidates": len(tools),
            "filtered_by_clearance": 0,
            "filtered_by_blocked_tags": 0,
            "filtered_by_allowed_tools": 0,
            "filtered_by_reliability": 0,
            "final_candidates": 0,
        }

        # 1) Security clearance
        max_level = _SECURITY_ORDER[request.security_clearance]
        next_allowed: list[Tool] = []
        for tool in allowed:
            if _SECURITY_ORDER[tool.security_level] <= max_level:
                next_allowed.append(tool)
            else:
                trace["filtered_by_clearance"] = int(trace["filtered_by_clearance"]) + 1
        allowed = next_allowed

        # 2) Blocked tags
        if request.blocked_tags:
            blocked = {t.lower() for t in request.blocked_tags}
            next_allowed = []
            for tool in allowed:
                tags = {t.lower() for t in tool.tags}
                if blocked & tags:
                    trace["filtered_by_blocked_tags"] = int(trace["filtered_by_blocked_tags"]) + 1
                else:
                    next_allowed.append(tool)
            allowed = next_allowed

        # 3) Allow-list names
        if request.allowed_tools:
            allow = {name.lower() for name in request.allowed_tools}
            next_allowed = []
            for tool in allowed:
                if tool.name.lower() in allow:
                    next_allowed.append(tool)
                else:
                    trace["filtered_by_allowed_tools"] = int(trace["filtered_by_allowed_tools"]) + 1
            allowed = next_allowed

        # 4) Min reliability threshold
        if request.min_reliability > 0:
            next_allowed = []
            for tool in allowed:
                if tool.reliability >= request.min_reliability:
                    next_allowed.append(tool)
                else:
                    trace["filtered_by_reliability"] = int(trace["filtered_by_reliability"]) + 1
            allowed = next_allowed

        trace["final_candidates"] = len(allowed)
        return PolicyResult(allowed_tools=allowed, trace=trace)
