# NEXUS — LLM Decision Router

A control plane for tool-using AI systems. NEXUS registers tools with real-world metadata, classifies intent, scores candidates deterministically, routes requests, and learns from outcomes.

## Features
- Tool Registry with JSON persistence
- Embedding-based Intent Classifier (sentence-transformers) with keyword fallback
- Weighted Decision Engine (semantic match, reliability, latency, cost, reputation)
- Policy Guardrails (security clearance, blocked tags, allow-list, min reliability)
- Router with telemetry logging and feedback ingestion
- Memory/Reputation loop with reinforcement and recency weighting
- Admin Self-Check endpoint for logical health diagnostics
- Risk Intelligence endpoint ranking tools by failure trend + latency drift
- FastAPI REST API + Python SDK client
- 3 demo tools preloaded
- 55 tests covering components and end-to-end flows

## Quick Start
```bash
# Install
pip install -e ".[dev]"

# Run API
uvicorn nexus.app.main:app --reload

# Run tests
python3 -m pytest nexus/tests -v
```

## API Surface
- POST /tools/register — register or update a tool
- GET /tools — list tools
- GET /tools/{name} — fetch a tool
- DELETE /tools/{name} — remove a tool
- POST /route — route a query to the best tool
- POST /feedback — submit execution result (updates telemetry + reputation)
- GET /metrics — system metrics
- GET /metrics/top-risks — ranked operational risk view for proactive mitigation
- POST /admin/recalculate — recompute reputations from history
- GET /admin/self-check — detect logical/operational issues early

## Repository Layout
- nexus/app — core services (registry, intent, decision, router, telemetry, memory, FastAPI app)
- nexus/models — Pydantic schemas
- nexus/sdk — Python SDK client
- nexus/data/tools.json — sample tools
- nexus/tests — component + API tests

## Demo Query
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Find anomalies in Q4 revenue"}'
```
