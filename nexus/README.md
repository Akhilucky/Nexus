# NEXUS — LLM Decision Router

A **control plane for tool-using AI systems**. Instead of letting an LLM randomly choose APIs, databases, or scripts, NEXUS decides — deterministically and intelligently — what should be used, when, and why.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
uvicorn nexus.app.main:app --reload

# Run tests
pytest
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/tools/register` | Register or update a tool |
| GET | `/tools` | List all registered tools |
| GET | `/tools/{name}` | Get a specific tool |
| DELETE | `/tools/{name}` | Remove a tool |
| POST | `/route` | Route a user query to the best tool |
| POST | `/feedback` | Report execution outcome |
| GET | `/metrics` | System-wide telemetry metrics |
| POST | `/admin/recalculate` | Recompute all reputations |
| GET | `/admin/self-check` | Run logical/operational diagnostics |

## Architecture

```
User Request
     ↓
NEXUS Router
 ├── Intent Classifier (embedding-based)
 ├── Tool Registry (JSON-backed)
 ├── Decision Engine (weighted scoring)
 ├── Policy Guardrails
 └── Telemetry + Memory (learning loop)
     ↓
Selected Tool → Response
```

## Decision Scoring

```
score =
  semantic_match * 0.40
+ reliability     * 0.20
+ latency_weight  * 0.15
+ cost_weight     * 0.15
+ past_success    * 0.10
```

## Demo

Three sample tools are pre-registered in `nexus/data/tools.json`:

| Tool | Purpose | Latency | Cost |
|------|---------|---------|------|
| warehouse_api | Accurate but slow | 800ms | High |
| cache_layer | Fast but partial | 40ms | Low |
| analytics_script | Compute heavy | 450ms | Medium |

```bash
# Register a tool
curl -X POST http://localhost:8000/tools/register \
  -H "Content-Type: application/json" \
  -d '{"name": "my_tool", "description": "Does something useful", "tags": ["demo"]}'

# Route a query
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Find anomalies in Q4 revenue", "security_clearance": "internal", "min_reliability": 0.7}'
```
