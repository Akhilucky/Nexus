"""Tool Registry — stores, retrieves, and persists tool metadata.

Uses a JSON file as persistent storage (SQLite → Postgres later as noted in spec).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from nexus.models.schemas import Tool, ToolRegistration

_DEFAULT_STORE = Path(__file__).resolve().parent.parent / "data" / "tools.json"


class ToolRegistry:
    """Thread-safe in-memory registry backed by a JSON file."""

    def __init__(self, store_path: Path | str | None = None) -> None:
        self._store_path = Path(store_path) if store_path else _DEFAULT_STORE
        self._lock = threading.Lock()
        self._tools: dict[str, Tool] = {}  # keyed by tool name
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._store_path.exists():
            with open(self._store_path, "r") as f:
                raw: list[dict] = json.load(f)
            for entry in raw:
                tool = Tool(**entry)
                self._tools[tool.name] = tool

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._store_path, "w") as f:
            json.dump(
                [t.model_dump(mode="json") for t in self._tools.values()],
                f,
                indent=2,
                default=str,
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, payload: ToolRegistration) -> Tool:
        """Register a new tool or update an existing one."""
        with self._lock:
            if payload.name in self._tools:
                existing = self._tools[payload.name]
                updated = existing.model_copy(
                    update=payload.model_dump(exclude_unset=True)
                )
                self._tools[payload.name] = updated
                self._save()
                return updated

            tool = Tool(**payload.model_dump())
            self._tools[tool.name] = tool
            self._save()
            return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                self._save()
                return True
            return False

    def update_tool(self, name: str, **kwargs) -> Optional[Tool]:
        """Partial update of mutable fields (reputation, stats, embedding, etc.)."""
        with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                return None
            updated = tool.model_copy(update=kwargs)
            self._tools[name] = updated
            self._save()
            return updated

    def clear(self) -> None:
        with self._lock:
            self._tools.clear()
            self._save()
