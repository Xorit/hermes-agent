"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "description": "Fact category (any non-empty string, e.g. 'user_pref', 'project', 'tool', 'general')"},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("hermes-memory-store", {}) or {}
    except yaml.YAMLError as e:
        logger.warning("holographic-memory-store config is not valid YAML — skipping plugin config: %s", e)
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.warning("Failed to save hermes-memory-store config: %s", e)

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
            {"key": "hrr_weight", "description": "HRR semantic similarity weight (0.0-1.0)", "default": "0.3"},
            {"key": "fts_weight", "description": "FTS5 full-text search weight (0.0-1.0)", "default": "0.4"},
            {"key": "jaccard_weight", "description": "Token-overlap reranking weight (0.0-1.0)", "default": "0.3"},
            {"key": "temporal_decay_half_life", "description": "Days for fact relevance half-life (0=disabled)", "default": "0"},
            {"key": "min_trust_threshold", "description": "Minimum trust score for search/probe results (0.0-1.0)", "default": "0.3"},
            {"key": "contradict_check", "description": "Run contradiction check at session end", "default": "true", "choices": ["true", "false"]},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        # Idempotent: keep existing store if it's still healthy.
        # Closing and recreating on every session start causes WAL replay issues
        # and "file is not a database" cascading failures (see Apr 21 incident).
        if self._store is not None:
            try:
                with self._store._lock:
                    self._store._conn.execute("SELECT 1").fetchone()
                # Store is healthy — just update session_id and reuse.
                self._session_id = session_id
                return
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning("holographic: existing store unhealthy, reinitializing")
                try:
                    self._store.close()
                except Exception:
                    pass
                self._store = None
                self._retriever = None

        import logging
        logger = logging.getLogger(__name__)

        try:
            from hermes_constants import get_hermes_home
            _hermes_home = str(get_hermes_home())
            _default_db = _hermes_home + "/memory_store.db"
            db_path = self._config.get("db_path", _default_db)
            # Expand $HERMES_HOME in user-supplied paths so config values like
            # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
            # resolve to the active profile's directory.
            if isinstance(db_path, str):
                db_path = db_path.replace("$HERMES_HOME", _hermes_home)
                db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
            default_trust = float(self._config.get("default_trust", 0.5))
            hrr_dim = int(self._config.get("hrr_dim", 1024))
            hrr_weight = float(self._config.get("hrr_weight", 0.3))
            fts_weight = float(self._config.get("fts_weight", 0.4))
            jaccard_weight = float(self._config.get("jaccard_weight", 0.3))
            temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

            self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
            self._retriever = FactRetriever(
                store=self._store,
                temporal_decay_half_life=temporal_decay,
                fts_weight=fts_weight,
                jaccard_weight=jaccard_weight,
                hrr_weight=hrr_weight,
                hrr_dim=hrr_dim,
            )
            self._session_id = session_id
        except Exception as e:
            logger.error("Holographic memory initialize failed: %s. Facts will be unavailable until the gateway restarts.", e)
            # Leave _store as None so handle_tool_call returns a clean error
            self._store = None
            self._retriever = None
            raise

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            with self._store._lock:
                total = self._store._conn.execute(
                    "SELECT COUNT(*) FROM facts"
                ).fetchone()[0]
        except Exception:
            total = 0

        lines = []

        # Contradictions — surfaced automatically so the agent can reason about conflicts
        if self._config.get("contradict_check", True):
            try:
                conflicts = self._retriever.contradict(limit=3)
                if conflicts:
                    lines.append("⚠️  Contradictions detected in stored facts:")
                    for c in conflicts:
                        a_content = c["fact_a"]["content"]
                        b_content = c["fact_b"]["content"]
                        a_short = a_content[:77] + "..." if len(a_content) > 80 else a_content
                        b_short = b_content[:77] + "..." if len(b_content) > 80 else b_content
                        lines.append(
                            f"  • \"{a_short}\" ↔ \"{b_short}\""
                            f" (shared: {', '.join(c['shared_entities'])}, score={c['contradiction_score']:.2f})"
                        )
                    lines.append("  Use fact_store(action='contradict') for full details.")
                    lines.append("")
            except Exception:
                pass

        if total == 0:
            lines.append(
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        else:
            lines.append(
                f"# Holographic Memory\n"
                f"Active. {total} facts stored with entity resolution and trust scoring.\n"
                f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
                f"Use fact_feedback to rate facts after using them (trains trust scores)."
            )

        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Holographic memory stores explicit facts via tools, not auto-sync.
        # The on_session_end hook handles auto-extraction if configured.
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "fact_store":
                return self._handle_fact_store(args)
            elif tool_name == "fact_feedback":
                return self._handle_fact_feedback(args)
            return tool_error(f"Unknown tool: {tool_name}")
        except AttributeError as e:
            if "'NoneType' object" in str(e):
                return tool_error("Holographic memory is not initialized. The database may be missing or corrupted. Try deleting $HERMES_HOME/memory_store.db and restarting the gateway.", success=False)
            return tool_error(f"Memory error: {e}")
        except Exception as e:
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._store or not messages:
            return

        # Phase 1: auto-extract new facts from conversation
        if self._config.get("auto_extract", False):
            self._auto_extract_facts(messages)

        # Phase 2: contradiction check — silent hygiene, logs only
        if self._config.get("contradict_check", True):
            try:
                conflicts = self._retriever.contradict(limit=5)
                if conflicts:
                    for conflict in conflicts:
                        logger.info(
                            "[memory] Contradiction detected (score=%.3f): %s vs %s",
                            conflict["contradiction_score"],
                            conflict["fact_a"]["content"][:80],
                            conflict["fact_b"]["content"][:80],
                        )
                    logger.info("[memory] %d contradiction(s) found — review with fact_store(action='contradict')", len(conflicts))
            except Exception as e:
                logger.debug("[memory] Contradiction check failed: %s", e)

        # Phase 3: snapshot facts to JSON for disaster recovery
        self._snapshot_facts_to_json()

    def on_memory_write(
        self, action: str, target: str, content: str, *, new_content: str | None = None
    ) -> None:
        """Mirror built-in memory writes as facts.

        Mirrors add, remove, and replace actions:
        - add: insert new fact
        - remove: find fact by content substring and delete it
        - replace: find fact by old_text substring, delete it, insert new_content
        """
        if not self._store or not content:
            return
        try:
            category = "user_pref" if target == "user" else "general"

            if action == "add":
                self._store.add_fact(content, category=category)
                self._snapshot_facts_to_json()

            elif action == "remove":
                rows = self._store._conn.execute(
                    "SELECT fact_id FROM facts WHERE content LIKE ?",
                    (f"%{content}%",),
                ).fetchall()
                for row in rows:
                    self._store.remove_fact(row["fact_id"])
                self._snapshot_facts_to_json()

            elif action == "replace":
                if not new_content or not str(new_content).strip():
                    logger.debug("on_memory_write replace skipped: missing new_content")
                    return

                rows = self._store._conn.execute(
                    "SELECT fact_id, category FROM facts WHERE content LIKE ?",
                    (f"%{content}%",),
                ).fetchall()
                if len(rows) > 1:
                    logger.debug(
                        "on_memory_write replace: %d facts match '%s...' — "
                        "re-adding replacement before removing old entries",
                        len(rows),
                        content[:50],
                    )

                # Hold the store lock for the entire replace operation so
                # add_fact (which also acquires the lock) cannot deadlock.
                with self._store._lock:
                    # Add replacement first so a failed insert never deletes old facts.
                    self._store.add_fact(new_content, category=category)

                    if not rows:
                        return

                    affected_categories = {
                        str(row["category"])
                        for row in rows
                        if row["category"] is not None
                    }

                    self._store._conn.execute("SAVEPOINT memory_write_replace")
                    try:
                        for row in rows:
                            self._store._conn.execute(
                                "DELETE FROM fact_entities WHERE fact_id = ?",
                                (row["fact_id"],),
                            )
                            self._store._conn.execute(
                                "DELETE FROM facts WHERE fact_id = ?",
                                (row["fact_id"],),
                            )
                        self._store._conn.execute("RELEASE SAVEPOINT memory_write_replace")
                    except Exception:
                        self._store._conn.execute("ROLLBACK TO SAVEPOINT memory_write_replace")
                        self._store._conn.execute("RELEASE SAVEPOINT memory_write_replace")
                        raise

                    # Rebuild all banks touched by deleted rows since we used raw
                    # SQL (bypassing remove_fact which rebuilds automatically).
                    for affected_category in affected_categories:
                        self._store._rebuild_bank(affected_category)

                self._snapshot_facts_to_json()

        except Exception as e:
            logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception as exc:
                logger.debug("HolographicMemoryProvider.close() raised during shutdown: %s", exc)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        if self._store is None:
            return tool_error("Holographic memory is not initialized. The database may be missing or corrupted. Try deleting $HERMES_HOME/memory_store.db and restarting the gateway.", success=False)
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                self._snapshot_facts_to_json()
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    entity=args.get("entity"),
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                self._snapshot_facts_to_json()
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                self._snapshot_facts_to_json()
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need|expect)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually|typically)\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(?:call|named?)\s+(?:it|this)\s+(.+?)(?:\s+which|\s+that|$)', re.IGNORECASE),
            re.compile(r'\b(?:just|simply|basically)\s+(?:use|do|call)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\b(?:we|I|they?|he|she|it)\s+(?:decided|agreed|chose|resolved)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\b(?:the|our|this)\s+(?:project|solution|setup|setup|approach)\s+(?:uses?|needs?|requires?|went with)\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(?:we|I)\s+(?:end up|typically end up|usually end up)\s+(?:using|doing|calling)\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(?:instead of|rather than)\s+(?:using|doing|calling)\s+(.+?)(?:\s+we\s+|\s+I\s+|$)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)

    def _snapshot_facts_to_json(self) -> None:
        """Snapshot all facts, entities, and fact_entities to a JSON recovery file.

        Called on every significant write (add/remove) and at session end.
        This is the primary disaster-recovery mechanism — if the SQLite DB is
        corrupted or lost, facts can be restored from this JSON using the
        /root/.hermes/scripts/fact_backup.sh restore command or a simple
        Python script reading the JSON.

        The snapshot is stored at:
          $HERMES_HOME/memory/memory_store_snapshot.json

        A timestamped copy is NOT written per-snapshot (that is handled by the
        cron-triggered fact_backup.sh which also snapshots DB+WAL+SHM). This
        method focuses on always having a current recovery file.
        """
        import json
        from datetime import datetime
        from pathlib import Path

        if not self._store:
            return

        try:
            from hermes_constants import get_hermes_home
            snap_base = get_hermes_home() / "memory"
            snap_base.mkdir(parents=True, exist_ok=True)
            snap_path = snap_base / "memory_store_snapshot.json"
        except Exception:
            return

        try:
            conn = self._store._conn
            with self._store._lock:
                facts = conn.execute(
                    "SELECT fact_id, content, category, tags, trust_score, "
                    "       retrieval_count, helpful_count, created_at, updated_at "
                    "FROM facts ORDER BY fact_id"
                ).fetchall()
                entities = conn.execute(
                    "SELECT entity_id, name, entity_type, aliases, created_at "
                    "FROM entities ORDER BY entity_id"
                ).fetchall()
                fact_entities = conn.execute(
                    "SELECT fact_id, entity_id FROM fact_entities"
                ).fetchall()

            snapshot = {
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "db_path": str(self._store.db_path),
                "facts": [dict(r) for r in facts],
                "entities": [dict(r) for r in entities],
                "fact_entities": [dict(r) for r in fact_entities],
                "counts": {
                    "facts": len(facts),
                    "entities": len(entities),
                    "fact_entities": len(fact_entities),
                },
            }

            # Atomic write: write to .tmp then rename
            tmp_path = snap_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
            tmp_path.replace(snap_path)
            logger.debug(
                "holographic: snapshot saved (%d facts, %d entities) -> %s",
                len(facts), len(entities), snap_path,
            )
        except Exception as e:
            logger.debug("holographic: snapshot failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
