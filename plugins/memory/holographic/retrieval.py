"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        with self.store._lock:
            # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
            candidates = self._fts_candidates(query, category, min_trust, limit * 3)

            if not candidates:
                return []

            # Stage 2: Rerank with Jaccard + HRR + trust + optional decay
            # Note: hrr_vector is still present in each fact dict at this point.
            # It is stripped below after scoring is complete.
            query_tokens = self._tokenize(query)
            scored = []

            for fact in candidates:
                content_tokens = self._tokenize(fact["content"])
                tag_tokens = self._tokenize(fact.get("tags", ""))
                all_tokens = content_tokens | tag_tokens

                jaccard = self._jaccard_similarity(query_tokens, all_tokens)
                fts_score = fact.get("fts_rank", 0.0)

                # HRR similarity
                if self.hrr_weight > 0 and fact.get("hrr_vector"):
                    fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                    hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
                else:
                    hrr_sim = 0.5  # neutral

                # Combine FTS5 + Jaccard + HRR
                relevance = (self.fts_weight * fts_score
                            + self.jaccard_weight * jaccard
                            + self.hrr_weight * hrr_sim)

                # Trust weighting
                score = relevance * fact["trust_score"]

                # Optional temporal decay
                if self.half_life > 0:
                    score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

                fact["score"] = score
                scored.append(fact)

            # Sort by score descending, return top limit
            scored.sort(key=lambda x: x["score"], reverse=True)
            results = scored[:limit]
            # Strip raw HRR bytes — callers expect JSON-serializable dicts
            for fact in results:
                fact.pop("hrr_vector", None)

            return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Unbinds entity from memory bank to extract associated content.
        This is NOT keyword search — it uses algebraic structure to find facts
        where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        with self.store._lock:
            conn = self.store._conn

            # Encode entity as role-bound vector
            role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = hrr.bind(entity_vec, role_entity)

            # Try category-specific bank first, then all facts
            if category:
                bank_name = f"cat:{category}"
                bank_row = conn.execute(
                    "SELECT vector FROM memory_banks WHERE bank_name = ?",
                    (bank_name,),
                ).fetchone()
                if bank_row:
                    bank_vec = hrr.bytes_to_phases(bank_row["vector"])
                    extracted = hrr.unbind(bank_vec, probe_key)
                    # Use extracted signal to score individual facts
                    return self._score_facts_by_vector(
                        extracted, category=category, limit=limit
                    )

            # Score against individual fact vectors directly
            where = "WHERE hrr_vector IS NOT NULL"
            params: list = []
            if category:
                where += " AND category = ?"
                params.append(category)

            rows = conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       hrr_vector
                FROM facts
                {where}
                """,
                params,
            ).fetchall()

            if not rows:
                # Final fallback: keyword search
                return self.search(entity, category=category, limit=limit)

            scored = []
            for row in rows:
                fact = dict(row)
                fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
                # Unbind probe key from fact to see if entity is structurally present
                residual = hrr.unbind(fact_vec, probe_key)
                # Compare residual against content signal
                role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
                content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
                sim = hrr.similarity(residual, content_vec)
                fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
                scored.append(fact)

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        with self.store._lock:
            conn = self.store._conn

            # Encode entity as a bare atom (not role-bound — we want ANY structural match)
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

            # Get all facts with vectors
            where = "WHERE hrr_vector IS NOT NULL"
            params: list = []
            if category:
                where += " AND category = ?"
                params.append(category)

            rows = conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       hrr_vector
                FROM facts
                {where}
                """,
                params,
            ).fetchall()

            if not rows:
                return self.search(entity, category=category, limit=limit)

            # Score each fact by how much the entity's atom appears in its vector
            # This catches both role-bound entity matches AND content word matches
            scored = []
            for row in rows:
                fact = dict(row)
                fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

                # Check structural similarity: unbind entity from fact
                residual = hrr.unbind(fact_vec, entity_vec)
                # A high-similarity residual to ANY known role vector means this entity
                # plays a structural role in the fact
                role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
                role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

                entity_role_sim = hrr.similarity(residual, role_entity)
                content_role_sim = hrr.similarity(residual, role_content)
                # Take the max — entity could appear in either role
                best_sim = max(entity_role_sim, content_role_sim)

                fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
                scored.append(fact)

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        with self.store._lock:
            conn = self.store._conn
            role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

            # For each entity, compute what the bank "remembers" about it
            # by unbinding entity+role from each fact vector
            entity_residuals = []
            for entity in entities:
                entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
                probe_key = hrr.bind(entity_vec, role_entity)
                entity_residuals.append(probe_key)

            # Get all facts with vectors
            where = "WHERE hrr_vector IS NOT NULL"
            params: list = []
            if category:
                where += " AND category = ?"
                params.append(category)

            rows = conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       hrr_vector
                FROM facts
                {where}
                """,
                params,
            ).fetchall()

            if not rows:
                query = " ".join(entities)
                return self.search(query, category=category, limit=limit)

            # Score each fact by how much EACH entity is structurally present.
            # A fact scores high only if ALL entities have structural presence
            # (AND semantics via min, vs OR which would use mean/max).
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

            scored = []
            for row in rows:
                fact = dict(row)
                fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

                entity_scores = []
                for probe_key in entity_residuals:
                    residual = hrr.unbind(fact_vec, probe_key)
                    sim = hrr.similarity(residual, role_content)
                    entity_scores.append(sim)

                min_sim = min(entity_scores)
                fact["score"] = (min_sim + 1.0) / 2.0 * fact["trust_score"]
                scored.append(fact)

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.
        """
        return self.contradiction(category=category, threshold=threshold, limit=limit)

    def contradiction(
        self,
        entity: str,
        category: str | None = None,
        threshold: float = 0.4,
        limit: int = 10,
    ) -> list[dict]:
        """Find facts that contradict each other via shared entity structure.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        with self.store._lock:
            conn = self.store._conn

            # Get all facts with vectors and their linked entities
            where = "WHERE f.hrr_vector IS NOT NULL"
            params: list = []
            if category:
                where += " AND f.category = ?"
                params.append(category)

            rows = conn.execute(
                f"""
                SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                       f.created_at, f.updated_at, f.hrr_vector
                FROM facts f
                {where}
                """,
                params,
            ).fetchall()

            if len(rows) < 2:
                return []

            # Guard against O(n^2) explosion on large fact stores.
            # At 500 facts, that's ~125K comparisons — acceptable.
            # Above that, only check the most recently updated facts.
            _MAX_CONTRADICT_FACTS = 500
            if len(rows) > _MAX_CONTRADICT_FACTS:
                rows.sort(
                    key=lambda r: (
                        r["updated_at"] is not None,
                        r["updated_at"] or r["created_at"] or "",
                    ),
                    reverse=True,
                )
                rows = rows[:_MAX_CONTRADICT_FACTS]

            # Build entity sets per fact
            fact_entities: dict[int, set[str]] = {}
            for row in rows:
                fid = row["fact_id"]
                entity_rows = conn.execute(
                    """
                    SELECT e.name FROM entities e
                    JOIN fact_entities fe ON fe.entity_id = e.entity_id
                    WHERE fe.fact_id = ?
                    """,
                    (fid,),
                ).fetchall()
                fact_entities[fid] = {r["name"].lower() for r in entity_rows}

            facts = [dict(r) for r in rows]
            contradictions = []
            start_time = time.monotonic()
            _MAX_CONTRADICT_SECONDS = 5.0
            timed_out = False

            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    if time.monotonic() - start_time > _MAX_CONTRADICT_SECONDS:
                        timed_out = True
                        break
                    f1, f2 = facts[i], facts[j]
                    ents1 = fact_entities.get(f1["fact_id"], set())
                    ents2 = fact_entities.get(f2["fact_id"], set())
                    if not ents1 or not ents2:
                        continue
                    entity_overlap = (
                        len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0
                    )
                    if entity_overlap < 0.3:
                        continue
                    v1 = hrr.bytes_to_phases(f1["hrr_vector"])
                    v2 = hrr.bytes_to_phases(f2["hrr_vector"])
                    content_sim = hrr.similarity(v1, v2)
                    contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)
                    if contradiction_score >= threshold:
                        f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                        f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                        contradictions.append({
                            "fact_a": f1_clean,
                            "fact_b": f2_clean,
                            "entity_overlap": round(entity_overlap, 3),
                            "content_similarity": round(content_sim, 3),
                            "contradiction_score": round(contradiction_score, 3),
                            "shared_entities": sorted(ents1 & ents2),
                        })
                if timed_out:
                    break

            contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
            return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        with self.store._lock:
            conn = self.store._conn

            where = "WHERE hrr_vector IS NOT NULL"
            params: list = []
            if category:
                where += " AND category = ?"
                params.append(category)

            rows = conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at,
                       hrr_vector
                FROM facts
                {where}
                """,
                params,
            ).fetchall()

            scored = []
            for row in rows:
                fact = dict(row)
                fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
                sim = hrr.similarity(target_vec, fact_vec)
                fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
                scored.append(fact)

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses bm25() scalar function — FTS5's native ranking — to get proper
        relevance scores. bm25() is a signed scalar: lower (more negative) =
        better match. We negate it and normalise to [0, 1] where 1 = best.

        Note: bm25() must appear in ORDER BY, not just SELECT, so the database
        returns the top-k rows by relevance before we re-score in Python.
        """
        conn = self.store._conn

        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        # bm25(facts_fts) returns a negative value per row (more negative = better).
        # Negate to make higher = better, then normalise to [0, 1].
        sql = f"""
            SELECT f.*, bm25(facts_fts) as fts_bm25_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY bm25(facts_fts)
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        raw_scores = [-row["fts_bm25_raw"] for row in rows]  # negate: higher = better
        max_score = max(raw_scores) if raw_scores else 1.0
        max_score = max(max_score, 1e-6)  # avoid div by zero

        results = []
        for row, score in zip(rows, raw_scores):
            fact = dict(row)
            fact.pop("fts_bm25_raw", None)
            # NOTE: hrr_vector is intentionally NOT stripped here.
            # search() needs it for HRR similarity scoring (lines ~88-93).
            # It is stripped only in the final output at search() line ~117.
            fact["fts_rank"] = score / max_score  # normalise to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Punctuation set matches encode_text() in holographic.py exactly.
        No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,!?;:\"'")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
