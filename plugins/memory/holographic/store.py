"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import contextlib
import logging
import re
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# HRR constants
_SNR_MIN         =  3.0   # minimum SNR before bundling; below this = noise
_PAGE_SIZE       =  4096  # set before journal_mode=WAL

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL COLLATE NOCASE,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


def _sqlite_error_message(exc: BaseException) -> str:
    return str(exc).lower()


def _is_lock_or_busy_error(exc: BaseException) -> bool:
    msg = _sqlite_error_message(exc)
    return (
        "database is locked" in msg
        or "database table is locked" in msg
        or "database schema is locked" in msg
        or "database is busy" in msg
        or "locked" in msg
        or "busy" in msg
    )


def _is_known_corruption_error(exc: BaseException) -> bool:
    msg = _sqlite_error_message(exc)
    return (
        "file is not a database" in msg
        or "database disk image is malformed" in msg
        or "malformed" in msg
        or "unsupported file format" in msg
        or "not a database" in msg
        or "disk I/O error" in msg
    )


def _wal_sidecar_paths(db_path: Path) -> tuple[Path, Path]:
    # Do not use with_suffix("-wal"): it raises on paths with no extension.
    base = str(db_path)
    return Path(base + "-wal"), Path(base + "-shm")


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        logger.info("holographic: opened database at %s", self.db_path)
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row

        # Online backup thread state
        self._online_backup_dir = self.db_path.parent / "memory_store_backups" / "online"
        self._online_backup_dir.mkdir(parents=True, exist_ok=True)
        self._backup_running = False
        self._backup_thread: threading.Thread | None = None
        self._backup_interval = 300  # 5 minutes

        # Initialise schema and run post-WAL integrity check.
        # Before any destructive recovery: snapshot facts to JSON as a safety net.
        try:
            self._conn.execute("PRAGMA integrity_check").fetchone()
            self._init_db()
            self._try_snapshot()
            self._startup_integrity_check()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            # Do not delete DB files for lock/busy contention — that is transient.
            if _is_lock_or_busy_error(exc):
                raise
            if not _is_known_corruption_error(exc):
                raise

            # SAFETY NET: snapshot before destructive recovery
            self._try_snapshot()

            logger.warning(
                "holographic: existing memory_store file appears corrupt (%s) — "
                "archiving and recreating. All stored facts will be lost.",
                exc,
            )
            self._conn.close()
            try:
                # Rename instead of delete — preserves the corrupt file for forensics
                import time as _time
                _ts = int(_time.time())
                _archive_path = self.db_path.parent / (self.db_path.name + f".corrupted.{_ts}")
                logger.info("holographic: archiving corrupted database %s → %s", self.db_path, _archive_path)
                self.db_path.rename(_archive_path)
                # Also archive sidecar files
                for sidecar in _wal_sidecar_paths(self.db_path):
                    if sidecar.exists():
                        _sidecar_archive = sidecar.parent / (sidecar.name + f".corrupted.{_ts}")
                        logger.info("holographic: archiving orphaned sidecar %s → %s", sidecar, _sidecar_archive)
                        try:
                            sidecar.rename(_sidecar_archive)
                        except Exception:
                            sidecar.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

            # Reconnect and try initialization again (one attempt only)
            logger.info("holographic: re-establishing connection to %s after recovery", self.db_path)
            self._conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_db()
            self._startup_integrity_check()
            logger.info("holographic: initialization complete after recovery")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist.

        Uses WAL journal mode for concurrent read/write access. WAL allows
        concurrent readers without blocking writers, and writers don't block
        readers — essential for a multi-threaded/multi-platform gateway.

        Safety properties:
        - page_size=4096: set BEFORE journal_mode=WAL; required for WAL header.
        - synchronous=FULL: fsync after every WAL frame write. Eliminates
          crash-during-write corruption at ~2x write latency cost. Worth it
          for a single-user memory store where every fact matters.
        - locking_mode=NORMAL: non-exclusive locking; other connections read.
        - timeout=10s: prevents "database locked" on contention.
        - RLock: serialises write access within this process.

        WAL-specific cleanup:
        - wal_checkpoint(TRUNCATE) on close: prevents orphaned -wal/-shm segments
          after a crash, keeping the DB as a single file for simple backup/restore.
        """
        # page_size MUST be set before journal_mode; changing it later is a no-op.
        self._conn.execute(f"PRAGMA page_size={_PAGE_SIZE}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA locking_mode=NORMAL")
        logger.debug("holographic: configured database (WAL, synchronous=FULL, page_size=%d)", _PAGE_SIZE)
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        self._conn.commit()

    def _startup_integrity_check(self) -> None:
        """Run a quick integrity check on the DB file after WAL recovery.

        If the previous shutdown was dirty (hard crash, SIGKILL, OOM),
        SQLite replays the WAL on open.  After replay, we verify the
        result with PRAGMA integrity_check.  On failure, we delete the
        orphaned WAL/SHM files, close the connection, and re-open —
        SQLite will then treat the DB as a clean, WAL-free database.

        This is safe because all real data lives in the main DB file.
        The WAL is just uncommitted redo log.  Discarding it means
        rolling back the last few transactions — acceptable trade-off
        for a memory store where recent facts may be re-learned.
        """
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] == "ok":
                logger.info("holographic: startup integrity check OK")
        except sqlite3.OperationalError as exc:
            # Lock/busy is transient contention, not corruption.
            # Never run destructive recovery in this case.
            if _is_lock_or_busy_error(exc):
                raise
            # Corruption can also surface as OperationalError (e.g. "database
            # disk image is malformed"). Route known corruption signatures to
            # the recovery path below instead of failing startup.
            if _is_known_corruption_error(exc):
                result = None
            else:
                # Unknown OperationalError: fail closed (no file deletion).
                raise
        except sqlite3.DatabaseError:
            # Integrity check itself failed — DB is unreadable/corrupt.
            # Proceed to WAL/SHM sidecar recovery path below.
            result = None
        if result and result[0] == "ok":
            return

        logger.warning(
            "holographic memory_store integrity check failed after WAL replay. "
            "Archiving DB and starting fresh — facts may be recoverable from backup archives.",
        )

        # Close connection and archive the DB (never silently delete).
        try:
            self._conn.close()
        except Exception:
            pass

        import time as _time
        _ts = int(_time.time())
        _archive = self.db_path.parent / (self.db_path.name + f".corrupted.{_ts}")
        try:
            logger.info("holographic: archiving %s → %s", self.db_path, _archive)
            self.db_path.rename(_archive)
        except FileNotFoundError:
            pass  # Already gone — no action needed
        except OSError as exc:
            logger.warning("holographic: could not archive corrupt DB: %s", exc)

        # Archive orphaned WAL/SHM sidecars too
        wal_path, shm_path = _wal_sidecar_paths(self.db_path)
        for path in (wal_path, shm_path):
            try:
                if path.exists():
                    _sidecar_archive = path.parent / (path.name + f".corrupted.{_ts}")
                    logger.info("holographic: archiving orphaned sidecar %s → %s", path, _sidecar_archive)
                    path.rename(_sidecar_archive)
            except Exception as exc:
                logger.warning("holographic: could not archive sidecar %s: %s", path, exc)

        # Open fresh DB (WAL mode will be re-established by _init_db).
        self._conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info("holographic memory_store re-opened after WAL recovery.")

    def _try_snapshot(self) -> None:
        """Snapshot facts to JSON as a safety net before destructive recovery.
        Uses the store's own connection — independent of the provider's method."""
        try:
            import json, os as _os
            from datetime import datetime
            from hermes_constants import get_hermes_home
            snap_path = Path(str(get_hermes_home())) / "memory" / "memory_store_snapshot.json"
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            rows = self._conn.execute(
                "SELECT fact_id, content, category, tags, trust_score, "
                "retrieval_count, helpful_count, created_at, updated_at, hrr_vector "
                "FROM facts ORDER BY fact_id"
            ).fetchall()
            # Convert BLOB hrr_vector to hex for JSON serialization
            facts = []
            for r in rows:
                d = dict(r)
                if d.get("hrr_vector") is not None:
                    d["hrr_vector"] = bytes(d["hrr_vector"]).hex()
                facts.append(d)
            snapshot = {
                "version": 2,
                "timestamp": datetime.now().isoformat(),
                "facts": facts,
            }
            tmp = snap_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
                f.flush()
            _os.replace(str(tmp), str(snap_path))
            logger.info("holographic: pre-recovery snapshot saved (%d facts)", len(facts))
        except Exception as exc:
            logger.warning("holographic: pre-recovery snapshot failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.

        All database work is wrapped in an explicit transaction so that a crash
        or exception anywhere in the chain rolls back cleanly — no partial state.
        The RLock is held only for the database I/O; numpy HRR encoding is
        performed without the lock held so concurrent readers are not starved.
        """
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")

        with self._lock:
            # Use an explicit savepoint so failures never leak partial writes.
            self._conn.execute("SAVEPOINT add_fact_start")
            add_savepoint_active = True

            try:
                try:
                    cur = self._conn.execute(
                        """
                        INSERT INTO facts (content, category, tags, trust_score)
                        VALUES (?, ?, ?, ?)
                        """,
                        (content, category, tags, self.default_trust),
                    )
                    fact_id: int = cur.lastrowid  # type: ignore[assignment]
                except sqlite3.IntegrityError:
                    # Duplicate content — rollback and return existing id.
                    self._conn.execute("ROLLBACK TO SAVEPOINT add_fact_start")
                    self._conn.execute("RELEASE SAVEPOINT add_fact_start")
                    add_savepoint_active = False
                    row = self._conn.execute(
                        "SELECT fact_id FROM facts WHERE content = ?", (content,)
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("duplicate fact lookup failed")
                    return int(row["fact_id"])

                # Entity resolution — collect entity_ids; rollback all on any failure.
                entity_ids: list[int] = []
                for name in self._extract_entities(content):
                    entity_id, _ = self._resolve_entity_with_flag(name)
                    entity_ids.append(entity_id)

                self._conn.execute("SAVEPOINT link_entities")
                link_savepoint_active = True
                try:
                    for entity_id in entity_ids:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                            (fact_id, entity_id),
                        )
                    self._conn.execute("RELEASE SAVEPOINT link_entities")
                    link_savepoint_active = False
                except Exception:
                    if link_savepoint_active:
                        self._conn.execute("ROLLBACK TO SAVEPOINT link_entities")
                        self._conn.execute("RELEASE SAVEPOINT link_entities")
                        link_savepoint_active = False
                    raise

                self._conn.execute("RELEASE SAVEPOINT add_fact_start")
                add_savepoint_active = False
                self._conn.commit()
                logger.info("holographic: added fact %d in category '%s'", fact_id, category)
            except Exception:
                # Any failure: rollback the whole add_fact transaction.
                if add_savepoint_active:
                    with contextlib.suppress(sqlite3.Error):
                        logger.warning("holographic: transaction failure in add_fact, rolling back")
                        self._conn.execute("ROLLBACK TO SAVEPOINT add_fact_start")
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.execute("RELEASE SAVEPOINT add_fact_start")
                raise

        # HRR vector and bank rebuild are slow numpy ops.
        # Release the lock first so concurrent readers are not starved.
        self._compute_hrr_vector(fact_id, content)
        self._rebuild_bank(category)

        return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            params: list = [query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.

        When content changes, entity re-linking is performed atomically inside
        a savepoint — if linking fails, the entity graph rolls back but the
        content/trust update is already committed (intentional: content change
        is the primary update; entity links are secondary).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []
            rebuild_old_cat: str | None = None  # non-None = old category needs rebuild

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
                # Detect category change: fetch old cat before UPDATE
                old_row = self._conn.execute(
                    "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()
                if old_row and old_row["category"] != category:
                    rebuild_old_cat = old_row["category"]
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # If content changed, re-extract and re-link entities atomically.
            # Use a savepoint so a linking failure rolls back the links only.
            if content is not None:
                self._conn.execute("SAVEPOINT relink_entities")
                try:
                    self._conn.execute(
                        "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                    )
                    for name in self._extract_entities(content):
                        entity_id, _ = self._resolve_entity_with_flag(name)
                        self._conn.execute(
                            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                            (fact_id, entity_id),
                        )
                    self._conn.execute("RELEASE SAVEPOINT relink_entities")
                    self._conn.commit()
                except Exception:
                    self._conn.execute("ROLLBACK TO SAVEPOINT relink_entities")
                    self._conn.execute("RELEASE SAVEPOINT relink_entities")
                    raise  # content is already updated; surface the linking error

            # Rebuild affected bank(s) — only when bank contents actually changed.
            rebuild_cats = []
            if content is not None or category is not None:
                cat = category or self._conn.execute(
                    "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()["category"]
                rebuild_cats.append(cat)
                if rebuild_old_cat is not None:
                    rebuild_cats.append(rebuild_old_cat)

        # Release the lock first so concurrent readers are not starved.
        if content is not None:
            self._compute_hrr_vector(fact_id, content)

        for cat in rebuild_cats:
            self._rebuild_bank(cat)

        logger.info("holographic: updated fact %d", fact_id)
        return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed.

        Rebuilds the category's memory bank only if the removed fact had an
        hrr_vector — facts without vectors have no effect on the bundle.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category, hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            cat = row["category"]
            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()

            # Only rebuild if the fact contributed a vector to the bundle.
            if row["hrr_vector"] is not None:
                self._rebuild_bank(cat)
            
            logger.info("holographic: removed fact %d from category '%s'", fact_id, cat)
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            logger.debug("holographic: found %d facts for query", len(rows))
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                   e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.

        NOTE: Multi-word entities (e.g. "Claude Code") are stored as a single
        atom in HRR space. This means probe/reason algebra works best with
        single-word entities. Multi-word entities are still stored and searchable
        via FTS, but the HRR structural role won't be fully compositional.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if not stripped:
                return
            # Normalize: strip trailing punctuation before dedup to prevent
            # 'Routine.' and 'Routine' becoming separate entities.
            normalized = stripped.rstrip(".,;:!?")
            if not normalized:
                return
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    def _resolve_entity_with_flag(self, name: str) -> tuple[int, bool]:
        """Find an existing entity by name or alias, or create one.

        Returns (entity_id, is_new) where is_new is True if the entity was created
        during this call. Callers can use is_new to track rollback candidates.

        Uses = with COLLATE NOCASE for exact-name matching (case-insensitive),
        and LIKE with % boundaries for alias matching — both avoiding the
        undefined LIKE case-sensitivity of the underlying column.
        """
        # Normalize: strip trailing punctuation for consistent matching.
        normalized = name.strip().rstrip(".,;:!?")
        if not normalized:
            normalized = name.strip()  # fall back if stripping kills it

        # Exact name match (case-insensitive).
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE",
            (normalized,),
        ).fetchone()
        if row is not None:
            return int(row["entity_id"]), False

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries.
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (normalized,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"]), False

        # Create new entity.
        # Do NOT commit here: callers may be inside an active SAVEPOINT.
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (normalized,)
        )
        return int(cur.lastrowid), True  # type: ignore[return-value]

    def _link_fact_entity_no_commit(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities without committing. Caller batches commits."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable.

        Re-checks numpy availability at call time (not just at __init__) in case
        it was installed after the store was created.

        The lock is NOT held during numpy encoding — this is intentional: numpy
        operations are orders of magnitude slower than SQLite I/O, and holding
        the lock blocks all concurrent readers. SQLite's WAL ensures the partial
        write (hrr_vector) is not visible to other connections until we commit.
        """
        # Re-check numpy availability per call.
        if not hrr._HAS_NUMPY:
            # add_fact has already inserted/linked rows before this call;
            # commit them so they are durable even without NumPy/HRR support.
            with self._lock:
                self._conn.commit()
            return

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

        logger.debug("holographic: computing vector for fact %d (%d entities)", fact_id, len(entities))
        vector = hrr.encode_fact(content, entities, self.hrr_dim)

        with self._lock:
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()
            logger.debug("holographic: saved vector for fact %d", fact_id)

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors.

        Validates SNR before bundling: if SNR < _SNR_MIN the bank is stored
        with NULL vector and fact_count=0, signalling that retrieval should fall
        back to plain FTS rather than holographic search.

        The lock is NOT held during numpy bundle() — same reasoning as above.
        """
        # Re-check numpy availability per call.
        if not hrr._HAS_NUMPY:
            # add_fact already persisted the row; ensure that pending write is
            # committed even when bank rebuild is skipped.
            with self._lock:
                self._conn.commit()
            return

        bank_name = f"cat:{category}"

        with self._lock:
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

        if not rows:
            with self._lock:
                # No facts with vectors yet — upsert an empty bank so existence is
                # recorded, avoiding repeated scans of an empty category.
                self._conn.execute(
                    """
                    INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                    VALUES (?, NULL, ?, 0, CURRENT_TIMESTAMP)
                    ON CONFLICT(bank_name) DO UPDATE SET
                        vector = NULL,
                        dim = excluded.dim,
                        fact_count = 0,
                        updated_at = excluded.updated_at
                    """,
                    (bank_name, self.hrr_dim),
                )
                self._conn.commit()
            return

        vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]
        fact_count = len(vectors)

        # Validate SNR before bundling.
        snr = hrr.snr_estimate(self.hrr_dim, fact_count)
        if snr < _SNR_MIN:
            # SNR too low — store empty bank; retrieval must use plain FTS.
            # Log the degraded category so it can be monitored.
            logger.warning(
                "holographic: category '%s' SNR=%.2f < %.1f threshold; bundle "
                "skipped for %d facts (falling back to FTS)",
                category, snr, _SNR_MIN, fact_count,
            )
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                    VALUES (?, NULL, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(bank_name) DO UPDATE SET
                        vector = NULL,
                        dim = excluded.dim,
                        fact_count = excluded.fact_count,
                        updated_at = excluded.updated_at
                    """,
                    (bank_name, self.hrr_dim, fact_count),
                )
                self._conn.commit()
            return

        logger.debug(
            "holographic: bundling %d vectors for category '%s' (SNR=%.2f)", 
            fact_count, category, snr
        )
        bank_vector = hrr.bundle(*vectors)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()
        logger.info("holographic: rebuilt memory bank for category '%s' (%d facts)", category, fact_count)

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.

        This is an admin tool — call it explicitly after recovering from
        corruption or upgrading numpy. It is NOT called automatically on startup.
        """
        with self._lock:
            if not hrr._HAS_NUMPY:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    # ------------------------------------------------------------------
    # Online backup (periodic, runs as subprocess every interval seconds)
    #
    # Runs as an independent subprocess — survives gateway SIGTERM/SIGKILL,
    # creating a consistent backup even if the gateway is killed mid-cycle.
    #
    # Each invocation: fires a Python subprocess that calls sqlite3.backup()
    # (source = live DB, dest = standalone .db file).  Fire-and-forget
    # so the gateway never blocks on I/O.  The subprocess is a child of
    # init (not the gateway), so systemd won't reap it.
    # ------------------------------------------------------------------

    def online_backup(self) -> str | None:
        """Create a consistent backup by spawning a standalone subprocess.

        The subprocess calls sqlite3.backup() using its OWN connection to the
        DB — completely independent of the gateway process.  If the gateway
        is killed (SIGTERM/SIGKILL), the subprocess survives and finishes
        the backup.

        Returns the backup path (if spawned), or None on failure.
        """
        import datetime as _datetime
        import time as _time

        _ts = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        _backup_path = self._online_backup_dir / f"memory_store_{_ts}.db"

        try:
            _script = f"""
import sqlite3, sys, os
from pathlib import Path

src = {str(repr(str(self.db_path)))}
dst = {str(repr(str(_backup_path)))}
keep = 28

try:
    # Open with isolation_level=None so sqlite3.backup() can proceed
    # while the gateway has the DB open with an active transaction.
    sc = sqlite3.connect(src, timeout=10.0, isolation_level=None)
    dc = sqlite3.connect(dst, timeout=10.0)
    sc.backup(dc)
    dc.close()
    sc.close()
    print('OK ' + dst, file=sys.stderr)
except Exception as e:
    print('ERR ' + str(e), file=sys.stderr)
    if os.path.exists(dst):
        try: os.unlink(dst)
        except: pass

# Prune oldest backups (keep most recent 28)
backup_dir = Path(dst).parent
try:
    backups = sorted(backup_dir.glob('memory_store_*.db'), key=lambda p: p.stat().st_mtime)
    for old in backups[:-keep]:
        old.unlink(missing_ok=True)
except Exception:
    pass
"""
            import subprocess as _subprocess
            _subprocess.Popen(
                [_subprocess.sys.executable, "-c", _script],
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.debug("holographic: backup subprocess spawned → %s", _backup_path)
            return str(_backup_path)
        except Exception as exc:
            logger.warning("holographic: failed to spawn backup subprocess: %s", exc)
            return None

    def _prune_online_backups(self) -> None:
        """Keep the most recent 28 online backups (~2 hours at 5-min intervals)."""
        try:
            backups = sorted(
                self._online_backup_dir.glob("memory_store_*.db"),
                key=lambda p: p.stat().st_mtime,
            )
            while len(backups) > 28:
                old = backups.pop(0)
                old.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("holographic: online backup prune failed: %s", exc)

    def start_online_backup(self, interval: int = 300) -> None:
        """Start the background online backup thread.

        Args:
            interval: Seconds between backups (default 300 = 5 minutes).
        """
        if self._backup_running:
            logger.debug("holographic: online backup thread already running")
            return

        self._backup_interval = interval
        self._backup_running = True

        def _loop() -> None:
            import time as _time
            while self._backup_running:
                _time.sleep(self._backup_interval)
                if not self._backup_running:
                    break
                self.online_backup()

        self._backup_thread = threading.Thread(
            target=_loop,
            name="holographic-online-backup",
            daemon=True,
        )
        self._backup_thread.start()
        logger.info("holographic: online backup thread started (interval=%ds)", interval)

    def stop_online_backup(self) -> None:
        """Stop the background online backup thread (graceful)."""
        self._backup_running = False
        if self._backup_thread is not None and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5.0)
        self._backup_thread = None
        logger.info("holographic: online backup thread stopped")

    def close(self) -> None:
        """Close the database connection.

        TRUNCATE checkpoint writes all WAL frames back to the main DB file,
        then truncates the WAL to 0 bytes — preventing orphaned -wal/-shm files
        after a crash and keeping the store as a single-file backup target.

        Only runs in WAL mode; DELETE/FULL journals do not support checkpoints.

        This method is guaranteed NOT to raise. Checkpoint failures are logged
        but do not prevent the connection from closing.

        Locking strategy: the RLock is acquired only briefly to finalise any
        pending transaction (commit or rollback) and check journal mode — the
        lock is released before the slow checkpoint/fsync I/O so that other
        threads are never blocked by a long-running checkpoint.
        """
        # Stop the online backup thread before closing the connection
        self.stop_online_backup()

        lock = getattr(self, "_lock", None)

        def _finalize_transaction() -> bool:
            """Return True if WAL mode is active (checkpoint is useful)."""
            try:
                if self._conn.in_transaction:
                    self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            try:
                row = self._conn.execute("PRAGMA journal_mode").fetchone()
                return row is not None and row[0].upper() == "WAL"
            except Exception:
                return False

        def _do_checkpoint() -> None:
            """Run the WAL checkpoint OUTSIDE the RLock — slow I/O must not block writers."""
            try:
                ckpt = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if ckpt and int(ckpt[0]) != 0:
                    logger.warning(
                        "WAL checkpoint(TRUNCATE) reported busy=%s for %s; "
                        "WAL may persist until next clean startup.",
                        ckpt[0],
                        self.db_path,
                    )
            except sqlite3.OperationalError as e:
                logger.warning(
                    "WAL checkpoint(TRUNCATE) failed for %s: %s. "
                    "The WAL file may be orphaned — it will be removed on next startup.",
                    self.db_path,
                    e,
                )
            except Exception as e:
                logger.warning(
                    "WAL checkpoint(TRUNCATE) raised unexpected %s for %s: %s. "
                    "WAL sidecar will be removed on next startup.",
                    type(e).__name__,
                    self.db_path,
                    e,
                )

        def _close_connection() -> None:
            try:
                self._conn.close()
                logger.info("holographic: closed database at %s", self.db_path)
            except Exception:
                pass  # Double-close is safe

        if lock is None:
            is_wal = _finalize_transaction()
            if is_wal:
                _do_checkpoint()
            _close_connection()
            return

        is_wal = False
        try:
            with lock:
                is_wal = _finalize_transaction()
            # checkpoint runs WITHOUT the lock held — slow I/O must not block writers
            if is_wal:
                _do_checkpoint()
        except Exception:
            pass  # never raise contract
        finally:
            try:
                _close_connection()
            except Exception:
                pass

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
