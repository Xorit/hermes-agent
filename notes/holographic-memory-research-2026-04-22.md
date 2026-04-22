# Holographic Memory Research Note — 2026-04-22

Purpose: internal maintainer note capturing the current operational reality of Hermes Agent's Holographic memory provider, based on official docs, local source, git history, and upstream GitHub issues/PRs.

## Scope and sources

Primary sources reviewed:
- `website/docs/user-guide/features/memory.md`
- `website/docs/user-guide/features/memory-providers.md`
- `website/docs/developer-guide/memory-provider-plugin.md`
- `website/docs/developer-guide/architecture.md`
- `plugins/memory/holographic/{README.md,__init__.py,store.py,retrieval.py}`
- `run_agent.py`
- `acp_adapter/session.py`
- `gateway/run.py`
- local git history around `holographic`, `fact_store`, `memory_store`, `integrity`, `backup`
- upstream issues/PRs: `#4781`, `#5544`, `#10427`, `#11333`, `#14024`, `#14033`

## What Holographic is in Hermes terms

Holographic is Hermes Agent's local structured-memory `MemoryProvider`.

When enabled, it complements the built-in compact memory files (`MEMORY.md`, `USER.md`) rather than replacing them. It participates in the provider lifecycle documented by Hermes:
- initialization per agent/session path
- provider context injection into prompt construction
- prefetch before turns
- sync hooks after turns / on session end
- provider-specific tools

Its exposed tools are:
- `fact_store`
- `fact_feedback`

Implementation-wise, it is a local SQLite-backed deep memory system with:
- FTS5 lexical retrieval
- trust scoring
- entity resolution
- HRR-based compositional retrieval/reasoning
- current singleton store lifetime management via `MemoryStore.acquire()/release()`
- startup integrity checks, corrupted-file archival, and online backups

## What appears mature today

The strong parts are now clearly the local-memory substrate itself:
- no external dependency beyond SQLite
- structured fact model with entity-aware lookup
- contradiction support
- trust score evolution from `fact_feedback`
- significant hardening around corruption handling and backups
- broad holographic E2E test coverage

Operationally, the provider is no longer experimental in concept. The core store/retrieval design looks serious and well-supported.

## Where the real fragility is

The most fragile parts are integration edges, not HRR math.

Recurring failure classes:
1. provider is registered but not actually activated in the live path
2. provider is activated but tool schemas are not injected
3. tools are present but calls are not routed through the provider/memory manager
4. search path behaves badly because raw FTS5 query handling is brittle
5. shutdown / process lifecycle regresses and SQLite reliability suffers

This means many user-visible failures look like "memory is flaky" even when the DB itself is fine.

## High-risk areas to keep watching

### 1. Entry-path activation drift
CLI, gateway, ACP, and other agent-construction paths can diverge.

Recent local fixes had to add `memory_provider_override` plus ACP-side injection, which is a strong signal that provider activation was not robust across all entry points.

Maintainer takeaway: every new entry path must be reviewed for holographic activation, not just plugin registration.

### 2. Tool injection / tool gating drift
Upstream issue history shows holographic can fail even when present:
- `#4781` — plugin registered but tools not injected into agent loop
- `#5544` — provider tools interacted badly with toolset/platform config, causing latency problems

Maintainer takeaway: "registered" is not the same as "usable in the live loop".

### 3. FTS5 query sanitization
This is still a live and important operational hazard.

Relevant upstream work:
- `#11333` — sanitize natural-language FTS queries for better recall
- `#14024` — hyphenated query terms like `pve-01` can break FTS5 matching (`no such column`)
- `#14033` — sanitize direct `fact_store(search)` FTS queries

Maintainer takeaway: unsanitized FTS queries can mimic missing memory or corruption. For infrastructure-heavy users, this is one of the most important unresolved risks.

### 4. Multi-user scoping
- `#10427` tracks per-user scoping for holographic memory

Maintainer takeaway: shared deployments should not assume fact isolation unless the deployed branch explicitly includes user scoping.

### 5. Shutdown and store lifecycle
Recent hardening work repeatedly targeted:
- corruption cascade prevention
- online backup integration
- startup integrity archival
- singleton store lifetime
- SIGTERM/shutdown behavior

Maintainer takeaway: changes in gateway shutdown behavior can break holographic indirectly.

## Docs vs code/history

Official docs are good at explaining the plugin architecture and the user-facing role of memory providers.

They are weaker on:
- activation failures across entry paths
- tool injection/routing failure modes
- platform tool gating and latency interactions
- FTS5 parser edge cases
- singleton-store / backup-thread lifecycle concerns
- operational shutdown sensitivity

Maintainer takeaway: docs are the architectural baseline; source and git history are the operational truth.

## Practical operating guidance

When holographic appears broken, check in this order:
1. active provider selection
2. actual provider activation in the specific entry path
3. tool schema presence
4. tool-call routing through memory manager/provider
5. `_store` initialization state
6. FTS query sanitization behavior
7. DB integrity, archives, and backups

Especially suspicious signals:
- CLI works but ACP/gateway does not
- `fact_store` is unknown or absent
- obvious facts are missing from search
- hyphenated hostnames or infra identifiers break lookup
- unclean shutdowns recently occurred

## Current local status as of this note

The local branch now includes a fix commit for ACP routing + shutdown safety:
- `4e7b3c23` — `fix: holographic memory ACP routing + SIGTERM shutdown safety`

That commit added/fixed:
- `memory_provider_override` support in `run_agent.py`
- ACP-side holographic injection in `acp_adapter/session.py`
- lazy-init fallback in holographic tool handlers
- async/non-blocking shutdown diagnostic behavior in `gateway/run.py`

Follow-up review still matters after such integration changes because these are exactly the fragile edges for holographic.

## Recommended future work

Priority order:
1. merge/port FTS sanitization fixes on both retriever and direct `fact_store(search)` paths
2. keep ACP/API-server/provider activation paths aligned
3. reconcile provider tool injection with toolset/platform gating
4. add/merge per-user scoping for shared deployments
5. keep docs aligned with current shutdown/backups/singleton-store realities
6. add explicit tests for ACP activation and malformed memory-config types

## Bottom line

Holographic is mature enough to be a serious local structured-memory backend for Hermes.

Its biggest risks are not retrieval theory or SQLite itself in isolation, but integration edges:
- activation
- injection
- routing
- query sanitization
- shutdown lifecycle

If those edges are correct, holographic is strong. If they drift, it fails in ways that look random and user-hostile.