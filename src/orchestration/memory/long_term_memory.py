"""Long-term semantic memory: what the system remembers across tasks.

Documents live in ChromaDB (embeddings + similarity search). Importance,
access counts, and expiration -- things Chroma has no native concept of --
are tracked in the `long_term_memory_meta` SQLite table, keyed by the same
id as the Chroma document. The two stores are kept in lockstep by always
writing/deleting through this module rather than touching either directly.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from orchestration import config
from orchestration.db.connection import get_connection

_client = None
_collection = None

# Recency decay applied per day since last access; access reinforces
# importance so frequently-retrieved memories stay "important" (spec 2.4).
DAILY_DECAY = 0.985
ACCESS_BOOST = 0.15
MAX_IMPORTANCE = 3.0
EXPIRE_BELOW_IMPORTANCE = 0.15
EXPIRE_AFTER_DAYS = 120


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils import embedding_functions

    config.settings.resolved_chroma_path().mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(config.settings.resolved_chroma_path()))
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.settings.openai_api_key, model_name="text-embedding-3-small"
    )
    _collection = _client.get_or_create_collection("long_term_memory", embedding_function=embed_fn)
    return _collection


@dataclass
class MemoryHit:
    id: str
    text: str
    kind: str
    user_id: str
    metadata: dict
    distance: float
    importance: float


def remember(*, user_id: str, kind: str, text: str, metadata: dict | None = None) -> str:
    """Store a new long-term memory. kind: task_summary/preference/fact/approach."""
    memory_id = f"mem_{uuid.uuid4().hex[:12]}"
    full_metadata = {"user_id": user_id, "kind": kind, **(metadata or {})}
    collection = _get_collection()
    collection.add(ids=[memory_id], documents=[text], metadatas=[full_metadata])

    conn = get_connection()
    conn.execute(
        "INSERT INTO long_term_memory_meta (id, user_id, kind) VALUES (?, ?, ?)",
        (memory_id, user_id, kind),
    )
    conn.commit()
    return memory_id


def recall(*, user_id: str, query: str, kind: str | None = None, n_results: int = 5) -> list[MemoryHit]:
    """Query long-term memory for use in planning. Reinforces importance and
    access_count on every hit, per the spec's importance-scoring requirement.
    """
    collection = _get_collection()
    where: dict = {"user_id": user_id} if not kind else {"$and": [{"user_id": user_id}, {"kind": kind}]}

    result = collection.query(query_texts=[query], n_results=n_results, where=where)
    ids = result["ids"][0] if result["ids"] else []
    if not ids:
        return []

    conn = get_connection()
    hits: list[MemoryHit] = []
    for i, mem_id in enumerate(ids):
        meta_row = conn.execute("SELECT * FROM long_term_memory_meta WHERE id = ?", (mem_id,)).fetchone()
        importance = meta_row["importance"] if meta_row else 1.0
        conn.execute(
            """
            UPDATE long_term_memory_meta
            SET access_count = access_count + 1,
                last_accessed_at = datetime('now'),
                importance = MIN(?, importance + ?)
            WHERE id = ?
            """,
            (MAX_IMPORTANCE, ACCESS_BOOST, mem_id),
        )
        hits.append(
            MemoryHit(
                id=mem_id,
                text=result["documents"][0][i],
                kind=result["metadatas"][0][i].get("kind", "unknown"),
                user_id=result["metadatas"][0][i].get("user_id", user_id),
                metadata=result["metadatas"][0][i],
                distance=result["distances"][0][i] if result.get("distances") else 0.0,
                importance=importance,
            )
        )
    conn.commit()
    return hits


def decay_and_expire(user_id: str | None = None) -> dict:
    """Apply recency decay to importance scores and delete memories that
    have decayed below the floor and aged past EXPIRE_AFTER_DAYS. Intended
    to run periodically (e.g. from the memory dashboard or a cron job).
    """
    conn = get_connection()
    where_clause = "WHERE user_id = ?" if user_id else ""
    params = (user_id,) if user_id else ()
    rows = conn.execute(f"SELECT * FROM long_term_memory_meta {where_clause}", params).fetchall()

    now = datetime.utcnow()
    expired_ids: list[str] = []
    decayed = 0
    for row in rows:
        last_accessed = datetime.fromisoformat(row["last_accessed_at"])
        days_since = max((now - last_accessed).days, 0)
        new_importance = row["importance"] * (DAILY_DECAY ** days_since)

        created = datetime.fromisoformat(row["created_at"])
        age_days = (now - created).days
        if new_importance < EXPIRE_BELOW_IMPORTANCE and age_days > EXPIRE_AFTER_DAYS:
            expired_ids.append(row["id"])
        else:
            conn.execute(
                "UPDATE long_term_memory_meta SET importance = ? WHERE id = ?", (new_importance, row["id"])
            )
            decayed += 1

    if expired_ids:
        _get_collection().delete(ids=expired_ids)
        conn.executemany("DELETE FROM long_term_memory_meta WHERE id = ?", [(i,) for i in expired_ids])
    conn.commit()
    return {"decayed": decayed, "expired": len(expired_ids)}


def consolidate(user_id: str, kind: str = "task_summary", similarity_threshold: float = 0.08) -> dict:
    """Merge near-duplicate memories into a single higher-level summary via
    an LLM call, per the spec's memory-consolidation requirement. Kept
    simple: greedy pass over one kind at a time, merging anything within
    `similarity_threshold` cosine distance of a memory already picked as a
    cluster head.
    """
    from orchestration.llm.provider import get_provider

    collection = _get_collection()
    conn = get_connection()
    rows = conn.execute(
        "SELECT id FROM long_term_memory_meta WHERE user_id = ? AND kind = ?", (user_id, kind)
    ).fetchall()
    if len(rows) < 2:
        return {"merged_groups": 0}

    all_docs = collection.get(ids=[r["id"] for r in rows])
    remaining = set(all_docs["ids"])
    id_to_text = dict(zip(all_docs["ids"], all_docs["documents"]))
    merged_groups = 0
    provider = get_provider()

    while remaining:
        head_id = next(iter(remaining))
        neighbors = collection.query(
            query_texts=[id_to_text[head_id]], n_results=min(10, len(remaining)),
            where={"$and": [{"user_id": user_id}, {"kind": kind}]},
        )
        cluster = [
            nid for nid, dist in zip(neighbors["ids"][0], neighbors["distances"][0])
            if nid in remaining and dist <= similarity_threshold
        ]
        if head_id not in cluster:
            cluster.append(head_id)

        remaining -= set(cluster)
        if len(cluster) < 2:
            continue

        texts = [id_to_text[c] for c in cluster]
        response = provider.complete(
            model=config.settings.specialist_model,
            messages=[
                {"role": "system", "content": "Merge these related memory notes into one concise summary "
                                               "that preserves every distinct fact. Output only the summary."},
                {"role": "user", "content": "\n---\n".join(texts)},
            ],
        )
        collection.delete(ids=cluster)
        conn.executemany("DELETE FROM long_term_memory_meta WHERE id = ?", [(c,) for c in cluster])
        conn.commit()
        remember(user_id=user_id, kind=kind, text=response.content, metadata={"consolidated_from": len(cluster)})
        merged_groups += 1

    return {"merged_groups": merged_groups}


def delete_user_memories(user_id: str) -> int:
    """Full delete endpoint for user data-deletion requests."""
    conn = get_connection()
    rows = conn.execute("SELECT id FROM long_term_memory_meta WHERE user_id = ?", (user_id,)).fetchall()
    ids = [r["id"] for r in rows]
    if ids:
        _get_collection().delete(ids=ids)
        conn.executemany("DELETE FROM long_term_memory_meta WHERE id = ?", [(i,) for i in ids])
        conn.commit()
    return len(ids)


def dashboard_snapshot(user_id: str) -> list[dict]:
    """What the system 'remembers' about a user, for the memory dashboard."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM long_term_memory_meta WHERE user_id = ? ORDER BY importance DESC", (user_id,)
    ).fetchall()
    if not rows:
        return []
    docs = _get_collection().get(ids=[r["id"] for r in rows])
    text_by_id = dict(zip(docs["ids"], docs["documents"]))
    return [
        {
            "id": r["id"], "kind": r["kind"], "importance": round(r["importance"], 3),
            "access_count": r["access_count"], "created_at": r["created_at"],
            "last_accessed_at": r["last_accessed_at"], "text": text_by_id.get(r["id"], ""),
        }
        for r in rows
    ]
