"""Route a natural-language request to at most eight canonical tool slugs.

Reads catalog/catalog.db (built by build_index.py):
  - BM25 comes from the SQLite FTS5 index
  - cosine similarity comes from the stored OpenAI embeddings
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_store import DB_PATH, _embed_batch, load_api_key, tokenize  # noqa: E402
from decompose import decompose  # noqa: E402

MAX_RESULTS = 8
CANDIDATES = 40
# text-embedding-3-small cosine. A real match sits at or above ~0.45; nearby
# tools stay if they are close to that best hit.
BEST_COSINE_FLOOR = 0.44
# A query word that is not in the catalog is an unknown constraint.
# Require a stronger semantic hit before returning anything.
UNKNOWN_TOKEN_COSINE_FLOOR = 0.55
CANDIDATE_COSINE_FLOOR = 0.42
COSINE_GAP = 0.12
BM25_NORM_KEEP = 0.80
BM25_WEIGHT = 0.45
COSINE_WEIGHT = 0.55
TOOLKIT_BOOST = 0.15
TOOLKIT_CANDIDATES = 25
MIN_TOOLKIT_LABEL = 4
TAG_BOOST = 0.04

STOPWORDS = {
    "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "with",
    "from", "my", "me", "please", "i", "want", "need", "can", "you", "your",
    "into", "onto", "via", "using", "use", "it", "this", "that", "is", "are",
    "was", "be", "do", "does", "how", "what", "when", "where", "who", "why",
    "just", "about", "would", "could", "should", "like", "some", "any", "our",
    "we", "they", "them", "their", "am", "off", "out",
}

INTENT_TAGS = {
    "send": "createHint",
    "create": "createHint",
    "add": "createHint",
    "draft": "createHint",
    "make": "createHint",
    "open": "createHint",
    "post": "createHint",
    "put": "createHint",
    "write": "createHint",
    "save": "createHint",
    "store": "createHint",
    "append": "createHint",
    "insert": "createHint",
    "log": "createHint",
    "list": "readOnlyHint",
    "find": "readOnlyHint",
    "search": "readOnlyHint",
    "fetch": "readOnlyHint",
    "show": "readOnlyHint",
    "read": "readOnlyHint",
    "lookup": "readOnlyHint",
    "check": "readOnlyHint",
    "update": "updateHint",
    "edit": "updateHint",
    "change": "updateHint",
    "modify": "updateHint",
    "delete": "destructiveHint",
    "remove": "destructiveHint",
    "cancel": "destructiveHint",
    "revoke": "destructiveHint",
}


@lru_cache(maxsize=1)
def load_store() -> dict:
    db_path = Path(os.environ.get("ROUTER_DB") or DB_PATH)
    if not db_path.exists():
        raise SystemExit(f"{db_path} not found. Run: python src/build_index.py")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    total = conn.execute(
        "SELECT COUNT(*) FROM tools WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    if total == 0:
        raise SystemExit("catalog.db has no embeddings. Run: python src/build_index.py embed")

    tools: list[dict] = []
    row_ids: list[int] = []
    vectors: np.ndarray | None = None
    cursor = conn.execute(
        "SELECT id, slug, toolkit_slug, toolkit_name, tags, embedding "
        "FROM tools WHERE embedding IS NOT NULL ORDER BY id"
    )
    for position, (row_id, slug, toolkit_slug, toolkit_name, tags, blob) in enumerate(cursor):
        vector = np.frombuffer(blob, dtype=np.float32)
        if vectors is None:
            vectors = np.empty((total, vector.shape[0]), dtype=np.float32)
        vectors[position] = vector
        row_ids.append(row_id)
        tools.append(
            {
                "slug": slug,
                "toolkit_slug": toolkit_slug,
                "toolkit_name": toolkit_name,
                "tags": json.loads(tags),
            }
        )
    assert vectors is not None
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    rows_by_toolkit: dict[str, list[int]] = {}
    labels: dict[str, set[str]] = {}
    for position, tool in enumerate(tools):
        rows_by_toolkit.setdefault(tool["toolkit_slug"], []).append(position)
        for raw in (tool["toolkit_slug"], tool["toolkit_name"]):
            label = compact(raw)
            if len(label) >= MIN_TOOLKIT_LABEL:
                labels.setdefault(label, set()).add(tool["toolkit_slug"])
    return {
        "conn": conn,
        "tools": tools,
        "row_ids": row_ids,
        "index_of": {row_id: position for position, row_id in enumerate(row_ids)},
        "vectors": vectors,
        "rows_by_toolkit": {k: np.asarray(v) for k, v in rows_by_toolkit.items()},
        "toolkit_labels": labels,
    }


def compact(value: str) -> str:
    return "".join(tokenize(value))


def mentioned_toolkits(request: str, labels: dict[str, set[str]]) -> set[str]:
    """Toolkits whose name appears in the request, ignoring spaces and underscores.

    "google calendar" matches the toolkit named googlecalendar because adjacent
    request words are joined before comparing.
    """
    tokens = tokenize(request)
    found: set[str] = set()
    for size in (1, 2, 3):
        for start in range(len(tokens) - size + 1):
            gram = "".join(tokens[start : start + size])
            # "google sheet" should match the toolkit googlesheets, and vice versa.
            for variant in (gram, gram + "s", gram[:-1] if gram.endswith("s") else gram):
                found |= labels.get(variant, set())
    return found


def query_tokens(request: str) -> list[str]:
    return [token for token in tokenize(request) if token not in STOPWORDS and len(token) > 1]


def intent_tags(tokens: list[str]) -> set[str]:
    return {INTENT_TAGS[token] for token in tokens if token in INTENT_TAGS}


def unique(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(tokens))


def known_terms(conn: sqlite3.Connection, tokens: list[str]) -> set[str]:
    if not tokens:
        return set()
    marks = ",".join("?" * len(tokens))
    rows = conn.execute(f"SELECT term FROM tools_fts_vocab WHERE term IN ({marks})", tokens)
    return {row[0] for row in rows}


def match_expression(tokens: list[str]) -> str:
    return " OR ".join(f'"{token}"' for token in tokens)


def lexical_top(conn: sqlite3.Connection, match: str, limit: int) -> list[int]:
    if not match:
        return []
    rows = conn.execute(
        "SELECT rowid FROM tools_fts WHERE tools_fts MATCH ? "
        "ORDER BY bm25(tools_fts) LIMIT ?",
        (match, limit),
    )
    return [row[0] for row in rows]


def lexical_scores(conn: sqlite3.Connection, match: str, row_ids: list[int]) -> dict[int, float]:
    """BM25 for the given rows. FTS5 returns lower-is-better, so flip the sign."""
    if not match or not row_ids:
        return {}
    marks = ",".join("?" * len(row_ids))
    rows = conn.execute(
        f"SELECT rowid, -bm25(tools_fts) FROM tools_fts "
        f"WHERE tools_fts MATCH ? AND rowid IN ({marks})",
        [match, *row_ids],
    )
    return {row[0]: row[1] for row in rows}


def tag_matches(wanted: set[str], tool: dict) -> bool:
    if not wanted:
        return False
    return any(tag in wanted for tag in tool["tags"])


def embed_queries(texts: list[str]) -> list[np.ndarray]:
    vectors = []
    for raw in _embed_batch(texts, load_api_key()):
        vector = np.asarray(raw, dtype=np.float32)
        vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
    return vectors


def rank(request: str, query_vector: np.ndarray | None = None) -> list[dict]:
    text = request.strip()
    if not text:
        return []
    store = load_store()
    conn = store["conn"]
    tools = store["tools"]
    row_ids = store["row_ids"]
    index_of = store["index_of"]

    tokens = unique(query_tokens(text))
    match = match_expression(tokens)

    if query_vector is None:
        query_vector = embed_queries([text])[0]
    cosine = store["vectors"] @ query_vector

    semantic_count = min(CANDIDATES, len(cosine))
    semantic_idx = set(int(i) for i in np.argpartition(cosine, -semantic_count)[-semantic_count:])
    lexical_idx = {
        index_of[row_id] for row_id in lexical_top(conn, match, CANDIDATES) if row_id in index_of
    }
    named = mentioned_toolkits(text, store["toolkit_labels"])
    toolkit_idx: set[int] = set()
    for toolkit_slug in named:
        rows = store["rows_by_toolkit"][toolkit_slug]
        take = min(TOOLKIT_CANDIDATES, len(rows))
        best = np.argpartition(cosine[rows], -take)[-take:]
        toolkit_idx |= {int(rows[i]) for i in best}
    candidates = sorted(semantic_idx | lexical_idx | toolkit_idx)
    bm25_by_row = lexical_scores(conn, match, [row_ids[i] for i in candidates])
    max_lexical = max(bm25_by_row.values(), default=0.0)
    wanted_tags = intent_tags(tokens)

    ranked = []
    for idx in candidates:
        tool = tools[idx]
        raw_bm25 = bm25_by_row.get(row_ids[idx], 0.0)
        lexical_norm = (raw_bm25 / max_lexical) if max_lexical > 0 else 0.0
        semantic = float(cosine[idx])
        score = COSINE_WEIGHT * max(semantic, 0.0) + BM25_WEIGHT * lexical_norm
        toolkit_hit = tool["toolkit_slug"] in named
        tag_hit = tag_matches(wanted_tags, tool)
        if toolkit_hit:
            score += TOOLKIT_BOOST
        if tag_hit:
            score += TAG_BOOST
        ranked.append(
            {
                "slug": tool["slug"],
                "score": score,
                "cosine": semantic,
                "bm25": raw_bm25,
                "toolkit_hit": toolkit_hit,
                "tag_hit": tag_hit,
            }
        )
    if not ranked:
        return []

    best_cosine = max(row["cosine"] for row in ranked)
    known = known_terms(conn, tokens)
    unknown = [token for token in tokens if token not in known]
    cosine_floor = UNKNOWN_TOKEN_COSINE_FLOOR if unknown else BEST_COSINE_FLOOR
    if best_cosine < cosine_floor:
        return []

    max_bm25 = max(row["bm25"] for row in ranked) or 1.0
    kept = []
    for row in ranked:
        bm25_norm = row["bm25"] / max_bm25
        close_to_best = row["cosine"] >= best_cosine - COSINE_GAP
        strong_lexical = bm25_norm >= BM25_NORM_KEEP
        if row["cosine"] >= CANDIDATE_COSINE_FLOOR and (close_to_best or strong_lexical):
            kept.append(row)
    kept.sort(key=lambda row: (-row["score"], row["slug"]))
    return kept


def merge_round_robin(lists: list[list[str]], limit: int) -> list[str]:
    """Take the best remaining slug from each list in turn, skipping duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for depth in range(max((len(items) for items in lists), default=0)):
        for items in lists:
            if depth < len(items) and items[depth] not in seen:
                seen.add(items[depth])
                merged.append(items[depth])
                if len(merged) == limit:
                    return merged
    return merged


def route(request: str) -> list[str]:
    """Return up to eight canonical slugs, or an empty list when nothing fits.

    A multi-step request is split into its parts. Each part is ranked on its own and
    the results are interleaved, so one dominant app cannot crowd out the others.
    """
    text = request.strip()
    if not text:
        return []
    parts = decompose(text)
    vectors = embed_queries(parts)
    per_part = [
        [row["slug"] for row in rank(part, vector)] for part, vector in zip(parts, vectors)
    ]
    return merge_round_robin(per_part, MAX_RESULTS)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python src/route.py "natural language request"')
    request = " ".join(sys.argv[1:])
    for slug in route(request):
        print(slug)


if __name__ == "__main__":
    main()
