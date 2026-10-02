"""Build catalog/catalog.db from the full Composio tools catalog.

Three resumable phases:
  fetch  page GET /api/v3.1/tools, keep slim records in SQLite
  embed  one OpenAI embedding per tool that does not have one yet
  index  rebuild the SQLite FTS5 (BM25) index over the search text

Run all phases:      python src/build_index.py
Run one phase:       python src/build_index.py fetch
Smoke test:          python src/build_index.py --pages 2 --db /tmp/test.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_store import (  # noqa: E402
    DB_PATH,
    _embed_batch,
    load_api_key,
    load_env,
    records_from_page,
)

COMPOSIO_URL = "https://backend.composio.dev/api/v3.1/tools"
PAGE_LIMIT = 1000
EMBED_MAX_ITEMS = 256
EMBED_MAX_CHARS = 500_000
EMBED_WORKERS = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS tools (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    toolkit_slug TEXT NOT NULL,
    toolkit_name TEXT NOT NULL,
    tags TEXT NOT NULL,
    parameters TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS tools_toolkit ON tools(toolkit_slug);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def fetch_page(api_key: str, cursor: str | None) -> dict:
    params = {"limit": PAGE_LIMIT}
    if cursor:
        params["cursor"] = cursor
    url = f"{COMPOSIO_URL}?{urllib.parse.urlencode(params)}"
    delay = 2.0
    last_error: Exception | None = None
    for _ in range(6):
        request = urllib.request.Request(url, headers={"x-api-key": api_key})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503):
                detail = error.read().decode("utf-8", errors="replace")[:300]
                raise SystemExit(f"Composio fetch failed ({error.code}): {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise SystemExit(f"Composio fetch failed after retries: {last_error}")


def phase_fetch(conn: sqlite3.Connection, max_pages: int | None) -> None:
    if get_meta(conn, "fetch_done") == "1" and max_pages is None:
        print("fetch: already complete")
        return
    api_key = load_env("COMPOSIO_API_KEY")
    cursor = get_meta(conn, "fetch_cursor") or None
    pages_done = int(get_meta(conn, "pages_fetched") or 0)
    skipped_total = int(get_meta(conn, "skipped_deprecated") or 0)
    fetched_this_run = 0
    while True:
        started = time.time()
        page = fetch_page(api_key, cursor)
        records, skipped = records_from_page(page)
        conn.executemany(
            "INSERT OR IGNORE INTO tools(slug, name, description, toolkit_slug, "
            "toolkit_name, tags, parameters, search_text) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    r["slug"],
                    r["name"],
                    r["description"],
                    r["toolkit_slug"],
                    r["toolkit_name"],
                    json.dumps(r["tags"]),
                    json.dumps(r["parameters"]),
                    r["search_text"],
                )
                for r in records
            ],
        )
        pages_done += 1
        fetched_this_run += 1
        skipped_total += skipped
        cursor = page.get("next_cursor")
        set_meta(conn, "pages_fetched", str(pages_done))
        set_meta(conn, "skipped_deprecated", str(skipped_total))
        set_meta(conn, "total_items_reported", str(page.get("total_items")))
        set_meta(conn, "fetch_cursor", cursor or "")
        if not cursor:
            set_meta(conn, "fetch_done", "1")
        conn.commit()
        stored = conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
        print(
            f"fetch: page {page.get('current_page')}/{page.get('total_pages')} "
            f"+{len(records)} (skipped {skipped}) stored={stored} "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
        if not cursor:
            break
        if max_pages is not None and fetched_this_run >= max_pages:
            print("fetch: stopped at --pages limit")
            break


def embed_batches(rows: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    chars = 0
    for row in rows:
        size = len(row[1])
        if current and (len(current) >= EMBED_MAX_ITEMS or chars + size > EMBED_MAX_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(row)
        chars += size
    if current:
        batches.append(current)
    return batches


def phase_embed(conn: sqlite3.Connection) -> None:
    api_key = load_api_key()
    rows = conn.execute(
        "SELECT id, search_text FROM tools WHERE embedding IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        print("embed: nothing to do")
        return
    batches = embed_batches(rows)
    print(f"embed: {len(rows)} tools in {len(batches)} batches", flush=True)
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        for start in range(0, len(batches), EMBED_WORKERS * 2):
            group = batches[start : start + EMBED_WORKERS * 2]
            futures = [
                pool.submit(_embed_batch, [text for _, text in batch], api_key)
                for batch in group
            ]
            for batch, future in zip(group, futures):
                vectors = future.result()
                if len(vectors) != len(batch):
                    raise SystemExit("embedding count does not match batch size")
                conn.executemany(
                    "UPDATE tools SET embedding = ? WHERE id = ?",
                    [
                        (np.asarray(vec, dtype=np.float32).tobytes(), row_id)
                        for (row_id, _), vec in zip(batch, vectors)
                    ],
                )
                done += len(batch)
            conn.commit()
            print(
                f"embed: {done}/{len(rows)} {time.time() - started:.0f}s",
                flush=True,
            )


def phase_index(conn: sqlite3.Connection) -> None:
    started = time.time()
    conn.executescript(
        """
        DROP TABLE IF EXISTS tools_fts_vocab;
        DROP TABLE IF EXISTS tools_fts;
        CREATE VIRTUAL TABLE tools_fts USING fts5(
            search_text, content='tools', content_rowid='id'
        );
        INSERT INTO tools_fts(tools_fts) VALUES ('rebuild');
        CREATE VIRTUAL TABLE tools_fts_vocab USING fts5vocab(tools_fts, 'row');
        """
    )
    terms = conn.execute("SELECT COUNT(*) FROM tools_fts_vocab").fetchone()[0]
    set_meta(conn, "indexed_terms", str(terms))
    conn.commit()
    print(f"index: {terms} terms in {time.time() - started:.1f}s")


def report(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM tools WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    toolkits = conn.execute("SELECT COUNT(DISTINCT toolkit_slug) FROM tools").fetchone()[0]
    print(
        f"tools={total} embedded={embedded} toolkits={toolkits} "
        f"pages={get_meta(conn, 'pages_fetched')} "
        f"skipped_deprecated={get_meta(conn, 'skipped_deprecated')} "
        f"reported_total={get_meta(conn, 'total_items_reported')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", nargs="?", default="all", choices=["all", "fetch", "embed", "index"])
    parser.add_argument("--pages", type=int, default=None, help="stop fetch after N pages")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    conn = connect(args.db)
    if args.phase in ("all", "fetch"):
        phase_fetch(conn, args.pages)
    if args.phase in ("all", "embed"):
        phase_embed(conn)
    if args.phase in ("all", "index"):
        phase_index(conn)
    report(conn)
    conn.close()


if __name__ == "__main__":
    main()
