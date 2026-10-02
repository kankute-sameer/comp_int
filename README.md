# comp_int

Route a natural-language request to Composio tool slugs.

```text
route(request) -> [tool_slug, ...]
```

The result is at most eight canonical slugs, or an empty list when nothing in the catalog fits.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
COMPOSIO_API_KEY=your_composio_key
OPENAI_API_KEY=your_openai_key
```

Build the local catalog. This pages the full Composio tools list, embeds each tool with `text-embedding-3-small`, and builds a SQLite full-text index. It is resumable: if it stops, run the same command again.

```bash
.venv/bin/python src/build_index.py
```

That writes `catalog/catalog.db` (gitignored; about 540 MB for the full catalog). A two-page smoke test:

```bash
.venv/bin/python src/build_index.py --pages 2 --db /tmp/test.db
ROUTER_DB=/tmp/test.db .venv/bin/python src/route.py "check if an email address is valid"
```

## Query

```bash
.venv/bin/python src/route.py "Summarize repository issues and put the results in a Google Sheet."
```

```text
GITHUB_LIST_REPOSITORY_ISSUES
GOOGLESHEETS_GET_SHEET_NAMES
GITHUB_LIST_MILESTONES
GOOGLESHEETS_CREATE_SPREADSHEET_ROW
...
```

Each query embeds the request, so it needs `OPENAI_API_KEY` and takes about one to three seconds.

## How routing works

1. **Catalog.** `src/build_index.py` reads `GET /api/v3.1/tools`, drops deprecated tools, and stores a slim record per tool: slug, name, description, toolkit, tags, and input parameter names and descriptions. Those fields become one search text. Output schemas, logos, and versions are not indexed.
2. **Lexical search.** SQLite FTS5 scores that text with BM25.
3. **Semantic search.** Cosine similarity against the stored embeddings. If the request names an app ("google sheet", "slack"), that toolkit's closest tools are added to the candidates.
4. **Multi-step requests.** A request with "and", "then", or a comma is split by `gpt-4o-mini` into one task per app (`src/decompose.py`). Each task is ranked on its own and the lists are interleaved, so one app cannot take all eight slots. A single-action request skips that model call.
5. **Cutoff.** If the best match is too weak, or the request names something absent from the catalog, the result is empty.

## Layout

| Path | Role |
| --- | --- |
| `src/build_index.py` | Fetch, embed, and index the catalog |
| `src/catalog_store.py` | Slim-record and embedding helpers |
| `src/route.py` | `route(request)` |
| `src/decompose.py` | Split a multi-step request |
| `catalog/catalog.db` | Built locally; not in git |
