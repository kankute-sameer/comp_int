"""Shared helpers: turn Composio tool items into slim records and embed text."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "catalog" / "catalog.db"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
MAX_SEARCH_CHARS = 8000

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_TOKEN = re.compile(r"[a-z0-9]+")


def load_env(name: str) -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                os.environ[name] = value.strip().strip('"').strip("'")
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is missing from the environment and .env")
    return value


def load_api_key() -> str:
    return load_env("OPENAI_API_KEY")


def words(value: str) -> str:
    spaced = _CAMEL.sub(r"\1 \2", value).replace("_", " ").replace("-", " ")
    return spaced.lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def parameter_fields(item: dict) -> list[dict]:
    properties = (item.get("input_parameters") or {}).get("properties") or {}
    if not isinstance(properties, dict):
        return []
    fields = []
    for name, spec in properties.items():
        description = ""
        if isinstance(spec, dict) and isinstance(spec.get("description"), str):
            description = spec["description"]
        fields.append({"name": str(name), "description": description})
    return fields


def search_text(record: dict) -> str:
    parts = [
        words(record["slug"]),
        record["name"],
        record["description"],
        record["toolkit_slug"],
        record["toolkit_name"],
        " ".join(words(tag) for tag in record["tags"]),
    ]
    for field in record["parameters"]:
        parts.append(field["name"].replace("_", " "))
        if field["description"]:
            parts.append(field["description"])
    text = "\n".join(part.strip() for part in parts if part and str(part).strip())
    return text[:MAX_SEARCH_CHARS]


def records_from_page(page: dict) -> tuple[list[dict], int]:
    """Slim records for one API page. Deprecated tools are skipped and counted."""
    records = []
    skipped = 0
    for item in page.get("items") or []:
        if item.get("is_deprecated") is True:
            skipped += 1
            continue
        toolkit = item.get("toolkit") or {}
        tags = [tag for tag in (item.get("tags") or []) if isinstance(tag, str)]
        record = {
            "slug": item.get("slug") or "",
            "name": item.get("name") or "",
            "description": item.get("description") or "",
            "toolkit_slug": toolkit.get("slug") or "",
            "toolkit_name": toolkit.get("name") or "",
            "tags": tags,
            "parameters": parameter_fields(item),
        }
        if not record["slug"]:
            continue
        record["search_text"] = search_text(record)
        records.append(record)
    return records, skipped


def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    body = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode()
    delay = 2.0
    last_error: Exception | None = None
    for _ in range(8):
        request = urllib.request.Request(
            EMBEDDING_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            rows = sorted(payload["data"], key=lambda row: row["index"])
            return [row["embedding"] for row in rows]
        except urllib.error.HTTPError as error:
            last_error = error
            detail = error.read().decode("utf-8", errors="replace")[:300]
            if error.code not in (429, 500, 502, 503):
                raise SystemExit(f"OpenAI embeddings failed ({error.code}): {detail}") from error
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise SystemExit(f"OpenAI embeddings failed after retries: {last_error}")
