"""Split a multi-step request into separate tool needs.

A request like "summarize issues and put them in a sheet" embeds as one blob that
favours its dominant app. Splitting it lets each part retrieve its own tools.
Only requests with connector words pay for the extra model call.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from catalog_store import load_api_key

CHAT_URL = "https://api.openai.com/v1/chat/completions"
DECOMPOSE_MODEL = "gpt-4o-mini"
MAX_TASKS = 4

_CONNECTORS = re.compile(
    r"\b(and|then|also|after|afterwards|before|plus|as well as|followed by)\b|[,;]",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You split a user request into the separate actions that would each need their "
    "own API tool call. Return JSON: {\"tasks\": [\"...\"]}. Each task is a short "
    "imperative sentence about exactly one app: name that app if the user did, and "
    "do not mention any other app in the same task. For example, 'post a Stripe "
    "payment in Slack' becomes 'Get the Stripe payment' and 'Post a message in "
    "Slack'. Keep app names and objects as the user wrote them. Do not add apps "
    "the request does not imply. Return a single task when the request is one "
    f"action. At most {MAX_TASKS} tasks."
)


def looks_multi_step(request: str) -> bool:
    return bool(_CONNECTORS.search(request))


def decompose(request: str) -> list[str]:
    """Return the sub-requests, or [request] when it is one action or the call fails."""
    text = request.strip()
    if not text or not looks_multi_step(text):
        return [text]
    body = json.dumps(
        {
            "model": DECOMPOSE_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
    ).encode()
    http_request = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {load_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=20) as response:
            payload = json.load(response)
        tasks = json.loads(payload["choices"][0]["message"]["content"])["tasks"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError):
        return [text]
    cleaned = [task.strip() for task in tasks if isinstance(task, str) and task.strip()]
    return cleaned[:MAX_TASKS] or [text]
