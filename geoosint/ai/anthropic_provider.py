"""Anthropic Claude provider (official ``anthropic`` SDK).

Optional web search uses Anthropic's server-side ``web_search`` tool; the URLs
Claude cites are returned as the verdict's sources.
"""

from __future__ import annotations

import base64
import logging
from functools import lru_cache
from typing import Any

from ..models import AiVerdict
from .common import SYSTEM_PROMPT, encode_image, redact, user_prompt, verdict_from_text

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
# Models that accept the server-side refusal fallback ("default" routing).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# Web search with dynamic filtering; older models only accept the basic variant.
_WEB_SEARCH_NEW = "web_search_20260209"
_WEB_SEARCH_BASIC = "web_search_20250305"
_MAX_CONTINUATIONS = 3


def _web_search_type(model: str) -> str:
    old = (
        "claude-3",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4-0",
        "claude-sonnet-4-0",
    )
    return _WEB_SEARCH_BASIC if model.startswith(old) else _WEB_SEARCH_NEW


@lru_cache(maxsize=4)
def _client(api_key: str, timeout_s: int, base_url: str = ""):
    import anthropic

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": float(timeout_s)}
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def _sources(content) -> list[dict[str, str]]:
    """Collect cited URLs (preferred) and, failing that, the raw search results."""
    cited: list[dict[str, str]] = []
    found: list[dict[str, str]] = []
    for block in content or []:
        btype = getattr(block, "type", "")
        if btype == "text":
            for cit in getattr(block, "citations", None) or []:
                url = getattr(cit, "url", None)
                if url:
                    item = {"title": getattr(cit, "title", None) or url, "url": url}
                    if item not in cited:
                        cited.append(item)
        elif btype == "web_search_tool_result":
            results = getattr(block, "content", None)
            if isinstance(results, list):  # an error result is a single object, not a list
                for r in results:
                    url = getattr(r, "url", None)
                    if url:
                        item = {"title": getattr(r, "title", None) or url, "url": url}
                        if item not in found:
                            found.append(item)
    return cited or found[:10]


def _text(content) -> str:
    return "".join(getattr(b, "text", "") for b in content or [] if getattr(b, "type", "") == "text")


def analyze(
    image_path: str,
    context: str,
    api_key: str | None,
    model: str,
    use_search: bool = False,
    timeout_s: int = 120,
    base_url: str = "",
) -> AiVerdict:
    if not api_key:
        return AiVerdict(model=model, error="Chiave Anthropic assente: impostala nel pannello Impostazioni.")
    try:
        import anthropic
    except ImportError:
        return AiVerdict(model=model, error="Pacchetto anthropic non installato (pip install anthropic).")

    image_b64 = base64.standard_b64encode(encode_image(image_path)).decode("ascii")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": user_prompt(context)},
            ],
        }
    ]
    params: dict[str, Any] = {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT}
    if use_search:
        params["tools"] = [{"type": _web_search_type(model), "name": "web_search", "max_uses": 5}]
    use_fallback = model in _FALLBACK_MODELS and not base_url

    client = _client(api_key, timeout_s, base_url)
    content: list = []
    try:
        for _ in range(_MAX_CONTINUATIONS + 1):
            if use_fallback:
                response = client.beta.messages.create(
                    **params, messages=messages, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                response = client.messages.create(**params, messages=messages)
            content.extend(response.content)
            if response.stop_reason != "pause_turn":
                break
            # A long server-side tool turn was paused: send it back to let Claude continue.
            messages = [*messages, {"role": "assistant", "content": response.content}]
    except anthropic.AuthenticationError:
        return AiVerdict(model=model, error="Chiave Anthropic non valida.")
    except anthropic.RateLimitError:
        return AiVerdict(model=model, error="Limite di richieste Anthropic raggiunto: riprova tra poco.")
    except anthropic.APIStatusError as exc:
        return AiVerdict(model=model, error=f"Errore API Anthropic ({exc.status_code}): {redact(str(exc.message), api_key)[:500]}")
    except anthropic.APIConnectionError as exc:
        return AiVerdict(model=model, error=f"Connessione ad Anthropic non riuscita: {redact(str(exc), api_key)[:300]}")

    served_by = getattr(response, "model", None) or model
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        reason = getattr(details, "explanation", None) or getattr(details, "category", None) or ""
        return AiVerdict(model=served_by, error=f"Il modello ha rifiutato la richiesta. {reason}".strip())
    if response.stop_reason == "max_tokens":
        return AiVerdict(model=served_by, error="Risposta del modello troncata (limite di token raggiunto).")

    verdict = verdict_from_text(_text(response.content) or _text(content), served_by)
    verdict.sources = _sources(content)
    return verdict


def list_models(api_key: str | None, timeout_s: int = 20, base_url: str = "") -> list[str]:
    if not api_key:
        raise ValueError("chiave API assente")
    return sorted(m.id for m in _client(api_key, timeout_s, base_url).models.list())
