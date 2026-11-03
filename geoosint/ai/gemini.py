"""Google Gemini provider.

The model receives the image plus everything the other engines found
(EXIF, OCR, objects, GeoCLIP candidates with their reverse-geocoded names)
and must answer with a strict JSON object.  With Google Search grounding
enabled it can verify shop names, landmarks and street signs online; the
web sources it used are returned for the analyst to check.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ..models import AiVerdict
from .common import (  # noqa: F401  (re-exported for backwards compatibility)
    MAX_SIDE,
    RESPONSE_SPEC,
    SYSTEM_PROMPT,
    build_context,
    encode_image,
    parse_verdict,
    redact,
    user_prompt,
    verdict_from_dict,
    verdict_from_text,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=4)
def _client(api_key: str, timeout_s: int, base_url: str = ""):
    from google import genai
    from google.genai import types

    options: dict[str, Any] = {"timeout": timeout_s * 1000}
    if base_url:
        options["base_url"] = base_url
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(**options))


def _image_part(image_path: str):
    from google.genai import types

    return types.Part.from_bytes(data=encode_image(image_path), mime_type="image/jpeg")


def _grounding_sources(response) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    try:
        for cand in response.candidates or []:
            meta = getattr(cand, "grounding_metadata", None)
            for chunk in getattr(meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    item = {"title": web.title or web.uri, "url": web.uri}
                    if item not in sources:
                        sources.append(item)
    except Exception:  # pragma: no cover - SDK shape changes
        log.debug("grounding metadata non leggibili", exc_info=True)
    return sources


def analyze(
    image_path: str,
    context: str,
    api_key: str | None,
    model: str,
    use_search: bool = True,
    timeout_s: int = 120,
    base_url: str = "",
) -> AiVerdict:
    if not api_key:
        return AiVerdict(error="Chiave Gemini assente: impostala nel pannello Impostazioni, in .env (GEMINI_API_KEY) o nelle opzioni.")
    try:
        from google.genai import types
    except ImportError:
        return AiVerdict(error="Pacchetto google-genai non installato (pip install google-genai).")

    config_kwargs: dict[str, Any] = {"temperature": 0.2, "system_instruction": SYSTEM_PROMPT}
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    else:
        config_kwargs["response_mime_type"] = "application/json"

    try:
        response = _client(api_key, timeout_s, base_url).models.generate_content(
            model=model,
            contents=[_image_part(image_path), user_prompt(context)],
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:  # network, quota, invalid key...
        return AiVerdict(model=model, error=f"Errore API Gemini: {redact(str(exc), api_key)[:500]}")
    verdict = verdict_from_text(response.text or "", model)
    verdict.sources = _grounding_sources(response)
    return verdict


def list_models(api_key: str | None, timeout_s: int = 20, base_url: str = "") -> list[str]:
    """Models that can generate content with this key (raises on auth/network errors)."""
    if not api_key:
        raise ValueError("chiave API assente")
    models = []
    for m in _client(api_key, timeout_s, base_url).models.list():
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        models.append((m.name or "").removeprefix("models/"))
    return sorted(filter(None, models))
