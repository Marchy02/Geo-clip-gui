"""LLM arbiter: cross-checks every clue and decides the location, through any LiteLLM provider.

The model receives the image plus everything the other engines found
(EXIF, OCR, objects, GeoCLIP candidates with their reverse-geocoded names)
and must answer with a strict JSON object. With web search enabled
(Google Search grounding on Gemini, the web search tool on Claude...) it can
verify shop names, landmarks and street signs online; the web sources it
cited are returned for the analyst to check.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from ..models import AiVerdict
from .common import SYSTEM_PROMPT, encode_image, redact, user_prompt, verdict_from_text
from .providers import ProviderConfig

log = logging.getLogger(__name__)

# The JSON verdict is short, but models that always reason spend output tokens on thinking first.
MAX_TOKENS = 16000


class LiteLLMMissing(RuntimeError):
    pass


def _litellm():
    try:
        import litellm
    except ImportError as exc:
        raise LiteLLMMissing('Pacchetto litellm non installato: pip install -e ".[ai]"') from exc
    litellm.suppress_debug_info = True
    litellm.drop_params = True  # silently drop options a provider does not support (e.g. web search)
    return litellm


def _call_kwargs(provider: ProviderConfig, api_key: str | None, timeout_s: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": timeout_s, "drop_params": True}
    if api_key:
        kwargs["api_key"] = api_key
    if provider.base_url:
        kwargs["api_base"] = provider.base_url
    return kwargs


def _friendly_error(exc: Exception, provider: ProviderConfig, key: str | None) -> str:
    name = type(exc).__name__
    detail = redact(str(getattr(exc, "message", None) or exc), key)
    hints = {
        "AuthenticationError": "chiave API non valida o assente",
        "PermissionDeniedError": "la chiave non ha accesso a questo modello",
        "NotFoundError": "modello o endpoint inesistente",
        "RateLimitError": "limite di richieste o quota esaurita, riprova più tardi",
        "Timeout": "il provider non ha risposto in tempo",
        "APIConnectionError": "connessione al provider non riuscita",
        "ContentPolicyViolationError": "richiesta bloccata dai filtri del provider",
        "ContextWindowExceededError": "richiesta troppo lunga per il modello",
    }
    hint = hints.get(name)
    prefix = f"{provider.name}: {hint}" if hint else f"{provider.name}: errore {name}"
    return f"{prefix}. {detail[:400]}".strip()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_sources(response: Any) -> list[dict[str, str]]:
    """Web sources cited by the model, whatever the provider's format."""
    out: list[dict[str, str]] = []

    def add(url: Any, title: Any = None) -> None:
        if isinstance(url, str) and url.startswith("http"):
            item = {"title": str(title or url), "url": url}
            if all(o["url"] != url for o in out):
                out.append(item)

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return out
    psf = _get(message, "provider_specific_fields") or {}
    # Anthropic: citations grouped per text block.
    for group in _get(psf, "citations") or []:
        for cit in group if isinstance(group, list) else [group]:
            add(_get(cit, "url"), _get(cit, "title"))
    # OpenAI-style url_citation annotations.
    for ann in _get(message, "annotations") or []:
        cit = _get(ann, "url_citation") or ann
        add(_get(cit, "url"), _get(cit, "title"))
    # Gemini / Vertex grounding.
    for meta in getattr(response, "vertex_ai_grounding_metadata", None) or []:
        for chunk in _get(meta, "groundingChunks") or []:
            web = _get(chunk, "web") or {}
            add(_get(web, "uri"), _get(web, "title"))
    # Perplexity / OpenRouter top-level citations.
    for url in getattr(response, "citations", None) or []:
        add(url)
    if not out:  # fall back to the raw search results Claude looked at
        for block in _get(psf, "web_search_results") or []:
            for r in _get(block, "content") or []:
                add(_get(r, "url"), _get(r, "title"))
    return out[:15]


def analyze(
    provider: ProviderConfig,
    image_path: str,
    context: str,
    timeout_s: int = 120,
    model: str | None = None,
    use_search: bool | None = None,
    api_key: str | None = None,
) -> AiVerdict:
    """Run the arbiter on ``provider``; ``model``/``use_search``/``api_key`` override it for one call."""
    if model:
        provider = ProviderConfig(**{**provider.to_dict(), "model": model.strip()})
    if not provider.model:
        return AiVerdict(error=f"{provider.name}: nessun modello impostato.")
    key = api_key or provider.resolved_key()
    try:
        litellm = _litellm()
    except LiteLLMMissing as exc:
        return AiVerdict(model=provider.model, error=str(exc))

    image_b64 = base64.standard_b64encode(encode_image(image_path)).decode("ascii")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt(context)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]
    kwargs = _call_kwargs(provider, key, timeout_s)
    if provider.web_search if use_search is None else use_search:
        kwargs["web_search_options"] = {"search_context_size": "medium"}
    try:
        response = litellm.completion(model=provider.litellm_model, messages=messages, max_tokens=MAX_TOKENS, **kwargs)
    except Exception as exc:
        log.warning("Arbitro IA fallito (%s): %s", provider.litellm_model, type(exc).__name__)
        return AiVerdict(model=provider.model, error=_friendly_error(exc, provider, key))

    choice = response.choices[0]
    served_by = getattr(response, "model", None) or provider.model
    if choice.finish_reason in {"refusal", "content_filter"}:
        return AiVerdict(model=served_by, error=f"{provider.name}: il modello ha rifiutato la richiesta.")
    text = choice.message.content or ""
    if not text and choice.finish_reason == "length":
        return AiVerdict(model=served_by, error=f"{provider.name}: risposta troncata (limite di token raggiunto).")
    verdict = verdict_from_text(text, served_by)
    verdict.sources = extract_sources(response)
    return verdict


def list_models(provider: ProviderConfig, api_key: str | None = None) -> list[str]:
    """Models the endpoint offers (asks the provider when it supports listing). Raises on errors."""
    litellm = _litellm()
    key = api_key or provider.resolved_key()
    models = litellm.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider=provider.kind,
        api_key=key,
        api_base=provider.base_url or None,
    )
    prefix = provider.kind + "/"
    return sorted({m.removeprefix(prefix) for m in models or []})


def test_connection(provider: ProviderConfig, timeout_s: int = 30, api_key: str | None = None) -> tuple[bool, str]:
    """Send a tiny text-only request to check endpoint, key and model. Never raises."""
    if not provider.model:
        return False, "Imposta prima un modello."
    key = api_key or provider.resolved_key()
    try:
        litellm = _litellm()
        response = litellm.completion(
            model=provider.litellm_model,
            messages=[{"role": "user", "content": "Rispondi solo con: OK"}],
            max_tokens=MAX_TOKENS,
            **_call_kwargs(provider, key, timeout_s),
        )
    except LiteLLMMissing as exc:
        return False, str(exc)
    except Exception as exc:
        return False, _friendly_error(exc, provider, key)
    reply = (response.choices[0].message.content or "").strip()[:60]
    note = ""
    try:
        if litellm.get_model_info(model=provider.litellm_model).get("supports_vision") is False:
            note = " Attenzione: secondo LiteLLM questo modello non accetta immagini."
    except Exception:
        pass  # unknown model: LiteLLM has no capability data for it
    return True, f"Connessione riuscita con {provider.litellm_model}" + (f" (risposta: «{reply}»)." if reply else ".") + note
