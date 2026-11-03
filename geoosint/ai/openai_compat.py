"""OpenAI-compatible provider (Chat Completions over plain HTTP).

Works with any service that speaks the OpenAI ``/chat/completions`` dialect:
OpenAI, OpenRouter, Mistral, Groq, Together, and local servers such as
Ollama, LM Studio or vLLM. The model must accept images.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import AiVerdict
from .common import SYSTEM_PROMPT, encode_image, redact, user_prompt, verdict_from_text

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"


class ProviderHTTPError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}" if status else message)
        self.status = status
        self.message = message


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "GeoOSINT"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _url(base_url: str, path: str) -> str:
    return (base_url or DEFAULT_BASE_URL).rstrip("/") + path


def _request(method: str, url: str, api_key: str | None, timeout_s: float, body: dict | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, headers=_headers(api_key), method=method)
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - URL is configured by the operator
            return json.loads(resp.read().decode("utf-8") or "{}")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            err = json.loads(raw).get("error", raw)
            message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        except (json.JSONDecodeError, AttributeError):
            message = raw
        raise ProviderHTTPError(exc.code, redact(message.strip(), api_key)[:500]) from None
    except URLError as exc:
        raise ProviderHTTPError(0, f"connessione non riuscita: {exc.reason}") from None
    except json.JSONDecodeError as exc:
        raise ProviderHTTPError(0, f"risposta non JSON: {exc}") from None


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some servers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def analyze(
    image_path: str,
    context: str,
    api_key: str | None,
    model: str,
    base_url: str = "",
    timeout_s: int = 120,
    temperature: float | None = 0.2,
) -> AiVerdict:
    image_b64 = base64.standard_b64encode(encode_image(image_path)).decode("ascii")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt(context)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    url = _url(base_url, "/chat/completions")
    try:
        try:
            data = _request("POST", url, api_key, timeout_s, body)
        except ProviderHTTPError as exc:
            # Some reasoning models only accept their default temperature: retry once without it.
            if exc.status == 400 and "temperature" in exc.message.lower() and "temperature" in body:
                body.pop("temperature")
                data = _request("POST", url, api_key, timeout_s, body)
            else:
                raise
    except ProviderHTTPError as exc:
        return AiVerdict(model=model, error=f"Errore API ({_url(base_url, '')}): {exc}")

    choices = data.get("choices") or []
    if not choices:
        return AiVerdict(model=model, error="Il provider non ha restituito alcuna risposta.")
    message = choices[0].get("message") or {}
    if message.get("refusal"):
        return AiVerdict(model=model, error=f"Il modello ha rifiutato la richiesta: {message['refusal']}")
    verdict = verdict_from_text(_message_text(message.get("content")), data.get("model") or model)
    # OpenRouter / Perplexity style citations, when present.
    for cit in data.get("citations") or []:
        if isinstance(cit, str):
            verdict.sources.append({"title": cit, "url": cit})
    return verdict


def list_models(api_key: str | None, timeout_s: int = 20, base_url: str = "") -> list[str]:
    data = _request("GET", _url(base_url, "/models"), api_key, timeout_s)
    items = data.get("data") if isinstance(data, dict) else data
    return sorted({str(m.get("id")) for m in items or [] if isinstance(m, dict) and m.get("id")})
