"""AI provider registry: configuration model, ready-made templates and dispatch.

A provider is one API endpoint the arbiter can call. Three protocols are
supported (``kind``):

* ``gemini``    Google Gemini via ``google-genai`` (optional Google Search grounding)
* ``anthropic`` Anthropic Claude via the official ``anthropic`` SDK (optional web search)
* ``openai``    any OpenAI-compatible Chat Completions endpoint: OpenAI, OpenRouter,
                Mistral, Groq, Ollama, LM Studio, vLLM...

The API key can be stored with the provider or read from an environment
variable (``api_key_env``), so secrets can stay out of the settings file.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields
from typing import Any

from ..models import AiVerdict

KINDS: dict[str, str] = {
    "gemini": "Google Gemini",
    "anthropic": "Anthropic Claude",
    "openai": "Compatibile OpenAI",
}
# Protocols that refuse to work without a key (local OpenAI-compatible servers usually need none).
KEY_REQUIRED = {"gemini", "anthropic"}
SEARCH_CAPABLE = {"gemini", "anthropic"}


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    name: str
    kind: str = "openai"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    web_search: bool = False
    enabled: bool = True

    # -- (de)serialisation ------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        for key in ("id", "name", "kind", "model", "base_url", "api_key", "api_key_env"):
            if key in clean:
                clean[key] = str(clean[key] or "").strip()
        for key in ("web_search", "enabled"):
            if key in clean:
                clean[key] = bool(clean[key])
        if clean.get("kind") not in KINDS:
            clean["kind"] = "openai"
        clean.setdefault("id", slugify(clean.get("name", "")) or "provider")
        clean.setdefault("name", clean["id"])
        return cls(**clean)

    def to_dict(self, include_secret: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_secret:
            data["api_key"] = ""
        return data

    # -- helpers ----------------------------------------------------------
    @property
    def kind_label(self) -> str:
        return KINDS.get(self.kind, self.kind)

    @property
    def supports_search(self) -> bool:
        return self.kind in SEARCH_CAPABLE

    def resolved_key(self) -> str | None:
        """Stored key, then the named environment variable, then the provider's standard variable."""
        if self.api_key:
            return self.api_key
        names = [self.api_key_env] if self.api_key_env else []
        names += STANDARD_KEY_ENV.get(self.kind, [])
        for name in names:
            value = os.getenv(name, "").strip()
            if value:
                return value
        return None

    def key_source(self) -> str:
        if self.api_key:
            return f"salvata (…{self.api_key[-4:]})" if len(self.api_key) > 8 else "salvata"
        for name in ([self.api_key_env] if self.api_key_env else []) + STANDARD_KEY_ENV.get(self.kind, []):
            if os.getenv(name, "").strip():
                return f"da ${name}"
        return "assente" if self.kind in KEY_REQUIRED else "non necessaria"

    def is_ready(self) -> bool:
        return self.enabled and bool(self.model) and (self.kind not in KEY_REQUIRED or bool(self.resolved_key()))


STANDARD_KEY_ENV: dict[str, list[str]] = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": [],
}

# Ready-made starting points shown in the "new provider" form.
TEMPLATES: dict[str, dict[str, Any]] = {
    "Google Gemini": {"kind": "gemini", "model": "gemini-2.5-flash", "web_search": True},
    "Anthropic Claude": {"kind": "anthropic", "model": "claude-opus-5-5", "web_search": True},
    "OpenAI": {"kind": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4.1-mini", "api_key_env": "OPENAI_API_KEY"},
    "OpenRouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "model": "", "api_key_env": "OPENROUTER_API_KEY"},
    "Mistral": {
        "kind": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "model": "pixtral-large-latest",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "Groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "model": "", "api_key_env": "GROQ_API_KEY"},
    "Ollama (locale)": {"kind": "openai", "base_url": "http://localhost:11434/v1", "model": "qwen2.5vl"},
    "LM Studio (locale)": {"kind": "openai", "base_url": "http://localhost:1234/v1", "model": ""},
    "Personalizzato (compatibile OpenAI)": {"kind": "openai", "base_url": "", "model": ""},
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40]


def default_providers(gemini_model: str = "gemini-2.5-flash", gemini_search: bool = True) -> tuple[ProviderConfig, ...]:
    """Provider list used until the user saves their own (keeps the .env-only setup working)."""
    return (ProviderConfig(id="gemini", name="Google Gemini", kind="gemini", model=gemini_model, web_search=gemini_search),)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


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
    key = api_key or provider.resolved_key()
    mdl = (model or provider.model or "").strip()
    search = provider.web_search if use_search is None else use_search
    if not mdl:
        return AiVerdict(error=f"Provider «{provider.name}»: nessun modello impostato.")
    if provider.kind == "gemini":
        from . import gemini

        return gemini.analyze(image_path, context, key, mdl, use_search=search, timeout_s=timeout_s, base_url=provider.base_url)
    if provider.kind == "anthropic":
        from . import anthropic_provider

        return anthropic_provider.analyze(image_path, context, key, mdl, use_search=search, timeout_s=timeout_s, base_url=provider.base_url)
    from . import openai_compat

    return openai_compat.analyze(image_path, context, key, mdl, base_url=provider.base_url, timeout_s=timeout_s)


def list_models(provider: ProviderConfig, timeout_s: int = 20, api_key: str | None = None) -> list[str]:
    """Ask the endpoint which models it offers. Raises on authentication or network errors."""
    key = api_key or provider.resolved_key()
    if provider.kind in KEY_REQUIRED and not key:
        raise ValueError("chiave API assente")
    if provider.kind == "gemini":
        from . import gemini

        return gemini.list_models(key, timeout_s, provider.base_url)
    if provider.kind == "anthropic":
        from . import anthropic_provider

        return anthropic_provider.list_models(key, timeout_s, provider.base_url)
    from . import openai_compat

    return openai_compat.list_models(key, timeout_s, provider.base_url)


def test_connection(provider: ProviderConfig, timeout_s: int = 20, api_key: str | None = None) -> tuple[bool, str, list[str]]:
    """Check credentials and endpoint by listing models; never raises."""
    from .common import redact

    key = api_key or provider.resolved_key()
    try:
        models = list_models(provider, timeout_s, api_key=key)
    except ImportError as exc:
        return False, f'Pacchetto mancante: {exc.name or exc}. Installa gli extra con pip install -e ".[ai]".', []
    except Exception as exc:
        return False, redact(str(exc), key)[:400] or type(exc).__name__, []
    if not models:
        return True, "Connessione riuscita, ma l'endpoint non ha elencato modelli.", []
    note = ""
    if provider.model and provider.model not in models:
        note = f" Attenzione: il modello «{provider.model}» non è tra quelli elencati."
    return True, f"Connessione riuscita: {len(models)} modelli disponibili.{note}", models
