"""AI provider configuration: the data model, ready-made templates and key resolution.

A provider is one LLM endpoint the arbiter can call. Calls go through LiteLLM,
so ``kind`` is a LiteLLM provider prefix (``gemini``, ``anthropic``, ``openai``,
``openrouter``, ``ollama_chat``...) and any provider LiteLLM supports works.

The API key can be stored with the provider, read from a named environment
variable (``api_key_env``) or, if both are empty, from the provider's standard
variable (``GEMINI_API_KEY``, ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``...),
which LiteLLM picks up on its own.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields
from typing import Any

# The most common LiteLLM providers, shown first in the UI. Any other LiteLLM prefix is accepted.
COMMON_KINDS: dict[str, str] = {
    "gemini": "Google Gemini (AI Studio)",
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI o compatibile OpenAI",
    "openrouter": "OpenRouter",
    "mistral": "Mistral",
    "groq": "Groq",
    "xai": "xAI Grok",
    "deepseek": "DeepSeek",
    "azure": "Azure OpenAI",
    "vertex_ai": "Google Vertex AI",
    "bedrock": "AWS Bedrock",
    "ollama_chat": "Ollama (locale)",
    "lm_studio": "LM Studio (locale)",
}
# Local servers usually run without a key.
NO_KEY_KINDS = {"ollama", "ollama_chat", "lm_studio", "hosted_vllm", "llamafile"}


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        known = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        for key in ("id", "name", "kind", "model", "base_url", "api_key", "api_key_env"):
            if key in clean:
                clean[key] = str(clean[key] or "").strip()
        for key in ("web_search", "enabled"):
            if key in clean:
                clean[key] = bool(clean[key])
        clean["kind"] = clean.get("kind") or "openai"
        clean["id"] = slugify(clean.get("id") or clean.get("name") or "") or "provider"
        clean["name"] = clean.get("name") or clean["id"]
        return cls(**clean)

    def to_dict(self, include_secret: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_secret:
            data["api_key"] = ""
        return data

    @property
    def kind_label(self) -> str:
        return COMMON_KINDS.get(self.kind, self.kind)

    @property
    def litellm_model(self) -> str:
        """Model name in LiteLLM's ``provider/model`` form."""
        model = self.model.strip()
        return model if model.startswith(self.kind + "/") else f"{self.kind}/{model}"

    def resolved_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "").strip() or None
        return None  # LiteLLM falls back to the provider's standard variable

    def missing_env_keys(self) -> list[str]:
        """Standard variables LiteLLM would need when no key is configured here."""
        if self.resolved_key() or self.kind in NO_KEY_KINDS or not self.model:
            return []
        try:
            import litellm

            info = litellm.validate_environment(model=self.litellm_model, api_base=self.base_url or None)
        except Exception:
            return []
        return [] if info.get("keys_in_environment") else list(info.get("missing_keys") or [])

    def key_source(self) -> str:
        if self.api_key:
            return f"salvata (…{self.api_key[-4:]})" if len(self.api_key) > 8 else "salvata"
        if self.api_key_env:
            return f"da ${self.api_key_env}" + ("" if os.getenv(self.api_key_env, "").strip() else " (non impostata)")
        if self.kind in NO_KEY_KINDS:
            return "non necessaria"
        missing = self.missing_env_keys()
        return f"assente ({' o '.join(missing)})" if missing else "dalla variabile d'ambiente standard"

    def problem(self) -> str:
        """Why this provider cannot run, or an empty string."""
        if not self.enabled:
            return "disattivato"
        if not self.model:
            return "nessun modello impostato"
        if self.api_key_env and not self.api_key and not os.getenv(self.api_key_env, "").strip():
            return f"variabile ${self.api_key_env} non impostata"
        missing = self.missing_env_keys()
        if missing:
            return "chiave API assente"
        return ""

    def is_ready(self) -> bool:
        return not self.problem()


# Ready-made starting points for the "new provider" form.
TEMPLATES: dict[str, dict[str, Any]] = {
    "Google Gemini": {"kind": "gemini", "model": "gemini-2.5-flash", "web_search": True},
    "Anthropic Claude": {"kind": "anthropic", "model": "claude-opus-5-5", "web_search": True},
    "OpenAI": {"kind": "openai", "model": "gpt-4.1-mini"},
    "OpenRouter": {"kind": "openrouter", "model": ""},
    "Mistral": {"kind": "mistral", "model": "mistral-medium-latest"},
    "Groq": {"kind": "groq", "model": ""},
    "xAI Grok": {"kind": "xai", "model": "", "web_search": True},
    "Ollama (locale)": {"kind": "ollama_chat", "base_url": "http://localhost:11434", "model": "qwen2.5vl"},
    "LM Studio (locale)": {"kind": "lm_studio", "base_url": "http://localhost:1234/v1", "model": ""},
    "Endpoint compatibile OpenAI": {"kind": "openai", "base_url": "https://", "model": ""},
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40]


def known_kinds() -> list[str]:
    """Common providers first, then every other provider LiteLLM knows (if installed)."""
    kinds = list(COMMON_KINDS)
    try:
        import litellm

        extra = sorted({str(getattr(p, "value", p)) for p in litellm.provider_list} - set(kinds))
        kinds += extra
    except ImportError:
        pass
    return kinds


def default_providers(gemini_model: str = "gemini-2.5-flash", gemini_search: bool = True) -> tuple[ProviderConfig, ...]:
    """Provider list used until the user saves their own: keeps the .env-only setup working."""
    key_env = "GOOGLE_API_KEY" if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY") else ""
    return (
        ProviderConfig(id="gemini", name="Google Gemini", kind="gemini", model=gemini_model, web_search=gemini_search, api_key_env=key_env),
    )
