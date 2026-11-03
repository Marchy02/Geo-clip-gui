"""Runtime settings, read from environment variables (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from .ai.providers import ProviderConfig, default_providers

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "sì"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- LLM arbiter (any LiteLLM provider; configured in the Impostazioni panel) ---
    ai_providers: tuple[ProviderConfig, ...] = ()
    ai_provider: str = "gemini"  # id of the active provider
    ai_timeout_s: int = 120

    # --- GeoCLIP ---
    geoclip_preset: str = "world"
    geoclip_top_k: int = 5

    # --- YOLO ---
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35

    # --- EasyOCR ---
    ocr_languages: list[str] = field(default_factory=lambda: ["en", "it", "de", "fr", "es"])
    ocr_conf: float = 0.2

    # --- Online services ---
    online_lookups: bool = True
    user_agent: str = "GeoOSINT/5.0 (+https://github.com/Marchy02/Geo-clip-gui)"
    http_timeout_s: float = 10.0

    # --- Output / server ---
    reports_dir: Path = PROJECT_ROOT / "reports"
    max_image_mb: int = 40
    server_host: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ai_providers=default_providers(_env_str("GEMINI_MODEL", "gemini-2.5-flash"), _env_bool("GEMINI_SEARCH", True)),
            ai_provider=_env_str("AI_PROVIDER", cls.ai_provider),
            ai_timeout_s=_env_int("AI_TIMEOUT", _env_int("GEMINI_TIMEOUT", cls.ai_timeout_s)),
            geoclip_preset=_env_str("GEOCLIP_PRESET", cls.geoclip_preset),
            geoclip_top_k=_env_int("GEOCLIP_TOP_K", cls.geoclip_top_k),
            yolo_model=_env_str("YOLO_MODEL", cls.yolo_model),
            yolo_conf=_env_float("YOLO_CONF", cls.yolo_conf),
            ocr_languages=_env_list("OCR_LANGUAGES", ["en", "it", "de", "fr", "es"]),
            ocr_conf=_env_float("OCR_CONF", cls.ocr_conf),
            online_lookups=_env_bool("ONLINE_LOOKUPS", cls.online_lookups),
            user_agent=_env_str("HTTP_USER_AGENT", cls.user_agent),
            http_timeout_s=_env_float("HTTP_TIMEOUT", cls.http_timeout_s),
            reports_dir=Path(_env_str("REPORTS_DIR", str(cls.reports_dir))),
            max_image_mb=_env_int("MAX_IMAGE_MB", cls.max_image_mb),
            server_host=_env_str("GRADIO_SERVER_NAME", cls.server_host),
            server_port=_env_int("GRADIO_SERVER_PORT", cls.server_port),
            share=_env_bool("GRADIO_SHARE", cls.share),
        )

    def with_overrides(self, **changes) -> Settings:
        """Return a copy with the given (non-None) fields replaced."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    def with_user_config(self, data: dict) -> Settings:
        """Apply what was saved from the dashboard on top of the environment."""
        from .user_config import general_overrides

        changes: dict = general_overrides(data)
        ai = data.get("ai") or {}
        if isinstance(ai.get("providers"), list):
            providers = []
            for item in ai["providers"]:
                if isinstance(item, dict):
                    p = ProviderConfig.from_dict(item)
                    if all(p.id != q.id for q in providers):
                        providers.append(p)
            changes["ai_providers"] = tuple(providers)
        if ai.get("active"):
            changes["ai_provider"] = str(ai["active"])
        return replace(self, **changes)

    def provider(self, provider_id: str | None = None) -> ProviderConfig | None:
        """The provider with this id, else the active one, else the first enabled one."""
        by_id = {p.id: p for p in self.ai_providers}
        for pid in (provider_id, self.ai_provider):
            if pid and pid in by_id and by_id[pid].enabled:
                return by_id[pid]
        return next((p for p in self.ai_providers if p.enabled), None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from . import user_config

    return Settings.from_env().with_user_config(user_config.load())
