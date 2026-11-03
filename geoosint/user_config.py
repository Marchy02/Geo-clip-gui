"""Settings saved from the dashboard (Impostazioni panel) in a JSON file.

Precedence: built-in defaults < environment variables / .env < this file.

Location: ``$GEOOSINT_CONFIG`` if set, otherwise the user's configuration
directory (``~/.config/geoosint/settings.json`` on Linux/macOS,
``%APPDATA%\\GeoOSINT\\settings.json`` on Windows). The file can contain API
keys, so it is written with owner-only permissions where the OS supports it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONFIG_ENV = "GEOOSINT_CONFIG"
LOCK_ENV = "GEOOSINT_SETTINGS_LOCKED"
FORMAT_VERSION = 1
_write_lock = threading.Lock()


class ConfigError(ValueError):
    pass


def config_path() -> Path:
    custom = os.getenv(CONFIG_ENV, "").strip()
    if custom:
        return Path(custom).expanduser()
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA") or Path.home() / "AppData" / "Roaming") / "GeoOSINT"
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config") / "geoosint"
    return base / "settings.json"


def is_locked() -> bool:
    return os.getenv(LOCK_ENV, "").strip().lower() in {"1", "true", "yes", "on", "si", "sì"}


def load(path: Path | None = None) -> dict[str, Any]:
    """Return the saved data, or an empty dict if the file is missing or unreadable."""
    p = path or config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Impostazioni in %s non leggibili, uso i valori predefiniti: %s", p, exc)
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict[str, Any], path: Path | None = None) -> Path:
    """Write atomically (temp file + rename) with 0600 permissions."""
    if is_locked():
        raise ConfigError(f"Impostazioni bloccate dall'amministratore ({LOCK_ENV}).")
    p = path or config_path()
    payload = {**data, "version": FORMAT_VERSION}
    with _write_lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".json", dir=p.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            if os.name == "posix":
                os.chmod(tmp, 0o600)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    _invalidate()
    return p


def update(mutator: Callable[[dict[str, Any]], None], path: Path | None = None) -> Path:
    """Load, change in place, save."""
    data = load(path)
    mutator(data)
    return save(data, path)


def _invalidate() -> None:
    from .config import get_settings

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# General settings editable from the dashboard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # int | float | bool | str | list | path
    lo: float | None = None
    hi: float | None = None


GENERAL_FIELDS: dict[str, Field] = {
    f.name: f
    for f in [
        Field("geoclip_preset", "str"),
        Field("geoclip_top_k", "int", 1, 20),
        Field("ocr_languages", "list"),
        Field("ocr_conf", "float", 0, 1),
        Field("yolo_model", "str"),
        Field("yolo_conf", "float", 0, 1),
        Field("online_lookups", "bool"),
        Field("http_timeout_s", "float", 1, 120),
        Field("user_agent", "str"),
        Field("ai_timeout_s", "int", 10, 900),
        Field("reports_dir", "path"),
        Field("max_image_mb", "int", 1, 1000),
    ]
}


def coerce(name: str, value: Any) -> Any:
    """Validate one general setting; raises ConfigError with an Italian message."""
    field = GENERAL_FIELDS.get(name)
    if field is None:
        raise ConfigError(f"Impostazione sconosciuta: {name}")
    try:
        if field.kind == "int":
            out: Any = int(round(float(value)))
        elif field.kind == "float":
            out = float(value)
        elif field.kind == "bool":
            out = bool(value)
        elif field.kind == "list":
            items = value.split(",") if isinstance(value, str) else list(value or [])
            out = [str(v).strip() for v in items if str(v).strip()]
            if not out:
                raise ConfigError(f"{name}: indica almeno un valore")
        else:
            out = str(value or "").strip()
            if not out:
                raise ConfigError(f"{name}: il valore non può essere vuoto")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{name}: valore non valido ({value!r})") from None
    if field.lo is not None and out < field.lo or field.hi is not None and out > field.hi:
        raise ConfigError(f"{name}: deve essere tra {field.lo:g} e {field.hi:g}")
    return out


def general_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Valid general settings from saved data, converted to Settings field types (bad entries are skipped)."""
    out: dict[str, Any] = {}
    for name, value in (data.get("general") or {}).items():
        try:
            v = coerce(name, value)
        except ConfigError as exc:
            log.warning("Impostazione ignorata: %s", exc)
            continue
        out[name] = Path(v).expanduser() if GENERAL_FIELDS[name].kind == "path" else v
    return out


def export_data(data: dict[str, Any], include_secrets: bool = False) -> dict[str, Any]:
    """Copy of the settings safe to share: stored API keys are blanked unless asked otherwise."""
    out = json.loads(json.dumps(data))
    if not include_secrets:
        for p in (out.get("ai") or {}).get("providers") or []:
            if isinstance(p, dict) and p.get("api_key"):
                p["api_key"] = ""
    out["version"] = FORMAT_VERSION
    return out
