"""Handlers of the Impostazioni tab: pure functions, testable without a browser.

Stored API keys never travel back to the browser: the key field is always
rendered empty and an empty value on save means "keep the stored key".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from .. import user_config
from ..ai.providers import COMMON_KINDS, TEMPLATES, ProviderConfig, known_kinds, slugify
from ..config import Settings, get_settings
from . import components as ui
from .handlers import OCR_LANGS

NEW = "__new__"
_CODE_TO_LABEL = {code: label for label, code in OCR_LANGS.items()}


def _ok(text: str) -> str:
    return ui.notice(text, "ok")


def _err(text: str) -> str:
    return ui.notice(text, "bad")


def _locked_msg() -> str | None:
    if user_config.is_locked():
        return _err(f"Le impostazioni sono in sola lettura ({user_config.LOCK_ENV} è attivo).")
    return None


# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------

GENERAL_ORDER = [
    "geoclip_preset",
    "geoclip_top_k",
    "ocr_languages",
    "ocr_conf",
    "yolo_model",
    "yolo_conf",
    "online_lookups",
    "http_timeout_s",
    "user_agent",
    "ai_timeout_s",
    "reports_dir",
    "max_image_mb",
]


def general_values(s: Settings | None = None) -> list[Any]:
    """Current values in GENERAL_ORDER, in the shape the form widgets expect."""
    s = s or get_settings()
    values: list[Any] = []
    for name in GENERAL_ORDER:
        v = getattr(s, name)
        if name == "ocr_languages":
            v = [_CODE_TO_LABEL.get(c, c) for c in v]
        elif name == "reports_dir":
            v = str(v)
        values.append(v)
    return values


def save_general(*values) -> str:
    if msg := _locked_msg():
        return msg
    changes: dict[str, Any] = {}
    try:
        for name, value in zip(GENERAL_ORDER, values, strict=True):
            if name == "ocr_languages":
                value = [OCR_LANGS.get(v, v) for v in value or []]
            changes[name] = user_config.coerce(name, value)
        path = user_config.update(lambda d: d.__setitem__("general", changes))
    except (user_config.ConfigError, OSError) as exc:
        return _err(f"Impostazioni non salvate: {exc}")
    return _ok(f"Impostazioni salvate in {path}. Valgono dalla prossima analisi.")


def reset_general():
    if msg := _locked_msg():
        return [*general_values(), msg]
    try:
        user_config.update(lambda d: d.pop("general", None))
    except (user_config.ConfigError, OSError) as exc:
        return [*general_values(), _err(str(exc))]
    return [*general_values(), _ok("Ripristinati i valori predefiniti (variabili d'ambiente e .env).")]


# ---------------------------------------------------------------------------
# AI providers
# ---------------------------------------------------------------------------

PROVIDER_HEADERS = ["Attivo", "Nome (id)", "Modello LiteLLM", "Endpoint", "Chiave", "Ricerca web", "Stato"]


def providers_rows(s: Settings | None = None) -> list[list[str]]:
    s = s or get_settings()
    active = s.provider()
    return [
        [
            "●" if active and p.id == active.id else "",
            f"{p.name} ({p.id})",
            p.litellm_model if p.model else f"{p.kind}/—",
            p.base_url or "predefinito",
            p.key_source(),
            "sì" if p.web_search else "no",
            p.problem() or "pronto",
        ]
        for p in s.ai_providers
    ]


def provider_choices(s: Settings | None = None, enabled_only: bool = True) -> list[tuple[str, str]]:
    s = s or get_settings()
    return [(f"{p.name} · {p.model or '?'}", p.id) for p in s.ai_providers if p.enabled or not enabled_only]


def editor_choices(s: Settings | None = None) -> list[tuple[str, str]]:
    return [("+ Nuovo provider", NEW), *provider_choices(s, enabled_only=False)]


def kind_choices() -> list[tuple[str, str]]:
    return [(f"{COMMON_KINDS.get(k, k)} ({k})" if k in COMMON_KINDS else k, k) for k in known_kinds()]


def _active_id(s: Settings) -> str | None:
    p = s.provider()
    return p.id if p else None


def active_id() -> str | None:
    return _active_id(get_settings())


def provider_search_default(provider_id: str | None) -> bool:
    p = get_settings().provider(provider_id)
    return bool(p and p.web_search)


def _form(p: ProviderConfig | None, template: str | None = None) -> list[Any]:
    """Editor values: name, kind, model, base_url, api_key (always blank), api_key_env, web_search, enabled, key hint."""
    if p is None:
        t = TEMPLATES.get(template or "", {})
        return [
            template or "",
            t.get("kind", "openai"),
            gr.update(value=t.get("model", ""), choices=[t["model"]] if t.get("model") else []),
            t.get("base_url", ""),
            "",
            t.get("api_key_env", ""),
            bool(t.get("web_search", False)),
            True,
            _hint("Nuovo provider: incolla la chiave oppure indica la variabile d'ambiente che la contiene."),
        ]
    return [
        p.name,
        p.kind,
        gr.update(value=p.model, choices=[p.model] if p.model else []),
        p.base_url,
        "",
        p.api_key_env,
        p.web_search,
        p.enabled,
        _hint(f"Chiave: {p.key_source()}. Lascia il campo vuoto per mantenerla."),
    ]


def _hint(text: str) -> str:
    return f'<p class="go-hint">{ui._e(text)}</p>'


def load_provider(provider_id: str | None, template: str | None = None) -> list[Any]:
    s = get_settings()
    p = next((x for x in s.ai_providers if x.id == provider_id), None)
    return _form(p, template)


def _refresh(s: Settings, selected: str | None) -> list[Any]:
    """Outputs shared by every provider action: table, editor dropdown, active dropdown, analysis dropdown."""
    active = _active_id(s)
    choices = provider_choices(s)
    return [
        providers_rows(s),
        gr.update(choices=editor_choices(s), value=selected or NEW),
        gr.update(choices=choices, value=active),
        gr.update(choices=choices, value=active),
    ]


def _stored(provider_id: str | None) -> ProviderConfig | None:
    return next((p for p in get_settings().ai_providers if p.id == provider_id), None)


def _build(selected, name, kind, model, base_url, api_key, api_key_env, web_search, enabled) -> ProviderConfig:
    """Provider from the editor fields; an empty key keeps the stored one."""
    stored = _stored(selected) if selected != NEW else None
    name = (name or "").strip()
    if not name:
        raise user_config.ConfigError("Dai un nome al provider.")
    kind = (kind or "").strip()
    if not kind:
        raise user_config.ConfigError("Scegli il tipo di provider.")
    base_url = (base_url or "").strip()
    if base_url and not base_url.startswith(("http://", "https://")):
        raise user_config.ConfigError("L'endpoint deve iniziare con http:// o https://")
    if base_url in ("http://", "https://"):
        base_url = ""
    return ProviderConfig.from_dict(
        {
            "id": stored.id if stored else slugify(name),
            "name": name,
            "kind": kind,
            "model": model or "",
            "base_url": base_url,
            "api_key": (api_key or "").strip() or (stored.api_key if stored else ""),
            "api_key_env": api_key_env or "",
            "web_search": web_search,
            "enabled": enabled,
        }
    )


def save_provider(selected, name, kind, model, base_url, api_key, api_key_env, web_search, enabled, make_active):
    s = get_settings()
    if msg := _locked_msg():
        return [msg, *_refresh(s, selected), ""]
    try:
        p = _build(selected, name, kind, model, base_url, api_key, api_key_env, web_search, enabled)
        is_new = selected in (None, "", NEW) or _stored(selected) is None
        if is_new:
            base = p.id
            n = 2
            while any(x.id == p.id for x in s.ai_providers):
                p = ProviderConfig(**{**p.to_dict(), "id": f"{base}-{n}"})
                n += 1
            providers = [*s.ai_providers, p]
        else:
            providers = [p if x.id == p.id else x for x in s.ai_providers]

        def mutate(d: dict) -> None:
            ai = d.setdefault("ai", {})
            ai["providers"] = [x.to_dict() for x in providers]
            current = ai.get("active") or s.ai_provider
            if make_active or current not in {x.id for x in providers if x.enabled}:
                ai["active"] = p.id
            else:
                ai["active"] = current

        user_config.update(mutate)
    except (user_config.ConfigError, OSError) as exc:
        return [_err(f"Provider non salvato: {exc}"), *_refresh(s, selected), ""]
    s = get_settings()
    problem = p.problem()
    text = f"Provider «{p.name}» salvato" + (" e impostato come attivo." if make_active else ".")
    if problem:
        text += f" Da completare: {problem}."
    return [_ok(text), *_refresh(s, p.id), ""]


def delete_provider(selected):
    s = get_settings()
    if msg := _locked_msg():
        return [msg, *_refresh(s, selected)]
    p = _stored(selected)
    if p is None:
        return [_err("Seleziona un provider salvato da eliminare."), *_refresh(s, selected)]
    remaining = [x for x in s.ai_providers if x.id != p.id]

    def mutate(d: dict) -> None:
        ai = d.setdefault("ai", {})
        ai["providers"] = [x.to_dict() for x in remaining]
        if ai.get("active") == p.id or s.ai_provider == p.id:
            ai["active"] = next((x.id for x in remaining if x.enabled), "")

    try:
        user_config.update(mutate)
    except (user_config.ConfigError, OSError) as exc:
        return [_err(str(exc)), *_refresh(s, selected)]
    return [_ok(f"Provider «{p.name}» eliminato."), *_refresh(get_settings(), None)]


def set_active(provider_id):
    s = get_settings()
    if msg := _locked_msg():
        return [msg, *_refresh(s, provider_id)]
    p = _stored(provider_id)
    if p is None or not p.enabled:
        return [_err("Scegli un provider attivo e abilitato."), *_refresh(s, provider_id)]

    def mutate(d: dict) -> None:
        ai = d.setdefault("ai", {})
        ai.setdefault("providers", [x.to_dict() for x in s.ai_providers])
        ai["active"] = p.id

    try:
        user_config.update(mutate)
    except (user_config.ConfigError, OSError) as exc:
        return [_err(str(exc)), *_refresh(s, provider_id)]
    return [_ok(f"«{p.name}» è ora il provider predefinito per le analisi."), *_refresh(get_settings(), p.id)]


def test_provider(selected, name, kind, model, base_url, api_key, api_key_env, web_search, enabled) -> str:
    from ..ai import arbiter

    try:
        p = _build(selected, name or "provider", kind, model, base_url, api_key, api_key_env, web_search, True)
    except user_config.ConfigError as exc:
        return _err(str(exc))
    ok, message = arbiter.test_connection(p, timeout_s=min(get_settings().ai_timeout_s, 60))
    return _ok(message) if ok else _err(message)


def fetch_models(selected, name, kind, model, base_url, api_key, api_key_env, web_search, enabled):
    from ..ai import arbiter

    try:
        p = _build(selected, name or "provider", kind, model, base_url, api_key, api_key_env, web_search, True)
        models = arbiter.list_models(p)
    except Exception as exc:
        return gr.update(), _err(f"Elenco modelli non disponibile: {str(exc)[:300]}. Puoi scrivere il nome del modello a mano.")
    if not models:
        return gr.update(), _err("Il provider non ha restituito modelli: scrivi il nome a mano.")
    value = model if model in models else model or models[0]
    return gr.update(choices=models, value=value), _ok(f"{len(models)} modelli disponibili: sceglilo dal menu.")


# ---------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------


def config_info() -> str:
    path = user_config.config_path()
    rows = [
        ("File delle impostazioni", str(path)),
        ("Stato", "presente" if path.is_file() else "non ancora creato (si crea al primo salvataggio)"),
        ("Modifiche dalla dashboard", "bloccate" if user_config.is_locked() else "consentite"),
        ("Cambia posizione", f"variabile d'ambiente {user_config.CONFIG_ENV}"),
    ]
    note = (
        "<p class='muted'>Ordine di priorità: valori predefiniti, poi variabili d'ambiente e <code>.env</code>, "
        "poi questo file. Le chiavi API salvate qui restano sul computer che esegue GeoOSINT: "
        "se esponi la dashboard in rete proteggila con <code>--auth</code> oppure blocca le modifiche con "
        f"<code>{user_config.LOCK_ENV}=true</code>.</p>"
    )
    return ui.section("Dove sono salvate", ui.kv_table(rows, mono=False) + note, "Configurazione")


def export_config(include_secrets: bool = False) -> str:
    data = user_config.export_data(user_config.load(), include_secrets=include_secrets)
    out = Path(tempfile.mkdtemp(prefix="geoosint_cfg_")) / "geoosint-settings.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def import_config(file_path: str | None):
    s = get_settings()
    if msg := _locked_msg():
        return [msg, config_info(), *general_values(s), *_refresh(s, None)]
    if not file_path:
        return [_err("Carica un file JSON esportato da GeoOSINT."), config_info(), *general_values(s), *_refresh(s, None)]
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("il file non contiene un oggetto JSON")
        Settings.from_env().with_user_config(data)  # validate before writing
        current = user_config.load()

        # Keys blanked by export stay as they are on this machine.
        old = {p.get("id"): p for p in (current.get("ai") or {}).get("providers") or [] if isinstance(p, dict)}
        for p in (data.get("ai") or {}).get("providers") or []:
            if isinstance(p, dict) and not p.get("api_key") and old.get(p.get("id"), {}).get("api_key"):
                p["api_key"] = old[p["id"]]["api_key"]
        user_config.save({k: v for k, v in data.items() if k in {"general", "ai"}})
    except (OSError, ValueError, TypeError, user_config.ConfigError) as exc:
        return [_err(f"Importazione non riuscita: {exc}"), config_info(), *general_values(s), *_refresh(s, None)]
    s = get_settings()
    return [_ok("Impostazioni importate."), config_info(), *general_values(s), *_refresh(s, None)]


def snapshot() -> list[Any]:
    """Everything the tab shows, re-read on page load so a reload never shows stale values."""
    s = get_settings()
    return [*_refresh(s, None), *general_values(s), config_info()]
