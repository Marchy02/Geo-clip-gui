import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from geoosint import user_config
from geoosint.ai.providers import ProviderConfig
from geoosint.config import get_settings


def test_defaults_without_file():
    s = get_settings()
    assert [p.id for p in s.ai_providers] == ["gemini"]
    assert s.provider().kind == "gemini" and s.provider().web_search


def test_save_load_roundtrip_and_permissions():
    path = user_config.save({"general": {"geoclip_top_k": 9}})
    assert path == user_config.config_path() and path.is_file()
    assert user_config.load()["general"]["geoclip_top_k"] == 9
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert get_settings().geoclip_top_k == 9  # cache invalidated on save


def test_corrupt_file_falls_back_to_defaults():
    path = user_config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{non json", encoding="utf-8")
    assert user_config.load() == {}
    assert get_settings().geoclip_top_k == 5


def test_invalid_general_values_are_ignored():
    user_config.save({"general": {"geoclip_top_k": 99, "ocr_conf": "0.5", "nope": 1, "reports_dir": "~/rep"}})
    s = get_settings()
    assert s.geoclip_top_k == 5 and s.ocr_conf == 0.5
    assert s.reports_dir == Path("~/rep").expanduser()


def test_locked_settings(monkeypatch):
    monkeypatch.setenv(user_config.LOCK_ENV, "true")
    with pytest.raises(user_config.ConfigError):
        user_config.save({})


def test_providers_from_file_and_active_selection():
    user_config.save(
        {
            "ai": {
                "active": "local",
                "providers": [
                    {"id": "claude", "name": "Claude", "kind": "anthropic", "model": "claude-opus-5-5", "api_key": "sk-ant-123456789"},
                    {"id": "local", "name": "Ollama", "kind": "ollama_chat", "model": "qwen2.5vl", "enabled": False},
                ],
            }
        }
    )
    s = get_settings()
    assert [p.id for p in s.ai_providers] == ["claude", "local"]
    assert s.provider().id == "claude"  # the saved active one is disabled: first enabled wins
    assert s.provider("claude").litellm_model == "anthropic/claude-opus-5-5"


def test_provider_key_resolution(monkeypatch):
    p = ProviderConfig(id="x", name="X", kind="openai", model="gpt-4.1-mini", api_key_env="MY_KEY")
    assert p.resolved_key() is None and "non impostata" in p.key_source() and p.problem()
    monkeypatch.setenv("MY_KEY", "abc")
    assert p.resolved_key() == "abc" and p.key_source() == "da $MY_KEY"
    stored = ProviderConfig(id="y", name="Y", model="m", api_key="sk-1234567890")
    assert stored.key_source() == "salvata (…7890)"
    assert ProviderConfig(id="o", name="O", kind="ollama_chat", model="llava").problem() == ""
    assert ProviderConfig(id="m", name="M", kind="openai", model="").problem() == "nessun modello impostato"
    assert ProviderConfig(id="p", name="P", model="openai/gpt-4o").litellm_model == "openai/gpt-4o"


def test_export_blanks_keys():
    data = {"ai": {"providers": [{"id": "a", "api_key": "secret"}]}}
    assert user_config.export_data(data)["ai"]["providers"][0]["api_key"] == ""
    assert user_config.export_data(data, include_secrets=True)["ai"]["providers"][0]["api_key"] == "secret"
    assert data["ai"]["providers"][0]["api_key"] == "secret"  # original untouched


# ---------------------------------------------------------------------------
# Dashboard handlers
# ---------------------------------------------------------------------------

gr = pytest.importorskip("gradio")


def _sh():
    from geoosint.ui import settings_handlers as sh

    return sh


def test_save_edit_activate_delete_provider():
    sh = _sh()
    form = ["Claude lavoro", "anthropic", "claude-opus-5-5", "", "sk-ant-secret-0001", "", True, True]
    out = sh.save_provider(sh.NEW, *form, False)
    assert "salvato" in out[0] and out[-1] == ""  # the key field is cleared, never echoed back
    s = get_settings()
    claude = s.provider("claude-lavoro")
    assert claude.api_key == "sk-ant-secret-0001"
    assert s.provider().id == "gemini"  # not made active
    assert any("(claude-lavoro)" in row[1] for row in sh.providers_rows())
    assert all("secret" not in str(cell) for row in sh.providers_rows() for cell in row)

    # Editing with an empty key keeps the stored one.
    sh.save_provider("claude-lavoro", "Claude lavoro", "anthropic", "claude-sonnet-5-5", "", "", "", False, True, True)
    s = get_settings()
    assert s.provider().id == "claude-lavoro"
    assert s.provider().model == "claude-sonnet-5-5" and s.provider().api_key == "sk-ant-secret-0001"

    form_values = sh.load_provider("claude-lavoro")
    assert form_values[4] == "" and "0001" in form_values[-1]

    sh.set_active("gemini")
    assert get_settings().provider().id == "gemini"
    out = sh.delete_provider("claude-lavoro")
    assert "eliminato" in out[0] and get_settings().provider("claude-lavoro").id == "gemini"


def test_save_provider_validation():
    sh = _sh()
    out = sh.save_provider(sh.NEW, "", "openai", "m", "", "", "", False, True, False)
    assert "Dai un nome" in out[0]
    out = sh.save_provider(sh.NEW, "Bad", "openai", "m", "ftp://x", "", "", False, True, False)
    assert "http" in out[0]


def test_duplicate_names_get_unique_ids():
    sh = _sh()
    sh.save_provider(sh.NEW, "Locale", "ollama_chat", "llava", "", "", "", False, True, False)
    sh.save_provider(sh.NEW, "Locale", "ollama_chat", "qwen2.5vl", "", "", "", False, True, False)
    ids = [p.id for p in get_settings().ai_providers]
    assert "locale" in ids and "locale-2" in ids


def test_general_settings_handlers():
    sh = _sh()
    values = sh.general_values()
    values[sh.GENERAL_ORDER.index("geoclip_top_k")] = 12
    values[sh.GENERAL_ORDER.index("ocr_languages")] = ["Italiano", "Tedesco"]
    assert "salvate" in sh.save_general(*values)
    s = get_settings()
    assert s.geoclip_top_k == 12 and s.ocr_languages == ["it", "de"]

    values[sh.GENERAL_ORDER.index("max_image_mb")] = 0
    assert "non salvate" in sh.save_general(*values)

    reset = sh.reset_general()
    assert "Ripristinati" in reset[-1] and get_settings().geoclip_top_k == 5


def test_export_import_keeps_local_keys(tmp_path):
    sh = _sh()
    sh.save_provider(sh.NEW, "OpenAI", "openai", "gpt-4.1-mini", "", "sk-local-key-1", "", False, True, True)
    exported = Path(sh.export_config(False))
    data = json.loads(exported.read_text(encoding="utf-8"))
    assert data["ai"]["providers"][-1]["api_key"] == ""
    data["general"] = {"geoclip_top_k": 3}
    src = tmp_path / "in.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    out = sh.import_config(str(src))
    assert "importate" in out[0]
    s = get_settings()
    assert s.geoclip_top_k == 3 and s.provider("openai").api_key == "sk-local-key-1"
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert "non riuscita" in sh.import_config(str(bad))[0]


def test_locked_handlers(monkeypatch):
    sh = _sh()
    monkeypatch.setenv(user_config.LOCK_ENV, "1")
    assert "sola lettura" in sh.save_general(*sh.general_values())
    assert "sola lettura" in sh.save_provider(sh.NEW, "X", "openai", "m", "", "", "", False, True, False)[0]


def test_snapshot_shape():
    sh = _sh()
    assert len(sh.snapshot()) == 4 + len(sh.GENERAL_ORDER) + 1


# ---------------------------------------------------------------------------
# Arbiter through LiteLLM (network mocked)
# ---------------------------------------------------------------------------


def _response(text, **extra):
    message = SimpleNamespace(content=text, provider_specific_fields=extra.pop("psf", None), annotations=extra.pop("annotations", None))
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=extra.pop("finish", "stop"))], model="m", **extra)


VERDICT = '{"override": true, "lat": 45.43, "lon": 12.33, "location_name": "Venezia", "country": "Italia", "confidence": 80}'


def test_arbiter_call_and_sources(monkeypatch, plain_png):
    litellm = pytest.importorskip("litellm")
    from geoosint.ai import arbiter

    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        psf = {"citations": [[{"url": "https://example.org/a", "title": "A"}, {"url": "javascript:alert(1)"}]]}
        return _response("```json\n" + VERDICT + "\n```", psf=psf)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    p = ProviderConfig(id="c", name="Claude", kind="anthropic", model="claude-opus-5-5", api_key="sk-x", web_search=True)
    v = arbiter.analyze(p, str(plain_png), "contesto")
    assert v.error is None and v.has_location and v.location_name == "Venezia"
    assert v.sources == [{"title": "A", "url": "https://example.org/a"}]
    assert calls["model"] == "anthropic/claude-opus-5-5" and calls["api_key"] == "sk-x"
    assert "web_search_options" in calls and "temperature" not in calls
    content = calls["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # Per-run overrides: other model, no search, session key.
    calls.clear()
    arbiter.analyze(p, str(plain_png), "ctx", model="claude-haiku-5-5", use_search=False, api_key="sk-session")
    assert calls["model"] == "anthropic/claude-haiku-5-5" and "web_search_options" not in calls and calls["api_key"] == "sk-session"


def test_arbiter_gemini_grounding_and_errors(monkeypatch, plain_png):
    litellm = pytest.importorskip("litellm")
    from geoosint.ai import arbiter

    grounding = [{"groundingChunks": [{"web": {"uri": "https://maps.example/x", "title": "Mappa"}}]}]
    monkeypatch.setattr(litellm, "completion", lambda **kw: _response(VERDICT, vertex_ai_grounding_metadata=grounding))
    p = ProviderConfig(id="g", name="Gemini", kind="gemini", model="gemini-2.5-flash", api_key="AIza-secret")
    v = arbiter.analyze(p, str(plain_png), "ctx")
    assert v.sources == [{"title": "Mappa", "url": "https://maps.example/x"}]

    def boom(**kw):
        raise litellm.AuthenticationError(message="bad key AIza-secret", llm_provider="gemini", model="gemini-2.5-flash")

    monkeypatch.setattr(litellm, "completion", boom)
    v = arbiter.analyze(p, str(plain_png), "ctx")
    assert "chiave API non valida" in v.error and "AIza-secret" not in v.error
    ok, msg = arbiter.test_connection(p)
    assert not ok and "AIza-secret" not in msg

    monkeypatch.setattr(litellm, "completion", lambda **kw: _response("non è json"))
    assert "JSON" in arbiter.analyze(p, str(plain_png), "ctx").error
    monkeypatch.setattr(litellm, "completion", lambda **kw: _response("", finish="length"))
    assert "troncata" in arbiter.analyze(p, str(plain_png), "ctx").error


def test_pipeline_uses_selected_provider(monkeypatch, plain_png):
    litellm = pytest.importorskip("litellm")
    from geoosint.pipeline import AnalysisOptions, Pipeline

    user_config.save({"ai": {"active": "o", "providers": [{"id": "o", "name": "Ollama", "kind": "ollama_chat", "model": "qwen2.5vl"}]}})
    seen = {}
    monkeypatch.setattr(litellm, "completion", lambda **kw: seen.update(kw) or _response(VERDICT))
    opts = AnalysisOptions(metadata=False, forensics=False, geoclip=False, ocr=False, yolo=False, reverse_geocode=False)
    report = Pipeline().run(plain_png, opts)
    assert seen["model"] == "ollama_chat/qwen2.5vl" and report.ai.location_name == "Venezia"
    assert report.final and report.final.source.value == "ai"


def test_pipeline_skips_ai_without_key(plain_png):
    from geoosint.pipeline import AnalysisOptions, Pipeline

    opts = AnalysisOptions(metadata=False, forensics=False, geoclip=False, ocr=False, yolo=False, reverse_geocode=False)
    report = Pipeline().run(plain_png, opts)
    step = next(s for s in report.steps if s.name == "Arbitro IA")
    assert step.status == "skipped" and "chiave" in step.detail
