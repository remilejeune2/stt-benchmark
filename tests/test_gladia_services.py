"""Tests for the Gladia service entries."""

import pytest

from stt_benchmark.config import reset_config
from stt_benchmark.models import ServiceName
from stt_benchmark.services import STT_SERVICES, create_gladia_solaria_3_fast


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GLADIA_API_KEY", "GLADIA_ENTERPRISE_API_KEY", "GLADIA_ENTERPRISE_API_URL"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    yield
    reset_config()


def test_solaria_3_fast_uses_the_enterprise_staging_api_by_default(monkeypatch):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")

    service = create_gladia_solaria_3_fast()

    assert service._url == "https://eu1.staging.api.gladia.io/v2/live"
    assert service._api_key == "enterprise-key"
    assert service._region is None


def test_solaria_3_fast_reads_the_api_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_URL", "https://us1.api.gladia.io/")

    service = create_gladia_solaria_3_fast()

    assert service._url == "https://us1.api.gladia.io/v2/live"


def test_solaria_3_fast_reads_the_api_key_from_the_env_file(tmp_path):
    (tmp_path / ".env").write_text("GLADIA_ENTERPRISE_API_KEY=key-from-file\n")

    service = create_gladia_solaria_3_fast()

    assert service._api_key == "key-from-file"


def test_solaria_3_fast_does_not_fall_back_to_the_cloud_api_key(monkeypatch):
    monkeypatch.setenv("GLADIA_API_KEY", "cloud-key")

    with pytest.raises(ValueError, match="GLADIA_ENTERPRISE_API_KEY"):
        create_gladia_solaria_3_fast()


def test_solaria_3_fast_session_settings(monkeypatch):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")

    settings = create_gladia_solaria_3_fast()._prepare_settings()

    assert settings["model"] == "solaria-3-fast"
    assert settings["endpointing"] == 0.64
    assert settings["language_config"] == {"languages": ["en"]}
    assert settings["encoding"] == "wav/pcm"
    assert settings["bit_depth"] == 16
    assert settings["channels"] == 1
    assert "pre_processing" not in settings
    assert "end_of_turn" not in settings
    assert "streaming_mode" not in settings


def test_solaria_3_fast_is_the_current_gladia_model():
    entry = STT_SERVICES[ServiceName.GLADIA_SOLARIA_3_FAST]

    assert entry.factory is create_gladia_solaria_3_fast
    assert entry.vendor == "Gladia"
    assert entry.model_label == "solaria-3-fast"
    assert entry.required_env_vars == ["GLADIA_ENTERPRISE_API_KEY"]
    assert entry.is_current


def test_solaria_1_stays_runnable_but_is_superseded():
    entry = STT_SERVICES[ServiceName.GLADIA]

    assert entry.model_label == "solaria-1"
    assert entry.required_env_vars == ["GLADIA_API_KEY"]
    assert not entry.is_current
