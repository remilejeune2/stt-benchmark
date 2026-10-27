"""Tests for the Gladia service entries."""

import pytest

from stt_benchmark.config import reset_config
from stt_benchmark.gladia_enterprise import GladiaEnterpriseSTTService
from stt_benchmark.models import ServiceName
from stt_benchmark.services import (
    STT_SERVICES,
    create_gladia_solaria_3_fast,
    create_gladia_solaria_3_fast_eot,
    create_gladia_trantoria_1_fast,
    create_gladia_trantoria_1_fast_eot,
)

ENTERPRISE_ENTRIES = {
    ServiceName.GLADIA_SOLARIA_3_FAST: (create_gladia_solaria_3_fast, "solaria-3-fast", False),
    ServiceName.GLADIA_SOLARIA_3_FAST_EOT: (
        create_gladia_solaria_3_fast_eot,
        "solaria-3-fast + end of turn",
        True,
    ),
    ServiceName.GLADIA_TRANTORIA_1_FAST: (
        create_gladia_trantoria_1_fast,
        "trantoria-1-fast",
        False,
    ),
    ServiceName.GLADIA_TRANTORIA_1_FAST_EOT: (
        create_gladia_trantoria_1_fast_eot,
        "trantoria-1-fast + end of turn",
        True,
    ),
}


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


@pytest.mark.parametrize("name", list(ENTERPRISE_ENTRIES))
def test_enterprise_entries_session_settings(monkeypatch, name):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")
    factory, label, with_end_of_turn = ENTERPRISE_ENTRIES[name]

    service = factory()
    settings = service._prepare_settings()

    assert isinstance(service, GladiaEnterpriseSTTService)
    assert service._url == "https://eu1.staging.api.gladia.io/v2/live"
    assert settings["model"] == label.removesuffix(" + end of turn")
    assert settings["endpointing"] == 0.3
    assert settings["language_config"] == {"languages": ["en"]}
    assert settings["encoding"] == "wav/pcm"
    assert settings["bit_depth"] == 16
    assert settings["channels"] == 1
    assert "maximum_duration_without_endpointing" not in settings
    assert "pre_processing" not in settings
    assert "streaming_mode" not in settings
    if with_end_of_turn:
        assert settings["end_of_turn"] == {}
    else:
        assert "end_of_turn" not in settings


@pytest.mark.parametrize("name", list(ENTERPRISE_ENTRIES))
def test_enterprise_entries_are_registered(name):
    factory, label, _ = ENTERPRISE_ENTRIES[name]
    entry = STT_SERVICES[name]

    assert entry.factory is factory
    assert entry.vendor == "Gladia"
    assert entry.model_label == label
    assert entry.required_env_vars == ["GLADIA_ENTERPRISE_API_KEY"]
    assert entry.is_current


def test_gladia_model_labels_are_unique():
    labels = [d.model_label for d in STT_SERVICES.values() if d.vendor == "Gladia"]

    assert len(labels) == len(set(labels))


def test_solaria_1_stays_runnable_but_is_superseded():
    entry = STT_SERVICES[ServiceName.GLADIA]

    assert entry.model_label == "solaria-1"
    assert entry.required_env_vars == ["GLADIA_API_KEY"]
    assert not entry.is_current
