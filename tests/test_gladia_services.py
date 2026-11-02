"""Tests for the Gladia service entries."""

import pytest
from pipecat.services.gladia.stt import GladiaSTTService

from stt_benchmark.config import reset_config
from stt_benchmark.models import ServiceName
from stt_benchmark.pipeline.benchmark_runner import BenchmarkRunner
from stt_benchmark.services import (
    GLADIA_SLOW_PROFILE_WINDOW_SECS,
    STT_SERVICES,
    create_gladia_solaria_3_balanced,
    create_gladia_solaria_3_fast,
    create_gladia_solaria_3_fast_eot,
    create_gladia_solaria_3_fast_finalize,
    create_gladia_solaria_3_medium,
    create_gladia_solaria_3_precise,
    create_gladia_trantoria_1_balanced,
    create_gladia_trantoria_1_fast,
    create_gladia_trantoria_1_fast_eot,
    create_gladia_trantoria_1_medium,
    create_gladia_trantoria_1_precise,
)

ENTERPRISE_ENTRIES = {
    ServiceName.GLADIA_SOLARIA_3_FAST: (create_gladia_solaria_3_fast, "solaria-3-fast", False),
    ServiceName.GLADIA_SOLARIA_3_FAST_EOT: (
        create_gladia_solaria_3_fast_eot,
        "solaria-3-fast + end of turn",
        True,
    ),
    ServiceName.GLADIA_SOLARIA_3_FAST_FINALIZE: (
        create_gladia_solaria_3_fast_finalize,
        "solaria-3-fast + finalize",
        False,
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
    ServiceName.GLADIA_SOLARIA_3_MEDIUM: (
        create_gladia_solaria_3_medium,
        "solaria-3-medium",
        False,
    ),
    ServiceName.GLADIA_SOLARIA_3_BALANCED: (
        create_gladia_solaria_3_balanced,
        "solaria-3-balanced",
        False,
    ),
    ServiceName.GLADIA_SOLARIA_3_PRECISE: (
        create_gladia_solaria_3_precise,
        "solaria-3-precise",
        False,
    ),
    ServiceName.GLADIA_TRANTORIA_1_MEDIUM: (
        create_gladia_trantoria_1_medium,
        "trantoria-1-medium",
        False,
    ),
    ServiceName.GLADIA_TRANTORIA_1_BALANCED: (
        create_gladia_trantoria_1_balanced,
        "trantoria-1-balanced",
        False,
    ),
    ServiceName.GLADIA_TRANTORIA_1_PRECISE: (
        create_gladia_trantoria_1_precise,
        "trantoria-1-precise",
        False,
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

    assert isinstance(service, GladiaSTTService)
    assert service._url == "https://eu1.staging.api.gladia.io/v2/live"
    assert settings["model"] == label.split(" + ")[0]
    assert settings["endpointing"] == 0.3
    assert settings["language_config"] == {"languages": ["en"]}
    assert settings["encoding"] == "wav/pcm"
    assert settings["bit_depth"] == 16
    assert settings["channels"] == 1
    assert "maximum_duration_without_endpointing" not in settings
    assert "pre_processing" not in settings
    assert "streaming_mode" not in settings
    assert service.end_of_turn_enabled is with_end_of_turn
    assert service._enable_finalize is label.endswith(" + finalize")
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


def test_every_model_runs_in_all_four_latency_profiles():
    labels = {d.model_label for d in STT_SERVICES.values() if d.vendor == "Gladia"}

    for model in ("solaria-3", "trantoria-1"):
        for profile in ("fast", "medium", "balanced", "precise"):
            assert f"{model}-{profile}" in labels


SLOW_PROFILES = [
    ServiceName(f"gladia_{model}_{profile}")
    for model in ("solaria_3", "trantoria_1")
    for profile in ("medium", "balanced", "precise")
]


@pytest.mark.parametrize("name", SLOW_PROFILES)
def test_slow_profiles_wait_longer_for_their_final(monkeypatch, name):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")
    entry = STT_SERVICES[name]

    assert entry.transcript_window_secs == GLADIA_SLOW_PROFILE_WINDOW_SECS
    assert entry.factory()._stt_ttfb_timeout == GLADIA_SLOW_PROFILE_WINDOW_SECS
    assert BenchmarkRunner().resolve_transcript_window(name) == GLADIA_SLOW_PROFILE_WINDOW_SECS


@pytest.mark.parametrize(
    "name",
    [
        ServiceName.GLADIA_SOLARIA_3_FAST,
        ServiceName.GLADIA_SOLARIA_3_FAST_EOT,
        ServiceName.GLADIA_TRANTORIA_1_FAST,
        ServiceName.GLADIA_TRANTORIA_1_FAST_EOT,
        ServiceName.DEEPGRAM,
    ],
)
def test_other_services_keep_the_shared_window(monkeypatch, name):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")
    entry = STT_SERVICES[name]

    assert entry.transcript_window_secs is None
    assert BenchmarkRunner().resolve_transcript_window(name) == 2.0
    if entry.vendor == "Gladia":
        assert entry.factory()._stt_ttfb_timeout == 2.0


FINALIZE_ENTRIES = [
    (model, profile)
    for model in ("solaria_3", "trantoria_1")
    for profile in ("fast", "medium", "balanced", "precise")
]


@pytest.mark.parametrize(("model", "profile"), FINALIZE_ENTRIES)
def test_every_profile_also_runs_with_finalize(monkeypatch, model, profile):
    monkeypatch.setenv("GLADIA_ENTERPRISE_API_KEY", "enterprise-key")
    plain = STT_SERVICES[ServiceName(f"gladia_{model}_{profile}")]
    entry = STT_SERVICES[ServiceName(f"gladia_{model}_{profile}_finalize")]

    service = entry.factory()

    assert entry.model_label == f"{plain.model_label} + finalize"
    assert entry.transcript_window_secs == plain.transcript_window_secs
    assert service._enable_finalize
    assert service._prepare_settings()["model"] == plain.model_label
