"""Tests for the Gladia Enterprise STT service and its end-of-turn handling."""

import json
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies, ExternalUserTurnStrategies

from stt_benchmark.gladia_enterprise import GladiaEnterpriseSTTService


def make_service(**kwargs) -> GladiaEnterpriseSTTService:
    service = GladiaEnterpriseSTTService(
        api_key="key",
        api_url="https://eu1.api.gladia.io/",
        settings=GladiaEnterpriseSTTService.Settings(model="solaria-3-fast"),
        **kwargs,
    )
    service.broadcast_frame = AsyncMock()
    service.push_frame = AsyncMock()
    service._call_event_handler = AsyncMock()
    return service


def transcript(text: str, is_final: bool) -> dict:
    return {"type": "transcript", "data": {"is_final": is_final, "utterance": {"text": text}}}


async def feed(service: GladiaEnterpriseSTTService, *messages) -> list:
    async def websocket():
        for message in messages:
            yield message if isinstance(message, str | bytes) else json.dumps(message)

    return [m async for m in service._handle_turn_messages(websocket())]


def broadcasts(service) -> list:
    return [call.args[0] for call in service.broadcast_frame.call_args_list]


def pushed(service) -> list:
    return [call.args[0] for call in service.push_frame.call_args_list]


def events(service) -> list:
    return [call.args[0] for call in service._call_event_handler.call_args_list]


def test_url_is_built_from_the_region_address():
    service = make_service()

    assert service._url == "https://eu1.api.gladia.io/v2/live"
    assert service._region is None


def test_end_of_turn_is_off_unless_asked():
    service = make_service()

    assert not service.end_of_turn_enabled
    assert "end_of_turn" not in service._prepare_settings()


def test_end_of_turn_options_are_sent_in_the_init_body():
    service = make_service(end_of_turn={"eot_threshold": 0.8, "stop_s": 1.2})

    assert service._prepare_settings()["end_of_turn"] == {"eot_threshold": 0.8, "stop_s": 1.2}


def test_empty_end_of_turn_options_turn_on_the_server_defaults():
    service = make_service(end_of_turn={})

    assert service.end_of_turn_enabled
    assert service._prepare_settings()["end_of_turn"] == {}


def test_eager_end_of_turn_requires_an_eager_threshold():
    with pytest.raises(ValueError, match="eager_eot_threshold"):
        make_service(end_of_turn={}, enable_eager_end_of_turn=True)


def test_eager_end_of_turn_asks_for_partial_transcripts():
    service = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6},
        enable_eager_end_of_turn=True,
    )

    assert service._prepare_settings()["messages_config"]["receive_partial_transcripts"] is True


def test_turn_strategies_are_left_to_gladia_with_end_of_turn():
    plain = make_service()
    with_eot = make_service(end_of_turn={})
    eager = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6},
        enable_eager_end_of_turn=True,
    )

    assert plain.service_metadata_frame().user_turn_strategies is None
    assert isinstance(
        with_eot.service_metadata_frame().user_turn_strategies, ExternalUserTurnStrategies
    )
    assert isinstance(eager.service_metadata_frame().user_turn_strategies, EagerUserTurnStrategies)


async def test_without_end_of_turn_every_message_reaches_pipecat():
    service = make_service()
    messages = [{"type": "speech_start", "data": {}}, {"type": "end_of_turn", "data": {}}]

    passed = await feed(service, *messages)

    assert [json.loads(m) for m in passed] == messages
    assert broadcasts(service) == []
    assert not service._finalize_pending


async def test_transcripts_and_other_messages_still_reach_pipecat():
    service = make_service(end_of_turn={})
    final = transcript("Hello.", True)

    passed = await feed(service, final, {"type": "audio_chunk", "acknowledged": True}, "not json")

    assert [json.loads(m) for m in passed[:2]] == [
        final,
        {"type": "audio_chunk", "acknowledged": True},
    ]
    assert passed[2] == "not json"


async def test_speech_start_proposes_the_start_of_the_turn_once():
    service = make_service(end_of_turn={})

    passed = await feed(
        service, {"type": "speech_start", "data": {}}, {"type": "speech_start", "data": {}}
    )

    assert passed == []
    assert broadcasts(service) == [ProposedUserStartedSpeakingFrame]


async def test_speech_end_does_not_end_the_turn():
    service = make_service(end_of_turn={})

    passed = await feed(
        service, {"type": "speech_start", "data": {}}, {"type": "speech_end", "data": {}}
    )

    assert passed == []
    assert broadcasts(service) == [ProposedUserStartedSpeakingFrame]


async def test_end_of_turn_proposes_the_end_and_finalizes_the_next_final():
    service = make_service(end_of_turn={})
    data = {"channel": 0, "probability": 0.97, "reason": "model", "end": 4.18}

    passed = await feed(
        service, {"type": "speech_start", "data": {}}, {"type": "end_of_turn", "data": data}
    )

    assert passed == []
    assert broadcasts(service) == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert service._finalize_pending
    assert events(service) == ["on_end_of_turn"]
    assert service._call_event_handler.call_args.args[1] == data


async def test_the_final_after_an_end_of_turn_is_reported_as_finalized(monkeypatch):
    service = make_service(end_of_turn={})
    del service.push_frame
    service.stop_ttfb_metrics = AsyncMock()
    monkeypatch.setattr(FrameProcessor, "push_frame", AsyncMock())

    await feed(service, {"type": "end_of_turn", "data": {"reason": "timeout"}})
    first = TranscriptionFrame("Hello.", "", "now")
    second = TranscriptionFrame("Again.", "", "now")
    await service.push_frame(first)
    await service.push_frame(second)

    assert first.finalized
    assert not second.finalized
    service.stop_ttfb_metrics.assert_awaited_once()


async def test_a_new_turn_after_an_end_of_turn_proposes_its_start_again():
    service = make_service(end_of_turn={})

    await feed(
        service,
        {"type": "speech_start", "data": {}},
        {"type": "end_of_turn", "data": {}},
        {"type": "speech_start", "data": {}},
    )

    assert broadcasts(service) == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
        ProposedUserStartedSpeakingFrame,
    ]


async def test_eager_end_of_turn_pushes_the_turn_text_so_far():
    service = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6},
        enable_eager_end_of_turn=True,
    )

    await feed(
        service,
        {"type": "speech_start", "data": {}},
        transcript("My number is", True),
        transcript("zero six", False),
        {"type": "eager_end_of_turn", "data": {"probability": 0.72}},
    )

    (frame,) = pushed(service)
    assert isinstance(frame, EagerTranscriptionFrame)
    assert frame.text == "My number is zero six"
    assert service.eager_end_of_turn_pending
    assert events(service) == ["on_eager_end_of_turn"]


async def test_eager_end_of_turn_prefers_the_text_gladia_sends():
    service = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6, "speculation": True},
        enable_eager_end_of_turn=True,
    )

    await feed(
        service,
        transcript("book a", False),
        {"type": "eager_end_of_turn", "data": {"text": "book a table for two"}},
    )

    assert pushed(service)[0].text == "book a table for two"


async def test_a_new_turn_starts_with_no_text():
    service = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6},
        enable_eager_end_of_turn=True,
    )

    await feed(
        service,
        {"type": "speech_start", "data": {}},
        transcript("First turn.", True),
        {"type": "end_of_turn", "data": {}},
        {"type": "speech_start", "data": {}},
        transcript("second", False),
        {"type": "eager_end_of_turn", "data": {}},
    )

    assert pushed(service)[-1].text == "second"


async def test_turn_resumed_withdraws_the_eager_end_of_turn():
    service = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6},
        enable_eager_end_of_turn=True,
    )

    await feed(
        service,
        transcript("I want", False),
        {"type": "eager_end_of_turn", "data": {}},
        {"type": "turn_resumed", "data": {"end": 4.62}},
    )

    assert isinstance(pushed(service)[-1], EagerEndOfTurnCancelFrame)
    assert not service.eager_end_of_turn_pending
    assert events(service) == ["on_eager_end_of_turn", "on_turn_resumed"]


async def test_end_of_turn_resolves_the_eager_end_of_turn_without_withdrawing_it():
    service = make_service(
        end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6},
        enable_eager_end_of_turn=True,
    )

    await feed(
        service,
        transcript("Done", False),
        {"type": "eager_end_of_turn", "data": {}},
        {"type": "end_of_turn", "data": {}},
    )

    assert not service.eager_end_of_turn_pending
    assert not any(isinstance(frame, EagerEndOfTurnCancelFrame) for frame in pushed(service))


async def test_eager_end_of_turn_is_ignored_when_not_enabled():
    service = make_service(end_of_turn={"eot_threshold": 0.9, "eager_eot_threshold": 0.6})

    passed = await feed(service, {"type": "eager_end_of_turn", "data": {}})

    assert passed == []
    assert pushed(service) == []
    assert not service.eager_end_of_turn_pending


async def test_hold_is_reported_as_an_event():
    service = make_service(end_of_turn={})

    passed = await feed(service, {"type": "hold", "data": {"kind": "phone_number"}})

    assert passed == []
    assert events(service) == ["on_hold"]
