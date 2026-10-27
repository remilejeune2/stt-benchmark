"""Gladia Enterprise API STT service, with optional server-side end of turn.

The Enterprise API speaks the same ``/v2/live`` protocol as Gladia's cloud API,
so this builds on pipecat's :class:`GladiaSTTService`. With ``end_of_turn`` on,
it follows the shape of pipecat's Deepgram Flux service: Gladia's speech start
and end of turn propose the user turn boundaries, an eager end of turn feeds
pipecat's eager end-of-turn strategy, and the final that closes a turn is
marked finalized so its latency is reported as soon as it arrives.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger
from pipecat.frames.frames import ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin


class GladiaEnterpriseSTTService(EagerEndOfTurnSTTServiceMixin, GladiaSTTService):
    """Gladia STT on the Enterprise API.

    Event handlers available when ``end_of_turn`` is on:

    - on_end_of_turn(service, data)
    - on_eager_end_of_turn(service, data)
    - on_turn_resumed(service, data)
    - on_hold(service, data)

    ``data`` is the ``data`` object of the Gladia message.
    """

    _TURN_MESSAGES = {
        "speech_start",
        "speech_end",
        "end_of_turn",
        "eager_end_of_turn",
        "turn_resumed",
        "hold",
    }

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        end_of_turn: dict[str, Any] | None = None,
        enable_eager_end_of_turn: bool = False,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            api_key: Gladia Enterprise API key.
            api_url: Address of the Enterprise API region, e.g. ``https://eu1.api.gladia.io``.
                The address selects the region, so no ``region`` is sent.
            end_of_turn: Gladia's ``end_of_turn`` options (``{}`` for the server defaults).
                None leaves end-of-turn detection off.
            enable_eager_end_of_turn: Answer Gladia's ``eager_end_of_turn`` ahead of the
                committed one. Requires ``eager_eot_threshold`` in ``end_of_turn``.
            **kwargs: Passed to :class:`GladiaSTTService`.
        """
        if enable_eager_end_of_turn and (end_of_turn or {}).get("eager_eot_threshold") is None:
            raise ValueError("enable_eager_end_of_turn requires end_of_turn.eager_eot_threshold")
        super().__init__(
            api_key=api_key,
            url=f"{api_url.rstrip('/')}/v2/live",
            enable_eager_end_of_turn=enable_eager_end_of_turn,
            **kwargs,
        )
        self._end_of_turn = end_of_turn
        self._turn_finals: list[str] = []
        self._turn_partial = ""
        self._in_turn = False
        for event in ("on_end_of_turn", "on_eager_end_of_turn", "on_turn_resumed", "on_hold"):
            self._register_event_handler(event)

    @property
    def end_of_turn_enabled(self) -> bool:
        """Whether Gladia detects the end of turn for this session."""
        return self._end_of_turn is not None

    def service_metadata_frame(self):
        """Leave turn detection to Gladia when its end of turn is on."""
        frame = super().service_metadata_frame()
        if self.end_of_turn_enabled:
            frame.user_turn_strategies = self.recommended_user_turn_strategies(
                enable_interruptions=self._should_interrupt
            )
        return frame

    def _prepare_settings(self) -> dict[str, Any]:
        settings = super()._prepare_settings()
        if self.end_of_turn_enabled:
            settings["end_of_turn"] = self._end_of_turn
        if self.eager_end_of_turn_enabled:
            settings.setdefault("messages_config", {})["receive_partial_transcripts"] = True
        return settings

    def _get_websocket(self):
        return self._handle_turn_messages(super()._get_websocket())

    async def _handle_turn_messages(self, websocket) -> AsyncIterator[str | bytes]:
        async for message in websocket:
            try:
                content = json.loads(message)
            except json.JSONDecodeError:
                yield message
                continue
            message_type = content.get("type")
            if message_type == "transcript":
                self._track_turn_text(content["data"])
            if not self.end_of_turn_enabled or message_type not in self._TURN_MESSAGES:
                yield message
                continue
            await self._on_turn_message(message_type, content.get("data") or {})

    async def _on_turn_message(self, message_type: str, data: dict[str, Any]):
        match message_type:
            case "speech_start":
                await self._on_turn_started()
            case "end_of_turn":
                await self._on_end_of_turn(data)
            case "eager_end_of_turn":
                await self._push_eager_end_of_turn(
                    data.get("text") or self._turn_text(),
                    user_id=self._user_id,
                    result=data,
                )
                await self._call_event_handler("on_eager_end_of_turn", data)
            case "turn_resumed":
                await self._cancel_eager_end_of_turn()
                await self._call_event_handler("on_turn_resumed", data)
            case "hold":
                await self._call_event_handler("on_hold", data)

    def _track_turn_text(self, data: dict[str, Any]):
        text = data["utterance"]["text"]
        if data["is_final"]:
            self._turn_finals.append(text)
            self._turn_partial = ""
        else:
            self._turn_partial = text

    def _turn_text(self) -> str:
        return " ".join(t.strip() for t in [*self._turn_finals, self._turn_partial] if t.strip())

    async def _on_turn_started(self):
        if self._in_turn:
            return
        self._in_turn = True
        self._turn_finals = []
        self._turn_partial = ""
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _on_end_of_turn(self, data: dict[str, Any]):
        logger.debug(f"{self} end of turn ({data.get('reason')})")
        self._in_turn = False
        self._clear_eager_end_of_turn()
        self.request_finalize()
        self.confirm_finalize()
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        await self._call_event_handler("on_end_of_turn", data)
