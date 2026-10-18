"""System audio mute, via the sound server's own API (no shell, works on Wayland).

Linux: PulseAudio / PipeWire-pulse through `pulsectl`. Windows/macOS backends are not implemented yet;
the action then fails gracefully with a clear message.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional, Protocol

from ..errors import ActionError, ActionUnsupported

log = logging.getLogger(__name__)


class AudioBackend(Protocol):
    def get_mute(self) -> bool: ...
    def set_mute(self, muted: bool) -> None: ...


class PulseAudio:
    """Mutes the default output sink."""

    def _sink(self, pulse):
        name = pulse.server_info().default_sink_name
        return pulse.get_sink_by_name(name)

    def get_mute(self) -> bool:
        pulsectl = _import()
        with pulsectl.Pulse("orbit-audio") as p:
            return bool(self._sink(p).mute)

    def set_mute(self, muted: bool) -> None:
        pulsectl = _import()
        try:
            with pulsectl.Pulse("orbit-audio") as p:
                p.sink_mute(self._sink(p).index, bool(muted))
        except Exception as exc:
            if isinstance(exc, ActionError):
                raise
            raise ActionError(f"could not change audio mute: {exc}") from exc


def _import():
    try:
        import pulsectl
    except Exception as exc:
        raise ActionUnsupported("audio control needs the 'pulsectl' package (pip install pulsectl) and PulseAudio/PipeWire") from exc
    return pulsectl


class UnsupportedAudio:
    def __init__(self, platform: str) -> None:
        self.platform = platform

    def get_mute(self) -> bool:
        raise ActionUnsupported(f"audio mute is not implemented on {self.platform} yet")

    def set_mute(self, muted: bool) -> None:
        raise ActionUnsupported(f"audio mute is not implemented on {self.platform} yet")


def default_audio_backend(platform: Optional[str] = None) -> AudioBackend:
    platform = platform or sys.platform
    if platform.startswith("linux"):
        return PulseAudio()
    return UnsupportedAudio(platform)
