from __future__ import annotations

from ..config import SmartMasteringConfig


class MasteringPresets:
    balanced = staticmethod(SmartMasteringConfig.balanced)
    edm = staticmethod(SmartMasteringConfig.edm)
    vocal = staticmethod(SmartMasteringConfig.vocal)

    @staticmethod
    def names() -> tuple[str, ...]:
        return SmartMasteringConfig.preset_names()


def mastering_preset(name: str) -> SmartMasteringConfig:
    return SmartMasteringConfig.from_preset(name)


def balanced() -> SmartMasteringConfig:
    return SmartMasteringConfig.balanced()


def edm() -> SmartMasteringConfig:
    return SmartMasteringConfig.edm()


def vocal() -> SmartMasteringConfig:
    return SmartMasteringConfig.vocal()


__all__ = [
    "MasteringPresets",
    "balanced",
    "edm",
    "mastering_preset",
    "vocal",
]
