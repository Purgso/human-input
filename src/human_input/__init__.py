from importlib.metadata import PackageNotFoundError, version

from . import settings as settings
from .settings import (
    FAST,
    NORMAL,
    SLOW,
    SUPERHUMAN,
    VERY_FAST,
    VERY_SLOW,
    Settings,
    Speed,
    speed,
)

try:
    __version__ = version("human-input")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "FAST",
    "NORMAL",
    "SLOW",
    "SUPERHUMAN",
    "VERY_FAST",
    "VERY_SLOW",
    "Settings",
    "Speed",
    "__version__",
    "settings",
    "speed",
]
