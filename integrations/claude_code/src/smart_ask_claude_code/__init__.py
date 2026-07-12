"""External Claude Code protocol adapter consuming SmartAsk's public API."""

from .app import create_app
from .catalog import CatalogEntry, StrategyCatalog
from .config import AdapterConfig, AdapterConfigError, load_adapter_config
from .metrics import JsonlSink
from .trace import JsonlTraceSink

__all__ = [
    "AdapterConfig",
    "AdapterConfigError",
    "CatalogEntry",
    "StrategyCatalog",
    "create_app",
    "JsonlSink",
    "JsonlTraceSink",
    "load_adapter_config",
]
