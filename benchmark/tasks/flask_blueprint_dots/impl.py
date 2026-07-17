"""
Minimal reproduction of Flask Blueprint dotted-name bug.
SWE-bench: pallets__flask-4045

Blueprint names containing dots should raise ValueError because dots are
used as separators in nested blueprint routing. The check was added for
endpoint names but not for blueprint names themselves.
"""

from __future__ import annotations


class Blueprint:
    """Simplified Flask Blueprint."""

    def __init__(self, name: str, import_name: str, url_prefix: str = "") -> None:
        # Bug: no validation for dots in name.
        # Endpoint names raise an error but blueprint names do not.
        if not name or name != name.strip():
            raise ValueError("Blueprint name must be non-empty and trimmed")
        # Missing check: if "." in name: raise ValueError(...)
        self.name = name
        self.import_name = import_name
        self.url_prefix = url_prefix
        self._routes: list[tuple[str, str]] = []

    def route(self, rule: str, endpoint: str | None = None):
        """Register a route."""
        def decorator(func):
            ep = endpoint or func.__name__
            if "." in ep:
                raise ValueError(
                    f"Blueprint endpoints must not contain dots: {ep!r}"
                )
            self._routes.append((rule, ep))
            return func
        return decorator

    def register(self, app_blueprints: dict) -> None:
        """Register this blueprint into an app's blueprint registry."""
        if self.name in app_blueprints:
            raise RuntimeError(f"Blueprint {self.name!r} already registered")
        app_blueprints[self.name] = self
