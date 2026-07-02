"""Errors raised while loading or building configured strategies."""


class StrategyConfigError(ValueError):
    """A strategy file is missing, malformed, or semantically invalid."""


class StrategyBuildError(RuntimeError):
    """A valid strategy cannot be built in the current runtime environment."""
