"""Validated pricing catalogs and provider-neutral cost calculation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from .._numeric import checked_fsum, checked_product, is_finite_real

if TYPE_CHECKING:
    from .models import TokenUsage


CostStatus = Literal["priced", "unpriced", "unavailable", "error"]
_REQUIRED_PRICE_FIELDS = frozenset({"input", "output"})
_OPTIONAL_PRICE_FIELDS = frozenset({
    "input_cache_read",
    "input_cache_write",
    "internal_reasoning",
    "request",
})


@dataclass(frozen=True)
class PriceCatalog:
    """An immutable price snapshot with enough provenance to reproduce a run."""

    catalog_id: str
    effective_date: str
    source: str
    prices: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        for name in ("catalog_id", "effective_date", "source"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
        try:
            parsed_date = date.fromisoformat(self.effective_date)
        except ValueError as exc:
            raise ValueError("effective_date must use YYYY-MM-DD") from exc
        if parsed_date.isoformat() != self.effective_date:
            raise ValueError("effective_date must use YYYY-MM-DD")
        if not isinstance(self.prices, Mapping):
            raise TypeError("prices must be a mapping")

        normalized: dict[str, Mapping[str, float]] = {}
        for model, raw_prices in self.prices.items():
            if (
                not isinstance(model, str)
                or not model
                or model != model.strip()
            ):
                raise ValueError(
                    "price catalog model names must be non-empty trimmed strings"
                )
            if not isinstance(raw_prices, Mapping):
                raise TypeError(f"prices for {model!r} must be a mapping")
            fields = set(raw_prices)
            if not _REQUIRED_PRICE_FIELDS <= fields or (
                fields - _REQUIRED_PRICE_FIELDS - _OPTIONAL_PRICE_FIELDS
            ):
                raise ValueError(
                    f"prices for {model!r} must contain input/output and only "
                    "supported optional rates"
                )
            model_prices: dict[str, float] = {}
            for token_kind in sorted(fields):
                value = raw_prices[token_kind]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not is_finite_real(value)
                    or value < 0
                ):
                    raise ValueError(
                        f"{token_kind} price for {model!r} must be finite "
                        "and non-negative"
                    )
                model_prices[token_kind] = float(value)
            normalized[model] = MappingProxyType(model_prices)
        object.__setattr__(self, "prices", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete immutable snapshot in JSON-compatible form."""

        return {
            "catalog_id": self.catalog_id,
            "effective_date": self.effective_date,
            "source": self.source,
            "prices": {
                model: dict(prices)
                for model, prices in self.prices.items()
            },
        }


@dataclass(frozen=True)
class PriceQuote:
    """The cost outcome for one call, including source and failure context."""

    cost_usd: float | None
    status: CostStatus
    source: str | None = None
    catalog: PriceCatalog | None = None
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("priced", "unpriced", "unavailable", "error"):
            raise ValueError(f"unknown cost status: {self.status!r}")
        if self.catalog is not None and not isinstance(self.catalog, PriceCatalog):
            raise TypeError("catalog must be a PriceCatalog or None")
        if self.catalog is not None:
            if self.source is None:
                object.__setattr__(self, "source", self.catalog.catalog_id)
            elif self.source != self.catalog.catalog_id:
                raise ValueError("quote source must match its catalog_id")
        if self.cost_usd is not None and (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, Real)
            or not is_finite_real(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite, non-negative, or None")
        if self.cost_usd is not None:
            object.__setattr__(self, "cost_usd", float(self.cost_usd))

        if self.status == "priced":
            if self.cost_usd is None or not self.source:
                raise ValueError("a priced quote requires cost_usd and source")
        elif self.cost_usd is not None:
            raise ValueError(f"a {self.status} quote cannot contain cost_usd")
        if self.diagnostic is not None and (
            not isinstance(self.diagnostic, str)
            or not self.diagnostic
            or self.diagnostic != self.diagnostic.strip()
        ):
            raise TypeError(
                "quote diagnostic must be a non-empty trimmed string or None"
            )
        if self.status != "priced" and self.diagnostic is None:
            raise ValueError(f"a {self.status} quote requires a diagnostic")
        if self.source is not None and (
            not isinstance(self.source, str)
            or not self.source
            or self.source != self.source.strip()
        ):
            raise ValueError("quote source must be a non-empty trimmed string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "usd": self.cost_usd,
            "status": self.status,
            "source": self.source,
            "catalog_id": (
                self.catalog.catalog_id if self.catalog is not None else None
            ),
            "diagnostic": self.diagnostic,
        }


# Per-token prices captured from the named providers on 2026-07-10.
DEFAULT_PRICE_CATALOG = PriceCatalog(
    catalog_id="provider-pricing-2026-07-10",
    effective_date="2026-07-10",
    source=(
        "https://openrouter.ai/api/v1/models; "
        "https://developers.openai.com/api/docs/models/all; "
        "https://groq.com/pricing"
    ),
    prices={
        "google/gemini-2.5-flash-lite": {
            "input": 0.0000001,
            "output": 0.0000004,
            "input_cache_read": 0.00000001,
            "input_cache_write": 0.00000008333333333333334,
            "internal_reasoning": 0.0000004,
        },
        "anthropic/claude-opus-4.8": {
            "input": 0.000005,
            "output": 0.000025,
            "input_cache_read": 0.0000005,
            "input_cache_write": 0.00000625,
        },
        "gpt-5.1-codex-mini": {
            "input": 0.00000025,
            "output": 0.000002,
            "input_cache_read": 0.000000025,
        },
        "gpt-5.3-codex": {
            "input": 0.00000175,
            "output": 0.000014,
            "input_cache_read": 0.000000175,
        },
        "openai/gpt-oss-20b": {
            "input": 0.000000075,
            "output": 0.0000003,
            "input_cache_read": 0.0000000375,
        },
        "openai/gpt-oss-120b": {
            "input": 0.00000015,
            "output": 0.0000006,
            "input_cache_read": 0.000000075,
        },
    },
)


def price_usage(
    model: str | None,
    usage: "TokenUsage",
    price_catalog: PriceCatalog,
) -> PriceQuote:
    """Price usage with a validated, reproducible catalog snapshot."""

    from .models import TokenUsage

    if model is not None and (
        not isinstance(model, str)
        or not model
        or model != model.strip()
    ):
        raise ValueError("model must be a non-empty trimmed string or None")
    if not isinstance(usage, TokenUsage):
        raise TypeError("usage must be a TokenUsage")
    if not isinstance(price_catalog, PriceCatalog):
        raise TypeError("price_catalog must be a PriceCatalog")
    if model is None:
        return PriceQuote(
            None,
            "unavailable",
            catalog=price_catalog,
            diagnostic="priced model is unavailable",
        )
    if usage.prompt_tokens is None or usage.completion_tokens is None:
        return PriceQuote(
            None,
            "unavailable",
            catalog=price_catalog,
            diagnostic="pricing requires input and output token counts",
        )
    prices = price_catalog.prices.get(model)
    if prices is None:
        return PriceQuote(
            None,
            "unpriced",
            catalog=price_catalog,
            diagnostic=f"model {model!r} is absent from the price catalog",
        )
    detail_requirements = (
        (
            "input_cache_read",
            usage.cached_input_tokens,
            prices["input"],
            "cached-input token count",
        ),
        (
            "input_cache_write",
            usage.cache_write_input_tokens,
            prices["input"],
            "cache-write token count",
        ),
        (
            "internal_reasoning",
            usage.reasoning_tokens,
            prices["output"],
            "reasoning token count",
        ),
    )
    for rate_name, token_count, fallback_rate, detail_name in detail_requirements:
        if (
            rate_name in prices
            and prices[rate_name] != fallback_rate
            and token_count is None
        ):
            return PriceQuote(
                None,
                "unavailable",
                catalog=price_catalog,
                diagnostic=f"pricing requires {detail_name} for {rate_name}",
            )

    cached_read = int(usage.cached_input_tokens or 0)
    cache_write = int(usage.cache_write_input_tokens or 0)
    reasoning = int(usage.reasoning_tokens or 0)
    prompt = int(usage.prompt_tokens or 0)
    completion = int(usage.completion_tokens or 0)
    ordinary_input = prompt - cached_read - cache_write
    ordinary_output = completion - reasoning
    try:
        cost_usd = checked_fsum(
            (
                checked_product(
                    ordinary_input,
                    prices["input"],
                    name="ordinary input price",
                ),
                checked_product(
                    cached_read,
                    prices.get("input_cache_read", prices["input"]),
                    name="cached input price",
                ),
                checked_product(
                    cache_write,
                    prices.get("input_cache_write", prices["input"]),
                    name="cache-write input price",
                ),
                checked_product(
                    ordinary_output,
                    prices["output"],
                    name="ordinary output price",
                ),
                checked_product(
                    reasoning,
                    prices.get("internal_reasoning", prices["output"]),
                    name="reasoning price",
                ),
                prices.get("request", 0.0),
            ),
            name="catalog price",
        )
    except (OverflowError, ValueError):
        return PriceQuote(
            None,
            "error",
            catalog=price_catalog,
            diagnostic=(
                "catalog price cannot be represented as a finite aggregate"
            ),
        )
    return PriceQuote(cost_usd, "priced", catalog=price_catalog)
