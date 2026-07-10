"""Public interfaces for the smart-ask routing application."""

from .application import SmartAsk
from .routing import SmartRouter
from .conversation import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationExecutor,
    ConversationMessage,
    ConversationMetricsStore,
    ConversationRequest,
    SessionContext,
)
from .domain import (
    Attempt,
    Context,
    ExecutionRequest,
    ModelResult,
    RouteResult,
    RoutingEvent,
    RunResult,
    Task,
)
from .executors import (
    HermesExecutor,
    ModelExecutor,
    OllamaConversationExecutor,
    OllamaExecutor,
    OpenRouterConversationExecutor,
    OpenRouterExecutor,
)
from .methods import (
    CascadeRoutingMethod,
    DifficultyClassification,
    DifficultyClassifier,
    DifficultyRoutingMethod,
    EscalationDecision,
    EscalationPolicy,
    FixedRoutingMethod,
    LLMDifficultyClassifier,
    MarkerEscalationPolicy,
    RoutingMethod,
)
from .strategy import (
    LoadedStrategy,
    StrategyBuildError,
    StrategyBuilder,
    StrategyConfig,
    StrategyConfigError,
    load_strategy,
)
from .metrics import (
    CallStats,
    RunStats,
    StatsCollector,
    StatsSummary,
    ResourceReport,
    TaskOutcome,
    TokenUsage,
    aggregate_resources,
    aggregate_stats,
)

__all__ = [
    "Attempt",
    "CallStats",
    "CascadeRoutingMethod",
    "Context",
    "ConversationEvent",
    "ConversationExecutionRequest",
    "ConversationExecutor",
    "ConversationMessage",
    "ConversationMetricsStore",
    "ConversationRequest",
    "ConversationRuntime",
    "DifficultyClassification",
    "DifficultyClassifier",
    "DifficultyRoutingMethod",
    "EscalationDecision",
    "EscalationPolicy",
    "ExecutionRequest",
    "FixedRoutingMethod",
    "HermesExecutor",
    "LLMDifficultyClassifier",
    "LoadedStrategy",
    "MarkerEscalationPolicy",
    "ModelExecutor",
    "ModelResult",
    "OpenRouterExecutor",
    "OpenRouterConversationExecutor",
    "OllamaConversationExecutor",
    "OllamaExecutor",
    "RouteResult",
    "RoutingEvent",
    "RoutingMethod",
    "RunResult",
    "RunStats",
    "ResourceReport",
    "SmartAsk",
    "SmartRouter",
    "SessionContext",
    "StrategyBuildError",
    "StrategyBuilder",
    "StrategyConfig",
    "StrategyConfigError",
    "StatsCollector",
    "StatsSummary",
    "TaskOutcome",
    "Task",
    "TokenUsage",
    "aggregate_stats",
    "aggregate_resources",
    "load_strategy",
]


def __getattr__(name):
    if name == "ConversationRuntime":
        from .conversation import ConversationRuntime

        return ConversationRuntime
    raise AttributeError(name)
