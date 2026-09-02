"""Minimal turn-end event registry for the gym.

An orchestrator emits `turn_end` when its agent loop finishes and control returns to the
user (the ReAct loop stops requesting tools). Handlers registered via `register_turn_end`
receive the live LLM client and the raw request payload (`{"messages": [...], "tools": [...]}`)
— i.e. exactly what the agent last sent to the model — so a handler can branch an
extraction call off the identical, already-cached prefix.

Handlers MUST be non-blocking (return quickly, e.g. enqueue work for a background task).
A handler exception is logged and swallowed so learning never breaks the agent rollout.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# handler(llm_client, context) -> None ; context = {"messages": list, "tools": list}
TurnEndHandler = Callable[[Any, dict], None]

_handlers: list[TurnEndHandler] = []


def register_turn_end(handler: TurnEndHandler) -> None:
    """Register a turn-end handler (idempotent per distinct callable)."""
    if handler not in _handlers:
        _handlers.append(handler)


def clear_turn_end_handlers() -> None:
    """Drop all registered handlers (used by tests / between runs)."""
    _handlers.clear()


def emit_turn_end(llm_client: Any, context: dict) -> None:
    """Notify every registered handler. Errors are logged, never raised."""
    for handler in _handlers:
        try:
            handler(llm_client, context)
        except Exception:
            logger.exception("turn_end handler failed")
