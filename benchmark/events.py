"""Minimal conversation-end event registry for the gym.

An orchestrator emits `conversation_end` when its agent loop finishes and control returns to the
user (the ReAct loop stops requesting tools). Handlers registered via `register_conversation_end`
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
ConversationEndHandler = Callable[[Any, dict], None]

_handlers: list[ConversationEndHandler] = []


def register_conversation_end(handler: ConversationEndHandler) -> None:
    """Register a conversation-end handler (idempotent per distinct callable)."""
    if handler not in _handlers:
        _handlers.append(handler)


def clear_conversation_end_handlers() -> None:
    """Drop all registered handlers (used by tests / between runs)."""
    _handlers.clear()


def emit_conversation_end(llm_client: Any, context: dict) -> None:
    """Notify every registered handler. Errors are logged, never raised."""
    for handler in _handlers:
        try:
            handler(llm_client, context)
        except Exception:
            logger.exception("conversation_end handler failed")
