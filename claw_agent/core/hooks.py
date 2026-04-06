"""
Hooks — 生命周期钩子系统

Unified lifecycle hook system for the agent engine.
统一的智能体引擎生命周期钩子系统。

Design:
  - HookEvent enum defines all lifecycle points
  - HookManager is a priority-ordered registry of async callbacks
  - Hooks can be synchronous (blocking) or fire-and-forget (background)
  - Engine fires hooks at key points: tool use, stop, compact, session

Architecture note:
  Production agent systems typically have two hook layers:
  1. User-facing shell-command hooks (utils/hooks.ts, 5000+ lines)
  2. Internal programmatic hooks (postSamplingHooks.ts, 71 lines)

  claw-agent merges both into a single Python callback system.
  Shell command hooks are NOT supported (yet) — only Python callables.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

class HookEvent(str, Enum):
    """Lifecycle events where hooks can be registered.
    生命周期事件——钩子注册点。

    Subset of the original's 20+ events, focused on the core agent loop.
    """

    # Tool lifecycle / 工具生命周期
    PRE_TOOL_USE = "pre_tool_use"       # Before tool execution (can block)
    POST_TOOL_USE = "post_tool_use"     # After tool execution
    POST_TOOL_FAILURE = "post_tool_failure"  # After tool failure

    # Post-sampling / LLM 响应后
    POST_SAMPLING = "post_sampling"     # After each LLM response (before tool exec decision)
                                        # This is where session persistence extraction triggers.

    # Turn lifecycle / 轮次生命周期
    STOP = "stop"                       # LLM produced no tool calls (turn end)
                                        # This is where autoDream fires.

    # Session lifecycle / 会话生命周期
    SESSION_START = "session_start"     # Session begins
    SESSION_END = "session_end"         # Session ends (cleanup)

    # Compact lifecycle / 压缩生命周期
    PRE_COMPACT = "pre_compact"         # Before auto-compact
    POST_COMPACT = "post_compact"       # After auto-compact


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

@dataclass
class HookContext:
    """Context passed to hook callbacks.
    传递给钩子回调的上下文。

    """
    # Full message history at the time of hook invocation
    messages: list = field(default_factory=list)

    # Tool-specific context (populated for PRE/POST_TOOL_USE)
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None

    # Back-reference to the engine (for abort checks, event injection, etc.)
    engine: Optional[Any] = None  # avoids circular import with Engine

    # Extra metadata
    event: Optional[HookEvent] = None
    metadata: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

@dataclass
class HookResult:
    """Result returned by a hook callback.
    钩子回调返回的结果。

    """
    # If set, blocks the operation and returns this error to the LLM
    blocking_error: Optional[str] = None

    # If True, prevents the agent from continuing (hard stop)
    prevent_continuation: bool = False
    stop_reason: Optional[str] = None

    # Additional context to inject into the conversation
    additional_context: Optional[str] = None

    # For PRE_TOOL_USE: modified input to use instead of the original
    updated_input: Optional[dict] = None


# ────────────────────────────────────────────────────────────────
# Hook Callback type
# ────────────────────────────────────────────────────────────────

# A hook callback receives HookContext and optionally returns HookResult.
# Returning None means "no opinion / pass through".
HookCallback = Callable[[HookContext], Awaitable[Optional[HookResult]]]


# ────────────────────────────────────────────────────────────────
# Registered Hook entry
# ────────────────────────────────────────────────────────────────

@dataclass
class _RegisteredHook:
    """Internal: a hook registration with metadata."""
    name: str
    callback: HookCallback
    priority: int  # Lower = runs first (0 = default)
    fire_and_forget: bool  # If True, runs via asyncio.create_task, non-blocking


# ────────────────────────────────────────────────────────────────
# HookManager — maps to postSamplingHooks registry + executeStopHooks
# ────────────────────────────────────────────────────────────────

class HookManager:
    """Lifecycle hook manager — register, execute, introspect.
    生命周期钩子管理器——注册、执行、内省。

      - registerPostSamplingHook / executePostSamplingHooks (postSamplingHooks.ts)
      - executeStopHooks / executePreToolHooks / ... (utils/hooks.ts)

    Usage:
        hooks = HookManager()
        hooks.register(HookEvent.STOP, my_callback, name="auto_dream")
        results = await hooks.execute(HookEvent.STOP, context)
    """

    def __init__(self):
        self._hooks: dict[HookEvent, list[_RegisteredHook]] = {
            event: [] for event in HookEvent
        }
        # Track fire-and-forget tasks for cleanup
        self._background_tasks: set[asyncio.Task] = set()

    def register(
        self,
        event: HookEvent,
        callback: HookCallback,
        *,
        name: str = "",
        priority: int = 0,
        fire_and_forget: bool = False,
    ) -> None:
        """Register a hook callback for an event.
        注册一个钩子回调到某个事件。

        Args:
            event: The lifecycle event to hook into
            callback: Async function(HookContext) -> HookResult | None
            name: Human-readable name for introspection/logging
            priority: Lower values run first (default 0)
            fire_and_forget: If True, callback runs as asyncio.create_task
                             (non-blocking, errors logged but not propagated).
        """
        if not name:
            name = getattr(callback, "__name__", "anonymous")

        entry = _RegisteredHook(
            name=name,
            callback=callback,
            priority=priority,
            fire_and_forget=fire_and_forget,
        )
        self._hooks[event].append(entry)
        # Keep sorted by priority (stable sort)
        self._hooks[event].sort(key=lambda h: h.priority)
        logger.debug(
            f"Hook registered: {name} → {event.value} "
            f"(priority={priority}, fire_and_forget={fire_and_forget})"
        )

    def unregister(self, event: HookEvent, name: str) -> bool:
        """Remove a hook by name. Returns True if found.
        按名称移除钩子。
        """
        hooks = self._hooks[event]
        before = len(hooks)
        self._hooks[event] = [h for h in hooks if h.name != name]
        removed = len(self._hooks[event]) < before
        if removed:
            logger.debug(f"Hook unregistered: {name} from {event.value}")
        return removed

    async def execute(
        self,
        event: HookEvent,
        context: HookContext,
    ) -> list[HookResult]:
        """Execute all hooks for an event, in priority order.
        按优先级顺序执行某事件的所有钩子。

        Synchronous hooks run sequentially (can block/modify).
        Fire-and-forget hooks launch via asyncio.create_task (results discarded).

          - executePostSamplingHooks() for background hooks
          - executeStopHooks() for blocking hooks
          - executePreToolHooks() for blocking+modifying hooks

        Returns:
            List of HookResult from synchronous hooks only.
            Fire-and-forget hooks are not included.
        """
        hooks = self._hooks.get(event, [])
        if not hooks:
            return []

        context.event = event
        results: list[HookResult] = []

        for entry in hooks:
            if entry.fire_and_forget:
                task = asyncio.create_task(
                    self._run_fire_and_forget(entry, context),
                    name=f"hook:{entry.name}",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                logger.debug(f"Hook fired (async): {entry.name}")
            else:
                # Synchronous: run and collect result
                try:
                    result = await entry.callback(context)
                    if result is not None:
                        results.append(result)
                        # Early exit on blocking error
                        if result.blocking_error or result.prevent_continuation:
                            logger.info(
                                f"Hook {entry.name} blocked: "
                                f"{result.blocking_error or result.stop_reason}"
                            )
                            break
                except Exception as e:
                    # Log but don't fail on hook errors
                    logger.error(f"Hook {entry.name} error: {e}")

        return results

    async def _run_fire_and_forget(
        self,
        entry: _RegisteredHook,
        context: HookContext,
    ) -> None:
        """Execute a fire-and-forget hook, logging errors.
        执行一个异步后台钩子，仅记录错误。
        """
        try:
            result = await entry.callback(context)
            if result and result.additional_context:
                logger.info(f"Hook {entry.name} completed: {result.additional_context[:100]}")
        except Exception as e:
            logger.error(f"Hook {entry.name} (background) failed: {e}")

    def clear(self, event: Optional[HookEvent] = None) -> None:
        """Clear hooks. If event is None, clear all.
        清除钩子。event 为 None 时清除全部。
        """
        if event:
            self._hooks[event].clear()
        else:
            for e in HookEvent:
                self._hooks[e].clear()

    def list_hooks(self) -> dict[str, list[dict]]:
        """Introspect: list all registered hooks grouped by event.
        内省：列出所有已注册钩子。
        """
        result: dict[str, list[dict]] = {}
        for event in HookEvent:
            hooks = self._hooks[event]
            if hooks:
                result[event.value] = [
                    {
                        "name": h.name,
                        "priority": h.priority,
                        "fire_and_forget": h.fire_and_forget,
                    }
                    for h in hooks
                ]
        return result

    async def shutdown(self) -> None:
        """Wait for all background tasks to complete (for graceful shutdown).
        等待所有后台任务完成（优雅关闭）。
        """
        if self._background_tasks:
            logger.info(f"Waiting for {len(self._background_tasks)} background hook tasks...")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
