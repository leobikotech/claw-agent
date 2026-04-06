"""
Engine — 核心 Agent 循环

The heart of the agent system — orchestrates the LLM ↔ Tool loop.
Features:
  - Auto-compact when approaching context window limit
  - Retry with exponential backoff on transient API errors
  - Abort mechanism (cooperative cancellation)
  - Async event queue for background worker notifications
  - **Streaming re-entry**: when background workers are pending, the engine
    suspends at the "no tool calls" exit point, awaits the event queue, injects
    arriving notifications, and re-enters the LLM loop. This mirrors the TS
    QueryEngine's decoupled event processing architecture.
  - Token usage tracking
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, AsyncGenerator, Optional

from claw_agent.core.hooks import HookContext, HookEvent, HookManager, HookResult

from claw_agent.config import Config
from claw_agent.core.messages import (
    AssistantMessage, Message, SystemMessage, ToolCall, ToolResult, UserMessage,
    messages_to_api,
)
from claw_agent.core.tool import Tool, ToolContext, ToolRegistry
from claw_agent.core.permissions import PermissionContext, check_permission
from claw_agent.providers import LLMProvider, create_provider
from claw_agent.instructions.prompts import PromptBuilder
from claw_agent.memory.compact import (
    auto_compact_if_needed,
    CompactTrackingState,
    reactive_compact,
    run_pre_compact_pipeline,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Retry + Error Classification
# ────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

# Max output token recovery nudge message 
MAX_OUTPUT_NUDGE = (
    "Output token limit hit. Resume directly — no apology, no recap of what "
    "you were doing. Pick up mid-thought if that is where the cut happened. "
    "Break remaining work into smaller pieces."
)

# Error pattern lists for classification
RETRYABLE_PATTERNS = [
    "rate_limit", "overloaded", "429", "503", "502",
    "timeout", "connection", "temporarily unavailable",
]
PROMPT_TOO_LONG_PATTERNS = [
    "prompt_too_long", "prompt is too long", "413",
    "maximum context length", "context_length_exceeded",
    "request too large", "too many tokens",
]


def _classify_api_error(error: Exception) -> str:
    """Classify API error for recovery routing.

    Returns: 'retryable' | 'prompt_too_long' | 'fatal'
    """
    err_str = str(error).lower()
    if any(p in err_str for p in PROMPT_TOO_LONG_PATTERNS):
        return "prompt_too_long"
    if any(p in err_str for p in RETRYABLE_PATTERNS):
        return "retryable"
    return "fatal"


def _fill_missing_tool_results(messages: list) -> list:
    """Ensure every tool_use (assistant with tool_calls) has matching tool_results.

    Prevents API rejection due to orphaned tool_use blocks when errors
    occur between tool_use emission and tool_result generation.
    """
    # Collect all tool_call IDs from assistant messages
    pending_ids: set[str] = set()
    for msg in messages:
        if hasattr(msg, 'tool_calls'):
            for tc in msg.tool_calls:
                pending_ids.add(tc.id if hasattr(tc, 'id') else tc.get('id', ''))
        if hasattr(msg, 'tool_call_id'):
            pending_ids.discard(msg.tool_call_id)
        elif isinstance(msg, dict) and msg.get('role') == 'tool':
            pending_ids.discard(msg.get('tool_call_id', ''))

    # Fill missing results
    if not pending_ids:
        return messages

    from claw_agent.core.messages import ToolResult as TR
    filled = list(messages)
    for tid in pending_ids:
        filled.append(TR(
            tool_call_id=tid,
            content="[Tool execution interrupted]",
            is_error=True,
        ))
    logger.warning(f"Filled {len(pending_ids)} missing tool_result(s) for orphaned tool_use blocks")
    return filled


# ────────────────────────────────────────────────────────────────
# Streaming re-entry constants
# ────────────────────────────────────────────────────────────────

# Max seconds to wait for a single event from the queue
EVENT_WAIT_TIMEOUT_S = 60.0

# Safety cap on total re-entry iterations to prevent infinite loops
EVENT_WAIT_MAX_CYCLES = 50

# After this many seconds without an event, inject a nudge so the LLM
# can produce a progress update or take corrective action
EVENT_NUDGE_INTERVAL_S = 30.0

# Terminal status values in <task-notification> XML
_TERMINAL_STATUSES = {"completed", "failed", "killed"}

# Regex to extract task-id and status from notification XML
_RE_TASK_ID = re.compile(r"<task-id>\s*(.+?)\s*</task-id>", re.DOTALL)
_RE_STATUS = re.compile(r"<status>\s*(.+?)\s*</status>", re.DOTALL)


def _parse_notification(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (task_id, status) from a <task-notification> XML string.
    Returns (None, None) if the text is not a notification.
    """
    if "<task-notification>" not in text:
        return None, None
    id_match = _RE_TASK_ID.search(text)
    status_match = _RE_STATUS.search(text)
    return (
        id_match.group(1) if id_match else None,
        status_match.group(1) if status_match else None,
    )


# ────────────────────────────────────────────────────────────────
# Engine — core agent loop
# ────────────────────────────────────────────────────────────────

class Engine:
    """The agent loop — core of the system / 智能体循环——系统核心

    Streaming re-entry architecture:
      When background workers are pending and the LLM produces no tool calls,
      the engine does NOT exit. Instead it suspends on the event queue, waits
      for worker notifications, injects them, and re-enters the LLM loop.
      The loop only exits when: no tool calls AND no pending background tasks.
    """

    def __init__(
        self,
        config: Config,
        registry: Optional[ToolRegistry] = None,
        tools: Optional[list[Tool]] = None,
        permission_ctx: Optional[PermissionContext] = None,
        provider: Optional[LLMProvider] = None,
        event_queue: Optional[asyncio.Queue[str]] = None,
        hook_manager: Optional[HookManager] = None,
        memory: Optional[Any] = None,  # claw_agent.memory.Memory
        session_persistence: Optional[Any] = None,  # claw_agent.memory.SessionPersistence
    ):
        self.config = config
        self.registry = registry or ToolRegistry()
        self.permission_ctx = permission_ctx or PermissionContext()
        self.memory = memory
        self.session_persistence = session_persistence
        self.messages: list[Message] = []

        self.aborted = False

        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        # Auto-compact tracking (maps to AutoCompactTrackingState)
        self._compact_tracking = CompactTrackingState()

        # Async event injection queue (for SubAgent <task-notification>)
        self.event_queue = event_queue

        # Hook manager — lifecycle event system
        self.hook_manager = hook_manager or HookManager()

        # --- Background task registry (streaming re-entry) ---
        # Tracks IDs of workers that are currently running.
        # Populated by coordinator tools via register_background_task(),
        # drained automatically when terminal <task-notification> arrives.
        self._bg_tasks: set[str] = set()

        # Create provider
        if provider:
            self._provider = provider
        else:
            self._provider = create_provider(
                provider=config.provider,
                api_key=config.effective_api_key,
                base_url=config.base_url,
            )

        if tools:
            for t in tools:
                self.registry.register(t)

        # --- Register session persistence hook ---
        if self.session_persistence is not None:
            from claw_agent.memory.session_persistence import create_session_persistence_hook
            self.hook_manager.register(
                HookEvent.POST_SAMPLING,
                create_session_persistence_hook(self.session_persistence, config),
                name="session_persistence",
                fire_and_forget=True,
            )

    # ────────────────────────────────────────────────────────────
    # Background task lifecycle — called by coordinator tools
    # ────────────────────────────────────────────────────────────

    def register_background_task(self, task_id: str) -> None:
        """Register a background worker as active.
        Called by SpawnWorkerTool/SendMessageTool after asyncio.create_task().
        注册一个后台 Worker 为活跃状态。
        """
        self._bg_tasks.add(task_id)
        logger.info(f"Background task registered: '{task_id}' (active: {len(self._bg_tasks)})")

    def complete_background_task(self, task_id: str) -> None:
        """Mark a background worker as finished.
        Called automatically when a terminal <task-notification> is parsed,
        or manually by TaskStopTool on cancellation.
        标记一个后台 Worker 为已完成。
        """
        self._bg_tasks.discard(task_id)
        logger.info(f"Background task completed: '{task_id}' (active: {len(self._bg_tasks)})")

    def has_pending_background_tasks(self) -> bool:
        """Check if any background workers are still running.
        检查是否有后台 Worker 仍在运行。
        """
        return len(self._bg_tasks) > 0

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt via PromptBuilder / 使用 PromptBuilder 组装"""
        builder = PromptBuilder(self.config.cwd)

        # Auto-discover CLAW.md instruction files
        builder.load_instructions()

        if self.config.language:
            builder.set_language(self.config.language)

        # Add system prompt overrides/additions from Config
        if self.config.system_prompt:
            builder.set_domain_instructions(self.config.system_prompt)
        if self.config.append_system_prompt:
            builder.set_memory(self.config.append_system_prompt)

        return builder.build()

    async def _call_llm_with_retry(
        self,
        api_messages: list[dict],
        tool_defs: Optional[list[dict]],
        *,
        max_tokens_override: Optional[int] = None,
    ) -> Any:
        """Call LLM with retry, backoff, and model fallback.
        """
        last_error = None
        current_model = self.config.effective_model
        max_tok = max_tokens_override or self.config.max_tokens

        for attempt in range(MAX_RETRIES + 1):
            if self.aborted:
                raise RuntimeError("Engine aborted")

            try:
                return await self._provider.chat(
                    messages=api_messages,
                    tools=tool_defs,
                    model=current_model,
                    max_tokens=max_tok,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                last_error = e
                err_class = _classify_api_error(e)

                if err_class == "prompt_too_long":
                    # Don't retry — let engine handle via reactive compact
                    raise

                if err_class == "retryable" and attempt < MAX_RETRIES:
                    backoff = min(
                        INITIAL_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1),
                        MAX_BACKOFF_S,
                    )
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

                # Model fallback on final retry failure
                if (
                    self.config.fallback_model
                    and current_model != self.config.fallback_model
                    and attempt == MAX_RETRIES
                ):
                    logger.warning(
                        f"Switching to fallback model '{self.config.fallback_model}' "
                        f"after {attempt + 1} failures on '{current_model}'"
                    )
                    current_model = self.config.fallback_model
                    # Fill any orphaned tool_use blocks before retry
                    self.messages = _fill_missing_tool_results(self.messages)
                    continue

                raise

        raise last_error  # type: ignore

    def _rebuild_messages_from_api(self, api_messages: list[dict]) -> list[Message]:
        """Rebuild Message objects from API-format dicts.

        Used after compact (snip/micro/reactive) modifies the API-format
        message list. Preserves the system prompt from self.messages[0].
        """
        system_prompt = self._build_system_prompt()
        rebuilt: list[Message] = [SystemMessage(system_prompt)]

        for nm in api_messages:
            role = nm.get("role", "user")
            if role == "system":
                continue  # Skip — we already prepend our own system prompt
            elif role == "user":
                rebuilt.append(UserMessage(nm.get("content", "")))
            elif role == "assistant":
                tc_list = []
                for tc in nm.get("tool_calls", []):
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        args = fn.get("arguments", "{}")
                        tc_list.append(ToolCall(
                            id=tc.get("id", ""),
                            name=fn.get("name", ""),
                            arguments=json.loads(args) if isinstance(args, str) else args,
                        ))
                    else:
                        tc_list.append(tc)
                rebuilt.append(AssistantMessage(
                    content=nm.get("content"),
                    tool_calls=tc_list,
                ))
            elif role == "tool":
                rebuilt.append(ToolResult(
                    tool_call_id=nm.get("tool_call_id", ""),
                    content=nm.get("content", ""),
                ))
            else:
                rebuilt.append(UserMessage(nm.get("content", "")))

        return rebuilt

    def _drain_event_queue(self) -> list[str]:
        """Non-blocking drain of all currently-queued events.
        Returns list of event message strings.
        """
        events: list[str] = []
        if self.event_queue:
            while not self.event_queue.empty():
                try:
                    events.append(self.event_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        return events

    def _process_notification(self, event_msg: str) -> None:
        """Parse a <task-notification> and update the background task registry.
        If the notification has a terminal status, remove the task from _bg_tasks.
        """
        task_id, status = _parse_notification(event_msg)
        if task_id and status and status in _TERMINAL_STATUSES:
            self.complete_background_task(task_id)

    async def _wait_and_drain_events(self) -> AsyncGenerator[dict, None]:
        """Streaming re-entry: wait for background task events.

        Suspends on the event queue. For each arriving event:
        1. Parse <task-notification>, update _bg_tasks
        2. Inject as UserMessage into conversation
        3. Yield worker_notification event to caller

        Exits when:
        - All background tasks have finished (normal exit)
        - EVENT_WAIT_MAX_CYCLES reached (safety)
        - Aborted

        After this generator returns, the caller re-enters the LLM loop.
        """
        if not self.event_queue:
            return

        cycles = 0

        while self.has_pending_background_tasks() and cycles < EVENT_WAIT_MAX_CYCLES:
            cycles += 1

            if self.aborted:
                yield {"type": "error", "content": "Engine aborted while waiting for background tasks"}
                return

            yield {
                "type": "waiting",
                "content": f"Waiting for background tasks ({len(self._bg_tasks)} active: {', '.join(sorted(self._bg_tasks))})...",
            }

            try:
                # Block until an event arrives or timeout
                event_msg = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=EVENT_WAIT_TIMEOUT_S,
                )

                # Process the notification (update _bg_tasks registry)
                self._process_notification(event_msg)

                # Inject into conversation history
                self.messages.append(UserMessage(content=event_msg))
                yield {"type": "worker_notification", "content": event_msg}

                # Drain any additional events that arrived concurrently
                for extra_msg in self._drain_event_queue():
                    self._process_notification(extra_msg)
                    self.messages.append(UserMessage(content=extra_msg))
                    yield {"type": "worker_notification", "content": extra_msg}

                # Return to caller — they will re-enter the LLM loop
                return

            except asyncio.TimeoutError:
                if not self.has_pending_background_tasks():
                    # All tasks finished while we were waiting (race condition)
                    return

                # Nudge: inject a system hint so the LLM knows we're still waiting
                nudge_msg = (
                    f"[System] Background tasks still running "
                    f"({len(self._bg_tasks)} active: {', '.join(sorted(self._bg_tasks))}). "
                    f"Waiting for results..."
                )
                self.messages.append(UserMessage(content=nudge_msg))
                yield {"type": "waiting", "content": nudge_msg}

                # Return to caller to re-enter LLM loop — the LLM can produce
                # a progress update or take corrective action (e.g. task_stop)
                return

        if cycles >= EVENT_WAIT_MAX_CYCLES:
            logger.warning(f"Event wait safety cap reached ({EVENT_WAIT_MAX_CYCLES} cycles)")
            yield {
                "type": "error",
                "content": f"Event wait safety cap reached ({EVENT_WAIT_MAX_CYCLES} cycles)",
            }

    # ────────────────────────────────────────────────────────────
    # Main agent loop
    # ────────────────────────────────────────────────────────────

    async def run(
        self,
        prompt: str,
        max_turns: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        """Run the agent loop for a user prompt / 为用户输入运行智能体循环

        Yields events: thinking, tool_call, tool_result, done, error, compact,
                       worker_notification, partial, waiting
        """
        max_turns = max_turns or self.config.max_turns
        turns = 0

        # Initialize with system prompt if fresh conversation
        if not self.messages:
            self.messages.append(SystemMessage(self._build_system_prompt()))

        # --- Initialize Memory Attachments ---
        attachments_text = ""
        if self.memory and prompt.strip():
            attachments_text = await self._build_memory_attachments(prompt)

        # Add user message
        if prompt.strip():
            self.messages.append(UserMessage(prompt + attachments_text))

        while turns < max_turns:
            turns += 1
            self._compact_tracking.turn_counter += 1

            # --- 0. Abort check ---
            if self.aborted:
                yield {"type": "error", "content": "Engine aborted by user"}
                return

            # --- 1. Pre-compact pipeline: snip + micro (zero API cost) ---
            api_messages = messages_to_api(self.messages)
            api_messages, pre_changed = run_pre_compact_pipeline(api_messages)
            if pre_changed:
                self.messages = self._rebuild_messages_from_api(api_messages)

            # --- 2. Auto-compact check (SM compact → Legacy LLM compact) ---
            api_messages = messages_to_api(self.messages)
            # PRE_COMPACT hook
            compact_ctx = HookContext(messages=list(self.messages), engine=self)
            await self.hook_manager.execute(HookEvent.PRE_COMPACT, compact_ctx)

            new_messages, was_compacted = await auto_compact_if_needed(
                messages=api_messages,
                model=self.config.effective_model,
                provider=self._provider,
                tracking=self._compact_tracking,
                session_persistence=self.session_persistence,
            )
            if was_compacted:
                self.messages = self._rebuild_messages_from_api(new_messages)
                yield {
                    "type": "compact",
                    "content": "Conversation compacted to stay within context window.",
                }
                api_messages = messages_to_api(self.messages)

                # POST_COMPACT hook
                post_compact_ctx = HookContext(messages=list(self.messages), engine=self)
                await self.hook_manager.execute(HookEvent.POST_COMPACT, post_compact_ctx)

            # --- 3. Check Event Queue (non-blocking fast path) ---
            pending_events = self._drain_event_queue()
            if pending_events:
                for event_msg in pending_events:
                    self._process_notification(event_msg)
                    self.messages.append(UserMessage(content=event_msg))
                    yield {"type": "worker_notification", "content": event_msg}
                api_messages = messages_to_api(self.messages)

            # --- 4. Call LLM via Provider (with retry + error recovery) ---
            try:
                tool_defs = self.registry.to_api() or None
                response = await self._call_llm_with_retry(api_messages, tool_defs)
            except Exception as e:
                err_class = _classify_api_error(e)

                # --- Reactive Compact recovery (Layer 4) ---
                if err_class == "prompt_too_long":
                    logger.warning("Engine: prompt_too_long — attempting reactive compact")
                    recovered = await reactive_compact(
                        messages=api_messages,
                        model=self.config.effective_model,
                        provider=self._provider,
                        tracking=self._compact_tracking,
                    )
                    if recovered:
                        self.messages = self._rebuild_messages_from_api(recovered)
                        yield {
                            "type": "compact",
                            "content": "Emergency compact: conversation compressed after prompt-too-long.",
                        }
                        continue  # Re-enter loop with compacted history

                # All recovery exhausted
                self.messages = _fill_missing_tool_results(self.messages)
                yield {"type": "error", "content": f"API error: {e}"}
                return

            # Track usage
            if response.usage:
                self.total_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                self.total_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)

            if response.thinking:
                yield {"type": "thinking", "content": response.thinking}

            # --- POST_SAMPLING hook (session persistence fires here) ---
            post_sampling_ctx = HookContext(
                messages=list(self.messages),
                engine=self,
            )
            await self.hook_manager.execute(HookEvent.POST_SAMPLING, post_sampling_ctx)

            # --- 4. Build internal message ---
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=[
                    ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                    for tc in response.tool_calls
                ],
                thinking=response.thinking,
            )
            self.messages.append(assistant_msg)

            # --- 6. No tool calls → check finish_reason + background tasks ---
            if not assistant_msg.tool_calls:
                if (
                    response.finish_reason == "length"
                    and self._compact_tracking.max_output_recovery_count
                    < self.config.max_output_recovery_limit
                ):
                    self._compact_tracking.max_output_recovery_count += 1
                    attempt = self._compact_tracking.max_output_recovery_count
                    logger.info(
                        f"Max output tokens hit — nudge recovery attempt "
                        f"{attempt}/{self.config.max_output_recovery_limit}"
                    )
                    self.messages.append(UserMessage(content=MAX_OUTPUT_NUDGE))
                    if assistant_msg.content:
                        yield {"type": "partial", "content": assistant_msg.content}
                    continue  # Re-enter LLM loop

                if not self.has_pending_background_tasks():
                    # --- STOP hook (turn end) ---
                    stop_ctx = HookContext(messages=list(self.messages), engine=self)
                    stop_results = await self.hook_manager.execute(HookEvent.STOP, stop_ctx)

                    for sr in stop_results:
                        if sr.blocking_error:
                            yield {"type": "error", "content": f"Stop hook blocked: {sr.blocking_error}"}
                            return

                    # Reset max_output counter on clean exit
                    self._compact_tracking.max_output_recovery_count = 0
                    yield {"type": "done", "content": assistant_msg.content or ""}
                    return

                # Background tasks are pending — emit partial response, then wait
                if assistant_msg.content:
                    yield {"type": "partial", "content": assistant_msg.content}

                # Enter streaming re-entry: wait for events from background workers
                # This suspends until at least one event arrives (or timeout + nudge)
                async for event in self._wait_and_drain_events():
                    yield event
                    if event["type"] == "error":
                        return

                # Events injected into self.messages → re-enter LLM loop
                continue

            # --- 7. Execute tool calls (parallel or sequential) ---
            ctx = ToolContext(
                cwd=self.config.cwd,
                config=self.config,
                messages=self.messages,
                aborted=self.aborted,
                engine=self,
            )

            if self.config.parallel_tool_execution and len(assistant_msg.tool_calls) > 1:
                from claw_agent.core.streaming_executor import execute_tools_parallel
                batch_results = await execute_tools_parallel(
                    tool_calls=assistant_msg.tool_calls,
                    registry=self.registry,
                    ctx=ctx,
                    hook_manager=self.hook_manager,
                    permission_ctx=self.permission_ctx,
                )
                for _tc, result, events in batch_results:
                    for ev in events:
                        yield ev
                    self.messages.append(result)
            else:
                # --- Sequential execution (original path) ---
                for tc in assistant_msg.tool_calls:
                    if self.aborted:
                        self.messages = _fill_missing_tool_results(self.messages)
                        yield {"type": "error", "content": "Engine aborted during tool execution"}
                        return

                    yield {"type": "tool_call", "name": tc.name, "arguments": tc.arguments}

                    tool_instance = self.registry.find(tc.name)
                    if not tool_instance:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            content=f"Error: unknown tool '{tc.name}'",
                            is_error=True,
                        )
                    else:
                        # PRE_TOOL_USE hook
                        pre_ctx = HookContext(
                            messages=list(self.messages),
                            tool_name=tc.name,
                            tool_input=tc.arguments,
                            engine=self,
                        )
                        pre_results = await self.hook_manager.execute(
                            HookEvent.PRE_TOOL_USE, pre_ctx
                        )

                        blocked = False
                        effective_args = tc.arguments
                        result: Optional[ToolResult] = None
                        for pr in pre_results:
                            if pr.blocking_error:
                                result = ToolResult(
                                    tool_call_id=tc.id,
                                    content=f"Blocked by hook: {pr.blocking_error}",
                                    is_error=True,
                                )
                                blocked = True
                                break
                            if pr.updated_input is not None:
                                effective_args = pr.updated_input

                        if blocked and result is not None:
                            self.messages.append(result)
                            yield {"type": "tool_result", "name": tc.name, "content": result.content}
                            continue

                        perm = check_permission(tool_instance, effective_args, self.permission_ctx)
                        if not perm.allowed:
                            result = ToolResult(
                                tool_call_id=tc.id,
                                content=f"Permission denied: {perm.reason}",
                                is_error=True,
                            )
                        else:
                            try:
                                output = await tool_instance.call(effective_args, ctx)
                                result = ToolResult(tool_call_id=tc.id, content=output)
                                post_ctx = HookContext(
                                    messages=list(self.messages),
                                    tool_name=tc.name,
                                    tool_input=effective_args,
                                    tool_output=output,
                                    engine=self,
                                )
                                await self.hook_manager.execute(HookEvent.POST_TOOL_USE, post_ctx)
                            except Exception as e:
                                result = ToolResult(
                                    tool_call_id=tc.id,
                                    content=f"Tool error: {e}",
                                    is_error=True,
                                )
                                fail_ctx = HookContext(
                                    messages=list(self.messages),
                                    tool_name=tc.name,
                                    tool_input=effective_args,
                                    tool_error=str(e),
                                    engine=self,
                                )
                                await self.hook_manager.execute(HookEvent.POST_TOOL_FAILURE, fail_ctx)

                    self.messages.append(result)
                    yield {"type": "tool_result", "name": tc.name, "content": result.content}

            # Loop continues: tool results added → next LLM call

        yield {"type": "error", "content": f"Max turns ({max_turns}) reached"}

    def abort(self):
        """Signal the engine to stop at the next safe point.
        """
        self.aborted = True

    def reset(self):
        """Clear conversation history / 清空对话历史"""
        self.messages.clear()
        self.aborted = False
        self._compact_tracking = CompactTrackingState()
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._bg_tasks.clear()
        if self.session_persistence:
            self.session_persistence.reset()

    def feed_event(self, event_msg: str):
        """Manually push an event into the conversation / 丢入异步事件"""
        if self.event_queue is not None:
            self.event_queue.put_nowait(event_msg)
        else:
            self.messages.append(UserMessage(content=event_msg))

    # ────────────────────────────────────────────────────────────
    # Memory helpers
    # ────────────────────────────────────────────────────────────

    async def _build_memory_attachments(self, query: str) -> str:
        """Find relevant memories and format as XML attachment string.
        """
        try:
            if not self.memory:
                return ""
            
            relevant = await self.memory.find_relevant(
                query=query,
                provider=self._provider,
                model=self.config.effective_model,
            )
            if not relevant:
                return ""

            attachment_docs = []
            for m in relevant:
                path = m.get("path")
                if not path:
                    continue
                content = self.memory.load_memory_content(path)
                basename = os.path.basename(path)
                attachment_docs.append(
                    f'<document path="{basename}">\n{content}\n</document>'
                )

            if not attachment_docs:
                return ""

            return (
                "\n\n<system-reminder>\n"
                "The following relevant memories were automatically retrieved "
                "from your persistent memory:\n"
                + "\n".join(attachment_docs)
                + "\n</system-reminder>\n"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch relevant memories: {e}")
            return ""
