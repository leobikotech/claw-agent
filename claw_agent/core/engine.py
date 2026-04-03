"""
Engine — 核心 Agent 循环
Maps to: src/QueryEngine.ts (QueryEngine class, submitMessage)

The heart of the agent system — orchestrates the LLM ↔ Tool loop.
Features:
  - Auto-compact when approaching context window limit
  - Retry with exponential backoff on transient API errors
  - Abort mechanism (cooperative cancellation)
  - Async event queue for background worker notifications
  - Token usage tracking
"""
from __future__ import annotations
import asyncio
import logging
import random
import time
from typing import Any, AsyncGenerator, Optional

from claw_agent.config import Config
from claw_agent.core.messages import (
    AssistantMessage, Message, SystemMessage, ToolCall, ToolResult, UserMessage,
    messages_to_api,
)
from claw_agent.core.tool import Tool, ToolContext, ToolRegistry
from claw_agent.core.permissions import PermissionContext, check_permission
from claw_agent.core.provider import LLMProvider, create_provider
from claw_agent.core.prompts import PromptBuilder
from claw_agent.core.compact import (
    auto_compact_if_needed,
    CompactTrackingState,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Retry configuration — maps to categorizeRetryableAPIError in services/api/errors.ts
# ────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

# Retryable error strings (rate limits, overloaded, network)
RETRYABLE_ERROR_PATTERNS = [
    "rate_limit",
    "overloaded",
    "429",
    "503",
    "502",
    "timeout",
    "connection",
    "temporarily unavailable",
]


def _is_retryable_error(error: Exception) -> bool:
    """Check if an API error is transient and should be retried.
    Maps to: categorizeRetryableAPIError() in services/api/errors.ts
    """
    err_str = str(error).lower()
    return any(pattern in err_str for pattern in RETRYABLE_ERROR_PATTERNS)


# ────────────────────────────────────────────────────────────────
# Engine — core agent loop
# ────────────────────────────────────────────────────────────────

class Engine:
    """The agent loop — core of the system / 智能体循环——系统核心
    Maps to: QueryEngine in src/QueryEngine.ts
    """

    def __init__(
        self,
        config: Config,
        registry: Optional[ToolRegistry] = None,
        tools: Optional[list[Tool]] = None,
        permission_ctx: Optional[PermissionContext] = None,
        provider: Optional[LLMProvider] = None,
        event_queue: Optional[asyncio.Queue[str]] = None,
        memory: Optional[Any] = None,  # claw_agent.memory.Memory
    ):
        self.config = config
        self.registry = registry or ToolRegistry()
        self.permission_ctx = permission_ctx or PermissionContext()
        self.memory = memory
        self.messages: list[Message] = []

        # Abort flag — cooperative cancellation (maps to AbortController in QueryEngine.ts)
        self.aborted = False

        # Token usage tracking (maps to cost-tracker.ts)
        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        # Auto-compact tracking (maps to AutoCompactTrackingState)
        self._compact_tracking = CompactTrackingState()

        # Async event injection queue (for SubAgent <task-notification>)
        self.event_queue = event_queue

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

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt via PromptBuilder / 使用 PromptBuilder 组装"""
        builder = PromptBuilder(self.config.cwd)

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
    ) -> Any:
        """Call LLM with retry + exponential backoff on transient errors.
        Maps to: retry logic in query.ts error handling
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            if self.aborted:
                raise RuntimeError("Engine aborted")

            try:
                response = await self._provider.chat(
                    messages=api_messages,
                    tools=tool_defs,
                    model=self.config.effective_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return response

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES and _is_retryable_error(e):
                    # Exponential backoff with jitter
                    backoff = min(
                        INITIAL_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1),
                        MAX_BACKOFF_S,
                    )
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    raise

        raise last_error  # type: ignore

    async def run(
        self,
        prompt: str,
        max_turns: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        """Run the agent loop for a user prompt / 为用户输入运行智能体循环

        Maps to: QueryEngine.submitMessage() in QueryEngine.ts
        Yields events: thinking, tool_call, tool_result, done, error, compact, worker_notification
        """
        max_turns = max_turns or self.config.max_turns
        turns = 0

        # Initialize with system prompt if fresh conversation
        if not self.messages:
            self.messages.append(SystemMessage(self._build_system_prompt()))

        # --- Initialize Memory Attachments ---
        attachments_text = ""
        if self.memory and prompt.strip():
            try:
                # Find relevant memories using LLM side-query
                # In full implementation, we track already_surfaced across turns
                relevant = await self.memory.find_relevant(
                    query=prompt,
                    provider=self._provider,
                    model=self.config.effective_model,
                )
                if relevant:
                    import os
                    attachment_docs = []
                    for m in relevant:
                        path = m.get("path")
                        if not path:
                            continue
                        content = self.memory.load_memory_content(path)
                        basename = os.path.basename(path)
                        attachment_docs.append(f"<document path=\"{basename}\">\n{content}\n</document>")
                    
                    if attachment_docs:
                        attachments_text = "\n\n<system-reminder>\nThe following relevant memories were automatically retrieved from your persistent memory:\n" + "\n".join(attachment_docs) + "\n</system-reminder>\n"
            except Exception as e:
                logger.warning(f"Failed to fetch relevant memories: {e}")

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

            # --- 1. Auto-compact check ---
            api_messages = messages_to_api(self.messages)
            new_messages, was_compacted = await auto_compact_if_needed(
                messages=api_messages,
                model=self.config.effective_model,
                provider=self._provider,
                tracking=self._compact_tracking,
            )
            if was_compacted:
                # Replace conversation history
                system_prompt = self._build_system_prompt()
                self.messages = [
                    SystemMessage(system_prompt),
                    UserMessage(new_messages[0]["content"]),
                ]
                yield {
                    "type": "compact",
                    "content": "Conversation compacted to stay within context window.",
                }
                api_messages = messages_to_api(self.messages)

            # --- 2. Check Event Queue ---
            if self.event_queue and not self.event_queue.empty():
                while not self.event_queue.empty():
                    event_msg = self.event_queue.get_nowait()
                    self.messages.append(UserMessage(content=event_msg))
                    yield {"type": "worker_notification", "content": event_msg}
                api_messages = messages_to_api(self.messages)

            # --- 3. Call LLM via Provider (with retry) ---
            try:
                tool_defs = self.registry.to_api() or None
                response = await self._call_llm_with_retry(api_messages, tool_defs)
            except Exception as e:
                yield {"type": "error", "content": f"API error: {e}"}
                return

            # Track usage
            if response.usage:
                self.total_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                self.total_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)

            if response.thinking:
                yield {"type": "thinking", "content": response.thinking}

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

            # --- 5. No tool calls → done ---
            if not assistant_msg.tool_calls:
                yield {"type": "done", "content": assistant_msg.content or ""}
                return

            # --- 6. Execute tool calls ---
            ctx = ToolContext(
                cwd=self.config.cwd,
                config=self.config,
                messages=self.messages,
                aborted=self.aborted,
            )

            for tc in assistant_msg.tool_calls:
                # Abort check between tool calls
                if self.aborted:
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
                    # Permission check
                    perm = check_permission(tool_instance, tc.arguments, self.permission_ctx)
                    if not perm.allowed:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            content=f"Permission denied: {perm.reason}",
                            is_error=True,
                        )
                    else:
                        try:
                            output = await tool_instance.call(tc.arguments, ctx)
                            result = ToolResult(tool_call_id=tc.id, content=output)
                        except Exception as e:
                            result = ToolResult(
                                tool_call_id=tc.id,
                                content=f"Tool error: {e}",
                                is_error=True,
                            )

                self.messages.append(result)
                yield {"type": "tool_result", "name": tc.name, "content": result.content}

            # Loop continues: tool results added → next LLM call

        yield {"type": "error", "content": f"Max turns ({max_turns}) reached"}

    def abort(self):
        """Signal the engine to stop at the next safe point.
        Maps to: AbortController.abort() in QueryEngine.ts
        """
        self.aborted = True

    def reset(self):
        """Clear conversation history / 清空对话历史"""
        self.messages.clear()
        self.aborted = False
        self._compact_tracking = CompactTrackingState()
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def feed_event(self, event_msg: str):
        """Manually push an event into the conversation / 丢入异步事件"""
        if self.event_queue is not None:
            self.event_queue.put_nowait(event_msg)
        else:
            self.messages.append(UserMessage(content=event_msg))
