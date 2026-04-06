"""
Parallel Tool Executor — 并行工具执行器

Execute multiple tool calls concurrently when the LLM returns a batch.
Production agent systems start tool execution while still streaming the
LLM response; our simplified version executes all tool calls in parallel
after the response is complete — still a major speedup over sequential.

Design:
  - Safe: all tools run via asyncio.gather with exception isolation
  - Ordered: results are returned in the original tool_call order
  - Hook-aware: PRE/POST hooks run for each tool, just like sequential mode
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from claw_agent.core.hooks import HookContext, HookEvent, HookManager
    from claw_agent.core.tool import Tool, ToolContext, ToolRegistry
    from claw_agent.core.permissions import PermissionContext

from claw_agent.core.messages import ToolCall, ToolResult

logger = logging.getLogger(__name__)


async def execute_tools_parallel(
    tool_calls: list[ToolCall],
    registry: Any,            # ToolRegistry
    ctx: Any,                 # ToolContext
    hook_manager: Any,        # HookManager
    permission_ctx: Any,      # PermissionContext
) -> list[tuple[ToolCall, ToolResult, list[dict]]]:
    """Execute tool calls concurrently, returning results in order.

    For each tool call, runs:
      1. PRE_TOOL_USE hook (can block)
      2. Permission check
      3. tool.call()
      4. POST_TOOL_USE / POST_TOOL_FAILURE hook

    Returns list of (tool_call, tool_result, events) tuples in original order.
    Each `events` list contains dicts the Engine should yield to the caller.
    """
    from claw_agent.core.hooks import HookContext, HookEvent
    from claw_agent.core.permissions import check_permission

    async def run_one(tc: ToolCall) -> tuple[ToolCall, ToolResult, list[dict]]:
        """Execute a single tool call with full hook lifecycle."""
        events: list[dict] = []
        events.append({"type": "tool_call", "name": tc.name, "arguments": tc.arguments})

        tool_instance = registry.find(tc.name)
        if not tool_instance:
            result = ToolResult(
                tool_call_id=tc.id,
                content=f"Error: unknown tool '{tc.name}'",
                is_error=True,
            )
            events.append({"type": "tool_result", "name": tc.name, "content": result.content})
            return tc, result, events

        # --- PRE_TOOL_USE hook ---
        pre_ctx = HookContext(
            messages=[],  # Lightweight — don't copy full history per-tool
            tool_name=tc.name,
            tool_input=tc.arguments,
            engine=ctx.engine,
        )
        pre_results = await hook_manager.execute(HookEvent.PRE_TOOL_USE, pre_ctx)

        effective_args = tc.arguments
        for pr in pre_results:
            if pr.blocking_error:
                result = ToolResult(
                    tool_call_id=tc.id,
                    content=f"Blocked by hook: {pr.blocking_error}",
                    is_error=True,
                )
                events.append({"type": "tool_result", "name": tc.name, "content": result.content})
                return tc, result, events
            if pr.updated_input is not None:
                effective_args = pr.updated_input

        # --- Permission check ---
        perm = check_permission(tool_instance, effective_args, permission_ctx)
        if not perm.allowed:
            result = ToolResult(
                tool_call_id=tc.id,
                content=f"Permission denied: {perm.reason}",
                is_error=True,
            )
            events.append({"type": "tool_result", "name": tc.name, "content": result.content})
            return tc, result, events

        # --- Execute tool ---
        try:
            output = await tool_instance.call(effective_args, ctx)
            result = ToolResult(tool_call_id=tc.id, content=output)

            # POST_TOOL_USE hook
            post_ctx = HookContext(
                messages=[],
                tool_name=tc.name,
                tool_input=effective_args,
                tool_output=output,
                engine=ctx.engine,
            )
            await hook_manager.execute(HookEvent.POST_TOOL_USE, post_ctx)

        except Exception as e:
            result = ToolResult(
                tool_call_id=tc.id,
                content=f"Tool error: {e}",
                is_error=True,
            )
            # POST_TOOL_FAILURE hook
            fail_ctx = HookContext(
                messages=[],
                tool_name=tc.name,
                tool_input=effective_args,
                tool_error=str(e),
                engine=ctx.engine,
            )
            await hook_manager.execute(HookEvent.POST_TOOL_FAILURE, fail_ctx)

        events.append({"type": "tool_result", "name": tc.name, "content": result.content})
        return tc, result, events

    # Run all tool calls concurrently
    results = await asyncio.gather(
        *(run_one(tc) for tc in tool_calls),
        return_exceptions=True,
    )

    # Convert exceptions to error results (safety net)
    final: list[tuple[ToolCall, ToolResult, list[dict]]] = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            tc = tool_calls[i]
            result = ToolResult(
                tool_call_id=tc.id,
                content=f"Parallel execution error: {r}",
                is_error=True,
            )
            final.append((tc, result, [
                {"type": "tool_call", "name": tc.name, "arguments": tc.arguments},
                {"type": "tool_result", "name": tc.name, "content": result.content},
            ]))
        else:
            final.append(r)

    return final
