"""
Sub-Agent — 子代理运行时 (含工具沙箱 + Agent 类型)
Maps to: src/tools/AgentTool/AgentTool.ts, utils/forkedAgent.ts

Fork an independent agent with its own message history and tool pool.
孵化独立的子代理，拥有独立的消息历史和工具池。

Architecture:
  - AgentType.WORKER  — background worker launched by coordinator (full tool access)
  - AgentType.INNER   — restricted helper for session extraction, analysis, etc.
  - Tool sandbox      — configurable allowlist/blocklist filtering per agent type
"""
from __future__ import annotations
import logging
from enum import Enum
from typing import Optional

from claw_agent.config import Config
from claw_agent.core.engine import Engine
from claw_agent.core.tool import Tool, ToolContext, ToolRegistry, RiskLevel

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Agent Types — maps to agentId / forkedContextType in AgentTool.ts
# ────────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    """Sub-agent type, determines tool access and behavior.
    Maps to: agentId conventions + ASYNC_AGENT_ALLOWED_TOOLS in AgentTool.ts
    """
    WORKER = "worker"       # Background worker — full tool access (coordinator)
    INNER = "inner"         # Restricted helper — limited tools (session, analysis)


# Default tool allowlist for INNER agents — safe, read-only tools
# Maps to: restrictedTools in AgentTool.ts for internal-use agents
INNER_AGENT_ALLOWED_TOOLS = {
    "read_file",
    "grep",
    "glob",
    "web_search",
    "web_fetch",
    "ask_user_question",
}

# Tools explicitly blocked from ALL sub-agents (for safety)
# Maps to: excludedTools in forkedAgent.ts
BLOCKED_TOOLS = {
    "spawn_worker",     # Prevent recursive worker spawning
    "send_message",     # Only coordinator can message workers
    "task_stop",        # Only coordinator can stop workers
}


# ────────────────────────────────────────────────────────────────
# Tool Sandbox — filter tools based on agent type
# ────────────────────────────────────────────────────────────────

def sandbox_tools(
    tools: list[Tool],
    agent_type: AgentType = AgentType.WORKER,
    *,
    extra_allowed: Optional[set[str]] = None,
    extra_blocked: Optional[set[str]] = None,
) -> list[Tool]:
    """Filter tools based on agent type and sandbox rules.
    Maps to: ASYNC_AGENT_ALLOWED_TOOLS filtering in AgentTool.ts

    Args:
        tools: Full tool list from the parent engine
        agent_type: WORKER gets all tools minus BLOCKED; INNER gets only allowlisted
        extra_allowed: Additional tool names to allow (INNER only)
        extra_blocked: Additional tool names to block (both types)
    """
    blocked = BLOCKED_TOOLS | (extra_blocked or set())

    if agent_type == AgentType.INNER:
        allowed = INNER_AGENT_ALLOWED_TOOLS | (extra_allowed or set())
        return [t for t in tools if t.name in allowed and t.name not in blocked]

    # WORKER: everything except blocked tools
    return [t for t in tools if t.name not in blocked]


# ────────────────────────────────────────────────────────────────
# Core sub-agent runner
# ────────────────────────────────────────────────────────────────

async def run_sub_agent(
    prompt: str,
    config: Config,
    tools: Optional[list[Tool]] = None,
    max_turns: int = 20,
    system_prompt: Optional[str] = None,
    agent_type: AgentType = AgentType.WORKER,
) -> str:
    """Run a sub-agent with isolated context and sandboxed tools.
    Maps to: runForkedAgent() in utils/forkedAgent.ts

    Key isolation properties (from AgentTool.ts):
      - Independent message history
      - Sandboxed tool pool (filtered by agent_type)
      - Budget/turn limits
    """
    # Apply tool sandbox
    effective_tools = tools
    if tools is not None:
        effective_tools = sandbox_tools(tools, agent_type)
        logger.debug(
            f"Sub-agent ({agent_type.value}): "
            f"{len(tools)} tools → {len(effective_tools)} after sandbox"
        )

    # Create isolated config (inherit provider settings, override behavior)
    sub_config = Config(
        provider=config.provider,
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        max_turns=max_turns,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        cwd=config.cwd,
        system_prompt=system_prompt,
        # Sub-agents don't get fallback_model (keep it simple)
        parallel_tool_execution=config.parallel_tool_execution,
    )

    # Create fresh engine with sandboxed tools
    registry = ToolRegistry()
    engine = Engine(config=sub_config, registry=registry, tools=effective_tools)

    # Collect all output
    output_parts = []
    async for event in engine.run(prompt, max_turns=max_turns):
        if event["type"] == "done":
            output_parts.append(event["content"])
        elif event["type"] == "error":
            output_parts.append(f"[sub-agent error] {event['content']}")

    return "\n".join(output_parts) if output_parts else "(no output from sub-agent)"


# ────────────────────────────────────────────────────────────────
# AgentTool — maps to AgentTool in src/tools/AgentTool/AgentTool.ts
# ────────────────────────────────────────────────────────────────

class AgentTool(Tool):
    """Tool that spawns sub-agents with sandboxed tool access.
    Maps to: AgentTool in src/tools/AgentTool/AgentTool.ts
    """
    name = "agent"
    description = (
        "Spawn an independent sub-agent to handle a complex sub-task. "
        "The sub-agent has its own conversation and sandboxed tools."
    )
    risk_level = RiskLevel.MEDIUM
    is_read_only = False

    def __init__(
        self,
        available_tools: Optional[list[Tool]] = None,
        agent_type: AgentType = AgentType.WORKER,
    ):
        self._tools = available_tools
        self._agent_type = agent_type

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task description for the sub-agent",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Max turns for the sub-agent (default: 20)",
                    "default": 20,
                },
            },
            "required": ["prompt"],
        }

    async def call(self, arguments: dict, context: ToolContext) -> str:
        prompt = arguments["prompt"]
        max_turns = arguments.get("max_turns", 20)

        if not context.config:
            return "Error: no config available for sub-agent"

        return await run_sub_agent(
            prompt=prompt,
            config=context.config,
            tools=self._tools,
            max_turns=max_turns,
            agent_type=self._agent_type,
        )
