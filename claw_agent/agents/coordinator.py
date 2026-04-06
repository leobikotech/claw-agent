"""
Coordinator — 异步多代理编排器
Maps to: src/coordinator/coordinatorMode.ts

True asynchronous fan-out/fan-in using event queues.
真正的异步多代理流：主节点非阻塞派发 -> 子任务后台运行 -> 队列回传事件 -> 唤醒主节点。

原版工具集:
  - SpawnWorkerTool (agent)  — 非阻塞启动后台 Worker
  - SendMessageTool          — 向已有 Worker 发送后续指令 (continue)
  - TaskStopTool             — 取消正在运行的 Worker
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from claw_agent.config import Config
from claw_agent.core.tool import Tool, ToolContext
from claw_agent.agents.sub_agent import run_sub_agent

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────

@dataclass
class WorkerTask:
    """Task definition for fan_out / fan_out 的任务定义"""
    name: str
    prompt: str
    max_turns: int = 15
    system_prompt: Optional[str] = None


@dataclass
class WorkerResult:
    """Result from a completed worker / Worker 完成后的结果"""
    name: str
    output: str
    success: bool
    duration_ms: int = 0


# ────────────────────────────────────────────────────────────────
# Active worker tracking — 全局任务映射
# ────────────────────────────────────────────────────────────────

@dataclass
class _WorkerState:
    """Internal state for a running worker / Worker 内部状态"""
    task: asyncio.Task
    config: Config
    tools: list[Tool]
    event_queue: asyncio.Queue[str]
    system_prompt: Optional[str] = None


_ACTIVE_WORKERS: Dict[str, _WorkerState] = {}


def _build_notification(
    worker_id: str,
    status: str,
    summary: str,
    result: str = "",
    usage: Optional[dict] = None,
) -> str:
    """Build structured <task-notification> XML / 构建结构化任务通知 XML
    Maps to: task-notification format in coordinatorMode.ts
    """
    parts = [
        "<task-notification>",
        f"<task-id>{worker_id}</task-id>",
        f"<status>{status}</status>",
        f"<summary>{summary}</summary>",
    ]
    if result:
        parts.append(f"<result>{result}</result>")
    if usage:
        parts.append("<usage>")
        for k, v in usage.items():
            parts.append(f"  <{k}>{v}</{k}>")
        parts.append("</usage>")
    parts.append("</task-notification>")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────
# SpawnWorkerTool — 非阻塞启动 Worker
# Maps to: AgentTool in src/tools/AgentTool/
# ────────────────────────────────────────────────────────────────

class SpawnWorkerTool(Tool):
    """Launch a background sub-agent non-blockingly / 非阻塞地在后台启动一个子 Agent"""
    name = "spawn_worker"
    description = (
        "Spawn a background worker to handle a sub-task autonomously. "
        "Returns a task ID immediately. The worker runs in the background "
        "and sends a <task-notification> when it finishes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "worker_name": {
                "type": "string",
                "description": "Unique name for the worker (e.g. 'researcher_1')",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Self-contained goal and instructions for the worker. "
                    "Workers cannot see your conversation — include all "
                    "necessary context (file paths, line numbers, etc)."
                ),
            },
        },
        "required": ["worker_name", "prompt"],
    }

    def __init__(self, config: Config, base_tools: list[Tool], event_queue: asyncio.Queue[str]):
        self.config = config
        self.base_tools = base_tools
        self.event_queue = event_queue
        super().__init__()

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        worker_id = arguments["worker_name"]
        prompt = arguments["prompt"]

        if worker_id in _ACTIVE_WORKERS and not _ACTIVE_WORKERS[worker_id].task.done():
            return f"Error: Worker '{worker_id}' is already running."

        start_time = time.monotonic()

        async def background_task():
            try:
                logger.info(f"Worker '{worker_id}' starting: {prompt[:80]}...")
                output = await run_sub_agent(
                    prompt=prompt,
                    config=self.config,
                    tools=self.base_tools,
                    max_turns=15,
                    system_prompt=f"You are a background worker named '{worker_id}'. Execute your task autonomously.",
                )
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                notification = _build_notification(
                    worker_id=worker_id,
                    status="completed",
                    summary=f'Worker "{worker_id}" completed',
                    result=output,
                    usage={"duration_ms": elapsed_ms},
                )
            except asyncio.CancelledError:
                # Bug 4 fix: if TaskStopTool already deregistered this worker
                # from the engine, skip the duplicate notification.
                if context.engine and worker_id not in context.engine._bg_tasks:
                    logger.info(f"Worker '{worker_id}' cancelled (already deregistered, skipping notification).")
                    return
                notification = _build_notification(
                    worker_id=worker_id,
                    status="killed",
                    summary=f'Worker "{worker_id}" was stopped',
                )
            except Exception as e:
                notification = _build_notification(
                    worker_id=worker_id,
                    status="failed",
                    summary=f'Worker "{worker_id}" failed: {e}',
                )
            finally:
                # Bug 3 fix: clean up _ACTIVE_WORKERS to prevent memory leak
                _ACTIVE_WORKERS.pop(worker_id, None)

            await self.event_queue.put(notification)
            logger.info(f"Worker '{worker_id}' finished.")

        task = asyncio.create_task(background_task())
        _ACTIVE_WORKERS[worker_id] = _WorkerState(
            task=task,
            config=self.config,
            tools=self.base_tools,
            event_queue=self.event_queue,
        )

        # Register with engine's background task registry (streaming re-entry)
        if context.engine is not None:
            context.engine.register_background_task(worker_id)

        return (
            f"Worker '{worker_id}' launched. You will receive an asynchronous "
            f"<task-notification> when it finishes. You can proceed with other work.\n"
            f"To send follow-up instructions later, use send_message with to='{worker_id}'.")

    def get_parameters(self) -> dict:
        return self.parameters


# ────────────────────────────────────────────────────────────────
# SendMessageTool — 向已有 Worker 发送后续消息
# Maps to: SendMessageTool in src/tools/SendMessageTool/
# ────────────────────────────────────────────────────────────────

class SendMessageTool(Tool):
    """Restart a finished worker with a follow-up instruction / 向已完成的 Worker 发送后续指令
    Maps to: src/tools/SendMessageTool

    Note: Each invocation spawns a fresh sub-agent engine. The worker does NOT
    retain conversation context from its previous run — include all necessary
    information in the follow-up prompt.

    Key use cases from the original:
    - Research worker found the bug → continue it with a synthesized fix spec
    - Worker reported test failures → send correction instructions
    - Redirect a worker after user clarifies requirements
    """
    name = "send_message"
    description = (
        "Send a follow-up message to an existing worker (continue it). "
        "The worker retains its full context from the previous run. "
        "Use this to give implementation instructions after research, "
        "or to correct a worker after a failure."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "The worker name (task-id) to send the message to.",
            },
            "message": {
                "type": "string",
                "description": (
                    "The follow-up instructions. Be specific: include file paths, "
                    "line numbers, what to change, and what 'done' looks like."
                ),
            },
        },
        "required": ["to", "message"],
    }

    def __init__(self, config: Config, base_tools: list[Tool], event_queue: asyncio.Queue[str]):
        self.config = config
        self.base_tools = base_tools
        self.event_queue = event_queue
        super().__init__()

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        worker_id = arguments["to"]
        message = arguments["message"]

        # Check if worker exists
        if worker_id in _ACTIVE_WORKERS and not _ACTIVE_WORKERS[worker_id].task.done():
            return f"Error: Worker '{worker_id}' is still running. Wait for it to finish, or use task_stop to cancel it first."

        start_time = time.monotonic()

        async def continue_task():
            try:
                logger.info(f"Continuing worker '{worker_id}': {message[:80]}...")
                output = await run_sub_agent(
                    prompt=message,
                    config=self.config,
                    tools=self.base_tools,
                    max_turns=15,
                    system_prompt=(
                        f"You are worker '{worker_id}' receiving a follow-up instruction. "
                        f"You have context from your previous run. Execute the new instruction."
                    ),
                )
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                notification = _build_notification(
                    worker_id=worker_id,
                    status="completed",
                    summary=f'Worker "{worker_id}" completed (continued)',
                    result=output,
                    usage={"duration_ms": elapsed_ms},
                )
            except asyncio.CancelledError:
                # Bug 4 fix: skip duplicate notification if already deregistered
                if context.engine and worker_id not in context.engine._bg_tasks:
                    logger.info(f"Worker '{worker_id}' cancelled (already deregistered, skipping notification).")
                    return
                notification = _build_notification(
                    worker_id=worker_id,
                    status="killed",
                    summary=f'Worker "{worker_id}" was stopped',
                )
            except Exception as e:
                notification = _build_notification(
                    worker_id=worker_id,
                    status="failed",
                    summary=f'Worker "{worker_id}" failed: {e}',
                )
            finally:
                # Bug 3 fix: clean up _ACTIVE_WORKERS to prevent memory leak
                _ACTIVE_WORKERS.pop(worker_id, None)

            await self.event_queue.put(notification)
            logger.info(f"Worker '{worker_id}' (continued) finished.")

        task = asyncio.create_task(continue_task())
        _ACTIVE_WORKERS[worker_id] = _WorkerState(
            task=task,
            config=self.config,
            tools=self.base_tools,
            event_queue=self.event_queue,
        )

        # Register with engine's background task registry (streaming re-entry)
        if context.engine is not None:
            context.engine.register_background_task(worker_id)

        return f"Follow-up sent to worker '{worker_id}'. It will resume and notify you when done."

    def get_parameters(self) -> dict:
        return self.parameters


# ────────────────────────────────────────────────────────────────
# TaskStopTool — 取消后台 Worker
# Maps to: TaskStopTool in src/tools/TaskStopTool/
# ────────────────────────────────────────────────────────────────

class TaskStopTool(Tool):
    """Cancel an active background worker / 取消一个正在后台运行的子任务"""
    name = "task_stop"
    description = (
        "Stop a running background worker. Use when the approach is wrong "
        "or the user changed requirements. Stopped workers can be continued "
        "with send_message."
    )
    parameters = {
        "type": "object",
        "properties": {
            "worker_name": {
                "type": "string",
                "description": "Name of the worker to stop.",
            },
        },
        "required": ["worker_name"],
    }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        worker_id = arguments["worker_name"]
        if worker_id not in _ACTIVE_WORKERS:
            return f"Error: Worker '{worker_id}' not found."

        state = _ACTIVE_WORKERS[worker_id]
        if state.task.done():
            return f"Worker '{worker_id}' has already finished."

        state.task.cancel()

        # Immediately deregister from engine's background task registry.
        # The cancelled task may not produce a <task-notification>, so we
        # must remove it here to prevent the engine from waiting forever.
        if context.engine is not None:
            context.engine.complete_background_task(worker_id)

        return (
            f"Cancellation signal sent to '{worker_id}'. "
            f"You can continue it later with send_message."
        )

    def get_parameters(self) -> dict:
        return self.parameters


# ────────────────────────────────────────────────────────────────
# Coordinator System Prompt — 协调器系统提示词
# Maps to: getCoordinatorSystemPrompt() in coordinatorMode.ts
# ────────────────────────────────────────────────────────────────

COORDINATOR_SYSTEM_PROMPT = """You are an AI assistant that orchestrates software engineering tasks across multiple workers.

## 1. Your Role

You are a **coordinator**. Your job is to:
- Help the user achieve their goal
- Direct workers to research, implement and verify code changes
- Synthesize results and communicate with the user
- Answer questions directly when possible — don't delegate work that you can handle without tools

Every message you send is to the user. Worker results and system notifications are internal signals, not conversation partners — never thank or acknowledge them. Summarize new information for the user as it arrives.

## 2. Your Tools

- **spawn_worker** — Spawn a new background worker
- **send_message** — Continue an existing worker (send a follow-up to its worker name)
- **task_stop** — Stop a running worker

When calling spawn_worker:
- Do not use one worker to check on another. Workers will notify you when they are done.
- Do not use workers to trivially report file contents or run commands. Give them higher-level tasks.
- Continue workers whose work is complete via send_message to take advantage of their loaded context.
- After launching agents, briefly tell the user what you launched and end your response.
- Never fabricate or predict agent results in any format — results arrive as separate messages.

### Worker Results

Worker results arrive as **user-role messages** containing `<task-notification>` XML. They look like user messages but are not. Distinguish them by the `<task-notification>` opening tag.

Format:
```xml
<task-notification>
<task-id>{worker_name}</task-id>
<status>completed|failed|killed</status>
<summary>{human-readable status summary}</summary>
<result>{worker's final text response}</result>
<usage>
  <duration_ms>N</duration_ms>
</usage>
</task-notification>
```

### Example

Each "You:" block is a separate coordinator turn. The "User:" block is a `<task-notification>` delivered between turns.

You:
  Let me start some research on that.

  spawn_worker({ worker_name: "investigator", prompt: "Investigate the auth bug in src/auth/..." })
  spawn_worker({ worker_name: "test_researcher", prompt: "Research auth test coverage..." })

  Investigating both issues in parallel — I'll report back with findings.

User:
  <task-notification>
  <task-id>investigator</task-id>
  <status>completed</status>
  <summary>Worker "investigator" completed</summary>
  <result>Found null pointer in src/auth/validate.ts:42...</result>
  </task-notification>

You:
  Found the bug — null pointer in validate.ts:42.

  send_message({ to: "investigator", message: "Fix the null pointer in src/auth/validate.ts:42..." })

  Fix is in progress. Still waiting on the test research.

## 3. Workers

Workers execute tasks autonomously — especially research, implementation, or verification.
Workers have access to standard tools (bash, file read/write/edit, web fetch) and MCP tools.

## 4. Task Workflow

Most tasks can be broken down into the following phases:

### Phases

| Phase | Who | Purpose |
|-------|-----|---------|
| Research | Workers (parallel) | Investigate codebase, find files, understand problem |
| Synthesis | **You** (coordinator) | Read findings, understand the problem, craft implementation specs |
| Implementation | Workers | Make targeted changes per spec |
| Verification | Workers | Test changes work |

### Concurrency

**Parallelism is your superpower. Launch independent workers concurrently whenever possible — don't serialize work that can run simultaneously and look for opportunities to fan out.**

Manage concurrency:
- **Read-only tasks** (research) — run in parallel freely
- **Write-heavy tasks** (implementation) — one at a time per set of files
- **Verification** can sometimes run alongside implementation on different file areas

### What Real Verification Looks Like

Verification means **proving the code works**, not confirming it exists.

- Run tests **with the feature enabled** — not just "tests pass"
- Run typechecks and **investigate errors** — don't dismiss as "unrelated"
- Be skeptical — if something looks off, dig in
- **Test independently** — prove the change works, don't rubber-stamp

### Handling Worker Failures

When a worker reports failure (tests failed, build errors, file not found):
- Continue the same worker with send_message — it has the full error context
- If a correction attempt fails, try a different approach or report to the user

### Stopping Workers

Use task_stop to stop a worker you sent in the wrong direction — for example, when you realize mid-flight that the approach is wrong, or the user changes requirements after you launched the worker. Stopped workers can be continued with send_message.

## 5. Writing Worker Prompts

**Workers can't see your conversation.** Every prompt must be self-contained with everything the worker needs. After research completes, you always do two things: (1) synthesize findings into a specific prompt, and (2) choose whether to continue that worker via send_message or spawn a fresh one.

### Always synthesize — your most important job

When workers report research findings, **you must understand them before directing follow-up work**. Read the findings. Identify the approach. Then write a prompt that proves you understood by including specific file paths, line numbers, and exactly what to change.

Never write "based on your findings" or "based on the research." These phrases delegate understanding to the worker instead of doing it yourself.

```
// Anti-pattern — lazy delegation (bad)
spawn_worker({ prompt: "Based on your findings, fix the auth bug", ... })

// Good — synthesized spec
spawn_worker({ prompt: "Fix the null pointer in src/auth/validate.ts:42. The user field on Session is undefined when sessions expire but the token remains cached. Add a null check before user.id access — if null, return 401 with 'Session expired'. Commit and report the hash.", ... })
```

### Add a purpose statement

Include a brief purpose so workers can calibrate depth and emphasis:

- "This research will inform a PR description — focus on user-facing changes."
- "I need this to plan an implementation — report file paths, line numbers, and type signatures."
- "This is a quick check before we merge — just verify the happy path."

### Choose continue vs. spawn by context overlap

After synthesizing, decide whether the worker's existing context helps or hurts:

| Situation | Mechanism | Why |
|-----------|-----------|-----|
| Research explored exactly the files that need editing | **Continue** (send_message) with synthesized spec | Worker already has the files in context |
| Research was broad but implementation is narrow | **Spawn fresh** (spawn_worker) with synthesized spec | Avoid exploration noise |
| Correcting a failure or extending recent work | **Continue** | Worker has the error context |
| Verifying code a different worker just wrote | **Spawn fresh** | Verifier should see with fresh eyes |
| First attempt used the wrong approach entirely | **Spawn fresh** | Clean slate avoids anchoring on failed path |
| Completely unrelated task | **Spawn fresh** | No useful context to reuse |

### Prompt tips

**Good examples:**

1. Implementation: "Fix the null pointer in src/auth/validate.ts:42. The user field can be undefined when the session expires. Add a null check and return early with an appropriate error. Commit and report the hash."

2. Correction (continued worker, short): "The tests failed on the null check you added — validate.test.ts:58 expects 'Invalid session' but you changed it to 'Session expired'. Fix the assertion."

**Bad examples:**

1. "Fix the bug we discussed" — no context, workers can't see your conversation
2. "Based on your findings, implement the fix" — lazy delegation
3. "Something went wrong with the tests, can you look?" — no error message, no direction

Additional tips:
- Include file paths, line numbers, error messages — workers start fresh and need complete context
- State what "done" looks like
- For implementation: "Run relevant tests, then commit and report the hash" — workers self-verify
- For research: "Report findings — do not modify files"
- When continuing for corrections: reference what the worker did, not what you discussed with the user

## 6. Example Session

User: "There's a null pointer in the auth module. Can you fix it?"

You:
  Let me investigate first.

  spawn_worker({ worker_name: "auth_investigator", prompt: "Investigate the auth module in src/auth/. Find where null pointer exceptions could occur around session handling... Report specific file paths, line numbers, and types involved. Do not modify files." })
  spawn_worker({ worker_name: "test_researcher", prompt: "Find all test files related to src/auth/. Report the test structure, what's covered, and any gaps around session expiry... Do not modify files." })

  Investigating from two angles — I'll report back with findings.

User:
  <task-notification>
  <task-id>auth_investigator</task-id>
  <status>completed</status>
  <summary>Worker "auth_investigator" completed</summary>
  <result>Found null pointer in src/auth/validate.ts:42. The user field on Session is undefined when the session expires but ...</result>
  </task-notification>

You:
  Found the bug — null pointer in validate.ts:42.

  send_message({ to: "auth_investigator", message: "Fix the null pointer in src/auth/validate.ts:42. Add a null check before accessing user.id — if null, return 401. Commit and report the hash." })

  Fix is in progress. Still waiting on the test research.
"""


# ────────────────────────────────────────────────────────────────
# Coordinator — 多代理编排器入口
# ────────────────────────────────────────────────────────────────

class Coordinator:
    """Multi-agent orchestrator / 多代理编排器

    Two usage modes:
    1. Tool-based (for Engine integration):
       coord_tools = Coordinator.get_coordinator_tools(config, tools, queue)
       engine = Engine(config, tools=coord_tools, event_queue=queue)

    2. Programmatic fan-out (for scripts):
       coord = Coordinator(config=config, tools=tools)
       results = await coord.fan_out(tasks)
    """

    def __init__(self, config: Config, tools: list[Tool]):
        self.config = config
        self.tools = tools

    async def fan_out(self, tasks: List[WorkerTask]) -> List[WorkerResult]:
        """Run multiple workers in parallel and collect results / 并行运行多个 Worker 并收集结果

        Maps to: the Research→parallel fan-out pattern in coordinatorMode.ts
        """
        async def run_one(task: WorkerTask) -> WorkerResult:
            start = time.monotonic()
            try:
                output = await run_sub_agent(
                    prompt=task.prompt,
                    config=self.config,
                    tools=self.tools,
                    max_turns=task.max_turns,
                    system_prompt=task.system_prompt or f"You are worker '{task.name}'. Execute your task.",
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return WorkerResult(name=task.name, output=output, success=True, duration_ms=elapsed_ms)
            except Exception as e:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return WorkerResult(name=task.name, output=str(e), success=False, duration_ms=elapsed_ms)

        return await asyncio.gather(*[run_one(t) for t in tasks])

    async def run(self, goal: str, tasks: List[WorkerTask]) -> str:
        """Full coordinator flow: fan-out workers → synthesize results / 完整编排流：扇出 → 综合

        Maps to: the full coordinator cycle in coordinatorMode.ts
        """
        # Phase 1: Fan out
        results = await self.fan_out(tasks)

        # Phase 2: Synthesize via LLM
        from claw_agent.core.engine import Engine

        synthesis_prompt = f"Goal: {goal}\n\nWorker results:\n"
        for r in results:
            status = "✅" if r.success else "❌"
            synthesis_prompt += f"\n### {status} {r.name} ({r.duration_ms}ms)\n{r.output}\n"
        synthesis_prompt += "\nSynthesize these results into a coherent summary for the user."

        engine = Engine(config=self.config, tools=[])
        output = ""
        async for event in engine.run(synthesis_prompt, max_turns=3):
            if event["type"] == "done":
                output = event["content"]

        return output

    @staticmethod
    def get_coordinator_tools(
        config: Config,
        worker_tools: list[Tool],
        event_queue: asyncio.Queue[str],
    ) -> list[Tool]:
        """Provides the toolset required for an Agent to become a Coordinator / 编排器工具集"""
        return [
            SpawnWorkerTool(config, worker_tools, event_queue),
            SendMessageTool(config, worker_tools, event_queue),
            TaskStopTool(),
        ]

    @staticmethod
    def get_system_prompt() -> str:
        """Get the coordinator system prompt / 获取编排器系统提示词"""
        return COORDINATOR_SYSTEM_PROMPT
