"""
Coordinator Example — 多代理编排示例

Two usage modes demonstrated:
1. Programmatic fan-out: Deploy parallel workers and collect results
2. Engine-integrated: Use coordinator tools within an Engine loop

将研究、实现、验证三个任务并行扇出给不同的 Worker。
"""
import asyncio
from claw_agent import Config
from claw_agent.tools import get_default_tools
from claw_agent.agents.coordinator import Coordinator, WorkerTask


async def main():
    config = Config()
    tools = get_default_tools()

    coordinator = Coordinator(config=config, tools=tools)

    # Define worker tasks / 定义 Worker 任务
    tasks = [
        WorkerTask(
            name="researcher",
            prompt="Research the best practices for Python project structure in 2025. "
                   "List the key directories and files needed. "
                   "Do not modify any files — research only.",
            max_turns=10,
        ),
        WorkerTask(
            name="implementer",
            prompt="Create a basic pyproject.toml for a Python package called 'my_app' "
                   "with FastAPI and SQLAlchemy as dependencies.",
            max_turns=10,
        ),
        WorkerTask(
            name="verifier",
            prompt="List common mistakes in Python project setup and how to avoid them. "
                   "Do not modify any files — research only.",
            max_turns=5,
        ),
    ]

    print("🧠 Coordinator: dispatching 3 workers in parallel...\n")

    # Fan out and collect results
    results = await coordinator.fan_out(tasks)

    for r in results:
        status = "✅" if r.success else "❌"
        print(f"{status} Worker [{r.name}] ({r.duration_ms}ms):")
        print(f"   {r.output[:200]}...")
        print()


async def engine_mode():
    """Alternative: Use coordinator tools within an Engine loop / 引擎集成模式

    The coordinator's system prompt teaches the LLM to:
    - spawn_worker for parallel research/implementation
    - send_message to continue workers with follow-up instructions
    - task_stop to cancel misdirected workers
    """
    from claw_agent.core.engine import Engine

    config = Config()
    event_queue = asyncio.Queue()

    # Standard tools for workers
    worker_tools = get_default_tools()

    # Coordinator tools for the Engine
    coord_tools = Coordinator.get_coordinator_tools(config, worker_tools, event_queue)

    # Use the coordinator system prompt so the LLM knows how to orchestrate
    config.system_prompt = Coordinator.get_system_prompt()

    engine = Engine(config=config, tools=coord_tools, event_queue=event_queue)

    async for event in engine.run("Investigate the auth module and fix any null pointer bugs"):
        if event["type"] == "done":
            print(event["content"])
        elif event["type"] == "worker_notification":
            print(f"📩 Worker notification received")


if __name__ == "__main__":
    asyncio.run(main())
