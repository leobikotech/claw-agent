"""
Custom Tool Example — 自定义工具示例
Shows how to create tools using the @tool decorator and Tool subclass.
展示如何用 @tool 装饰器和 Tool 子类创建自定义工具。
"""
import asyncio
from claw_agent import Engine, Config, Tool, tool, ToolRegistry
from claw_agent.core.tool import ToolContext, RiskLevel


# --- Method 1: @tool decorator (simplest) ---
registry = ToolRegistry()

@tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
    registry=registry,
)
async def get_weather(args: dict, ctx: ToolContext) -> str:
    city = args["city"]
    # In real use, call a weather API here
    return f"Weather in {city}: 22°C, sunny ☀️"


# --- Method 2: Tool subclass (full control) ---
class DatabaseQueryTool(Tool):
    name = "query_db"
    description = "Execute a SQL query against the application database"
    risk_level = RiskLevel.MEDIUM
    is_read_only = True

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to execute"},
            },
            "required": ["sql"],
        }

    async def call(self, arguments: dict, context: ToolContext) -> str:
        sql = arguments["sql"]
        # In real use, execute against a real database
        return f"Query result for: {sql}\n| id | name |\n| 1  | Alice |"


# Register the subclass tool
registry.register(DatabaseQueryTool())


async def main():
    config = Config()
    engine = Engine(config=config, registry=registry)

    async for event in engine.run("What's the weather in Shanghai? Also query the users table."):
        if event["type"] == "tool_call":
            print(f"  ⚡ {event['name']}")
        elif event["type"] == "tool_result":
            print(f"  → {event['content'][:100]}")
        elif event["type"] == "done":
            print(f"\n{event['content']}")


if __name__ == "__main__":
    asyncio.run(main())
