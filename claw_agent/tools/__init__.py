"""
Tools — 工具集入口
Maps to: src/tools.ts getAllBaseTools()

Registry of all built-in tools.
"""
from claw_agent.tools.bash_tool import BashTool
from claw_agent.tools.file_tools import FileReadTool, FileWriteTool, FileEditTool
from claw_agent.tools.glob_tool import GlobTool
from claw_agent.tools.grep_tool import GrepTool
from claw_agent.tools.web_fetch_tool import WebFetchTool


def get_default_tools():
    """Get all built-in tools / 获取所有内置工具
    Maps to: getAllBaseTools() in src/tools.ts
    """
    return [
        BashTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GlobTool(),
        GrepTool(),
        WebFetchTool(),
    ]
