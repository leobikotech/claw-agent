"""
GlobTool — 文件模式搜索工具
Maps to: src/tools/GlobTool/GlobTool.ts + prompt.ts

Fast file pattern matching using glob patterns.
"""
from __future__ import annotations
import glob as _glob
import os
from typing import Any

from claw_agent.core.tool import Tool, ToolContext, RiskLevel

GLOB_TOOL_NAME = "glob"

# Match original: max 100 results
MAX_GLOB_RESULTS = 100


def _get_glob_prompt() -> str:
    """1:1 translation of GlobTool/prompt.ts DESCRIPTION."""
    return """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the agent tool instead"""


class GlobTool(Tool):
    """Maps to: src/tools/GlobTool/GlobTool.ts"""
    name = GLOB_TOOL_NAME
    description = _get_glob_prompt()
    risk_level = RiskLevel.LOW
    is_read_only = True

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": 'Glob pattern to match files (e.g. "**/*.py", "src/**/*.ts")',
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (defaults to cwd). Must be an absolute path.",
                },
            },
            "required": ["pattern"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        pattern = arguments["pattern"]
        search_dir = arguments.get("path", context.cwd)
        if not os.path.isabs(search_dir):
            search_dir = os.path.join(context.cwd, search_dir)

        if not os.path.isdir(search_dir):
            return f"Error: directory not found: {search_dir}"

        try:
            full_pattern = os.path.join(search_dir, pattern)
            matches = _glob.glob(full_pattern, recursive=True)

            # Filter out directories — only return files
            files = [f for f in matches if os.path.isfile(f)]

            # Sort by modification time (newest first), matching original behavior
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            if not files:
                return f"No files found matching pattern: {pattern}"

            total = len(files)
            truncated = files[:MAX_GLOB_RESULTS]

            # Return relative paths for readability
            result_lines = []
            for f in truncated:
                try:
                    rel = os.path.relpath(f, search_dir)
                except ValueError:
                    rel = f
                result_lines.append(rel)

            output = "\n".join(result_lines)
            if total > MAX_GLOB_RESULTS:
                output += f"\n\n... ({total - MAX_GLOB_RESULTS} more files not shown, {total} total matches)"

            return output
        except Exception as e:
            return f"Error running glob: {e}"
