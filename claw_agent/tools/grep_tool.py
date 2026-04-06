"""
GrepTool — 内容搜索工具

Powerful search tool using regex, with output modes and file filtering.
Tries rg (ripgrep), then falls back to grep.
"""
from __future__ import annotations
import asyncio
import os
import shutil
from typing import Any

from claw_agent.core.tool import Tool, ToolContext, RiskLevel

GREP_TOOL_NAME = "grep"

# Match original: max 100 file results
MAX_RESULTS = 100


def _get_grep_prompt() -> str:
    """1:1 translation of GrepTool/prompt.ts getDescription()."""
    return """A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use grep for search tasks. NEVER invoke `grep` or `rg` as a bash command. The grep tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
  - Filter files with include parameter (e.g., "*.js", "**/*.tsx")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use agent tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) — literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`
"""


def _find_search_cmd() -> str:
    """Find the best available search command."""
    if shutil.which("rg"):
        return "rg"
    if shutil.which("grep"):
        return "grep"
    return ""


class GrepTool(Tool):
    name = GREP_TOOL_NAME
    description = _get_grep_prompt()
    risk_level = RiskLevel.LOW
    is_read_only = True

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (defaults to cwd). Must be an absolute path.",
                },
                "include": {
                    "type": "string",
                    "description": 'Glob pattern to filter files (e.g., "*.py", "*.ts")',
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output format: content (matching lines), files_with_matches (file paths only), count (match counts)",
                    "default": "content",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: true)",
                    "default": True,
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline matching for cross-line patterns",
                    "default": False,
                },
            },
            "required": ["pattern"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        pattern = arguments["pattern"]
        search_path = arguments.get("path", context.cwd)
        if not os.path.isabs(search_path):
            search_path = os.path.join(context.cwd, search_path)
        include = arguments.get("include", "")
        output_mode = arguments.get("output_mode", "content")
        case_sensitive = arguments.get("case_sensitive", True)
        multiline = arguments.get("multiline", False)

        if not os.path.exists(search_path):
            return f"Error: path not found: {search_path}"

        search_cmd = _find_search_cmd()
        if not search_cmd:
            # Fallback: Python-based search
            return await self._python_search(pattern, search_path, include, output_mode, case_sensitive)

        try:
            if search_cmd == "rg":
                return await self._rg_search(
                    pattern, search_path, include, output_mode, case_sensitive, multiline
                )
            else:
                return await self._grep_search(
                    pattern, search_path, include, output_mode, case_sensitive
                )
        except Exception as e:
            return f"Error running search: {e}"

    async def _rg_search(
        self, pattern: str, path: str, include: str,
        output_mode: str, case_sensitive: bool, multiline: bool,
    ) -> str:
        """Search using ripgrep (rg) — preferred."""
        cmd = ["rg", "--no-heading", "--line-number", "--color=never"]

        if not case_sensitive:
            cmd.append("-i")
        if multiline:
            cmd.append("--multiline")

        # Output mode
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")

        # Include filter
        if include:
            cmd.extend(["-g", include])

        # Max results
        cmd.extend(["--max-count", str(MAX_RESULTS)])

        cmd.append(pattern)
        cmd.append(path)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace").strip()

        if not output:
            return f"No matches found for pattern: {pattern}"

        # Truncate if too many lines
        lines = output.split("\n")
        if len(lines) > MAX_RESULTS:
            output = "\n".join(lines[:MAX_RESULTS])
            output += f"\n\n... ({len(lines) - MAX_RESULTS} more results not shown)"

        return output

    async def _grep_search(
        self, pattern: str, path: str, include: str,
        output_mode: str, case_sensitive: bool,
    ) -> str:
        """Fallback search using GNU grep."""
        cmd = ["grep", "-rn", "--color=never"]

        if not case_sensitive:
            cmd.append("-i")
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        if include:
            cmd.extend(["--include", include])

        cmd.append(pattern)
        cmd.append(path)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace").strip()

        if not output:
            return f"No matches found for pattern: {pattern}"

        lines = output.split("\n")
        if len(lines) > MAX_RESULTS:
            output = "\n".join(lines[:MAX_RESULTS])
            output += f"\n\n... ({len(lines) - MAX_RESULTS} more results not shown)"

        return output

    async def _python_search(
        self, pattern: str, path: str, include: str,
        output_mode: str, case_sensitive: bool,
    ) -> str:
        """Pure Python fallback when no search binary is available."""
        import re
        import glob as _glob

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Collect files
        if os.path.isfile(path):
            files = [path]
        else:
            glob_pattern = include if include else "**/*"
            files = [
                f for f in _glob.glob(os.path.join(path, glob_pattern), recursive=True)
                if os.path.isfile(f)
            ]

        results = []
        match_count = 0

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            if output_mode == "files_with_matches":
                                results.append(filepath)
                                break
                            elif output_mode == "count":
                                match_count += 1
                            else:
                                results.append(f"{filepath}:{line_no}:{line.rstrip()}")
                            if len(results) >= MAX_RESULTS:
                                break
            except (OSError, UnicodeDecodeError):
                continue

            if len(results) >= MAX_RESULTS:
                break

        if output_mode == "count":
            return f"Match count: {match_count}"

        if not results:
            return f"No matches found for pattern: {pattern}"

        output = "\n".join(results)
        return output
