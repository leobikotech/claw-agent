"""
File Tools — 文件操作工具集

Read, write, and edit files with full prompt translations matching the original.
"""
from __future__ import annotations
import os
from typing import Any

from claw_agent.core.tool import Tool, ToolContext, RiskLevel

# Constants matching original prompt.ts
FILE_READ_TOOL_NAME = "file_read"
FILE_EDIT_TOOL_NAME = "file_edit"
FILE_WRITE_TOOL_NAME = "file_write"
MAX_LINES_TO_READ = 2000
FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier Read "
    "tool_result in this conversation is still current — refer to that "
    "instead of re-reading."
)


# ────────────────────────────────────────────────────────────────
# FileReadTool — 1:1 translation of FileReadTool/prompt.ts
# ────────────────────────────────────────────────────────────────

def _get_file_read_prompt() -> str:
    """Full translation of renderPromptTemplate() from FileReadTool/prompt.ts."""
    return f"""Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to {MAX_LINES_TO_READ} lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows reading images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as a multimodal LLM.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- This tool can only read files, not directories. To read a directory, use an ls command via the bash tool.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""


class FileReadTool(Tool):
    name = FILE_READ_TOOL_NAME
    description = _get_file_read_prompt()
    risk_level = RiskLevel.LOW
    is_read_only = True

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start reading from (1-indexed)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to read (default: {MAX_LINES_TO_READ})",
                    "default": MAX_LINES_TO_READ,
                },
            },
            "required": ["file_path"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        file_path = arguments["file_path"]
        # Support both absolute and relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)

        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", MAX_LINES_TO_READ)

        if not os.path.exists(file_path):
            return f"Error: file not found: {file_path}"

        if os.path.isdir(file_path):
            return f"Error: {file_path} is a directory, not a file. Use 'ls' via bash tool to list directory contents."

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)

            if total == 0:
                return "(empty file — this file exists but has no content)"

            # Apply offset and limit
            start = max(0, offset)
            selected = lines[start : start + limit]

            # Format with line numbers (cat -n style, 1-indexed)
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"  {i}\t{line.rstrip()}")

            content = "\n".join(numbered)

            if total > start + limit:
                content += f"\n\n... ({total - start - limit} more lines not shown)"

            return content
        except Exception as e:
            return f"Error reading file: {e}"


# ────────────────────────────────────────────────────────────────
# FileEditTool — 1:1 translation of FileEditTool/prompt.ts
# ────────────────────────────────────────────────────────────────

def _get_file_edit_prompt() -> str:
    """Full translation of getEditToolDescription() from FileEditTool/prompt.ts."""
    return f"""Performs exact string replacements in files.

Usage:
- You must use your `{FILE_READ_TOOL_NAME}` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""


class FileEditTool(Tool):
    name = FILE_EDIT_TOOL_NAME
    description = _get_file_edit_prompt()
    risk_level = RiskLevel.MEDIUM
    is_read_only = False

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find and replace (must be unique in the file unless replace_all is set)",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace ALL occurrences of old_string",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        file_path = arguments["file_path"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)
        old_text = arguments["old_string"]
        new_text = arguments["new_string"]
        replace_all = arguments.get("replace_all", False)

        if not os.path.exists(file_path):
            return f"Error: file not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_text)
            if count == 0:
                return "Error: old_string not found in file. Make sure the text matches exactly, including whitespace and line endings."

            if count > 1 and not replace_all:
                return (
                    f"Error: old_string found {count} times — must be unique. "
                    f"Provide more surrounding context to make it unique, or set replace_all=true."
                )

            if replace_all:
                new_content = content.replace(old_text, new_text)
            else:
                new_content = content.replace(old_text, new_text, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            replacements = count if replace_all else 1
            return f"Successfully edited {file_path} ({replacements} replacement{'s' if replacements > 1 else ''})"
        except Exception as e:
            return f"Error editing file: {e}"


# ────────────────────────────────────────────────────────────────
# FileWriteTool — 1:1 translation of FileWriteTool/prompt.ts
# ────────────────────────────────────────────────────────────────

def _get_file_write_prompt() -> str:
    """Full translation of getWriteToolDescription() from FileWriteTool/prompt.ts."""
    return f"""Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the {FILE_READ_TOOL_NAME} tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the {FILE_EDIT_TOOL_NAME} tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""


class FileWriteTool(Tool):
    name = FILE_WRITE_TOOL_NAME
    description = _get_file_write_prompt()
    risk_level = RiskLevel.MEDIUM
    is_read_only = False

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["file_path", "content"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        file_path = arguments["file_path"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)
        content = arguments["content"]

        try:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} chars to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"
