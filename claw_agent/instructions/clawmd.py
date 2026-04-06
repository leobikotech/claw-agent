"""
CLAW.md Config — 项目级指令文件系统

Discovers and loads CLAW.md instruction files from multiple sources,
assembled in priority order and injected into the system prompt.

File loading hierarchy (lowest → highest priority):
  1. User instructions  (~/.claw/CLAW.md)       — Private global instructions
  2. Project instructions (CLAW.md, .claw/CLAW.md, .claw/rules/*.md)
     — Checked into the codebase, discovered by traversing from CWD upward
  3. Local instructions  (CLAW.local.md)          — Private project-specific, not committed

@include directive:
  Memory files can reference other files using @ notation.
  Syntax: @path, @./relative/path, @~/home/path, @/absolute/path
  Included files are loaded as separate entries after the including file.
  Circular references are prevented. Non-existent files silently ignored.

Architecture note / 架构说明:
  This is the Claw Agent equivalent of CLAUDE.md. It provides project-level
  and user-level instruction injection — the most important configuration
  mechanism for customizing agent behavior per-project.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Types
# ────────────────────────────────────────────────────────────────

MemoryType = Literal["User", "Project", "Local"]

INSTRUCTION_PROMPT = (
    "Codebase and user instructions are shown below. "
    "Be sure to adhere to these instructions. "
    "IMPORTANT: These instructions OVERRIDE any default behavior "
    "and you MUST follow them exactly as written."
)

MAX_MEMORY_CHARACTER_COUNT = 40_000
MAX_INCLUDE_DEPTH = 5

# Text file extensions allowed for @include (prevents loading binary files)
TEXT_FILE_EXTENSIONS = frozenset({
    ".md", ".txt", ".text",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv",
    ".html", ".htm", ".css", ".scss", ".less",
    ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
    ".py", ".pyi",
    ".rb", ".erb",
    ".go", ".rs",
    ".java", ".kt", ".scala",
    ".c", ".cpp", ".cc", ".h", ".hpp",
    ".cs", ".swift",
    ".sh", ".bash", ".zsh", ".fish",
    ".env", ".ini", ".cfg", ".conf", ".config",
    ".sql", ".graphql",
    ".vue", ".svelte",
    ".php", ".lua", ".r", ".dart", ".ex",
    ".hs", ".ml", ".elm",
    ".rst", ".org", ".tex",
    ".lock", ".log", ".diff", ".patch",
    ".proto", ".cmake", ".makefile", ".gradle",
})


@dataclass
class InstructionFile:
    """A single discovered instruction file.

    Attributes:
        path:    Absolute path to the file.
        type:    User / Project / Local.
        content: File content (frontmatter stripped, HTML comments stripped).
        parent:  Path of the file that @included this one, if any.
        globs:   Glob patterns from frontmatter `paths:` field (for conditional rules).
    """
    path: str
    type: MemoryType
    content: str
    parent: Optional[str] = None
    globs: Optional[List[str]] = None


# ────────────────────────────────────────────────────────────────
# Frontmatter parsing (YAML-like header between --- fences)
# ────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n(.*?)\n---\s*\n?",
    re.DOTALL,
)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Strip optional YAML frontmatter and return (metadata_dict, body).

    Only extracts top-level `key: value` pairs and simple list items.
    Keeps the parser dependency-free (no pyyaml needed).
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw

    header = m.group(1)
    body = raw[m.end():]
    meta: dict = {}

    current_key: Optional[str] = None
    for line in header.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # List item under current_key
        if stripped.startswith("- ") and current_key:
            meta.setdefault(current_key, [])
            if isinstance(meta[current_key], list):
                meta[current_key].append(stripped[2:].strip())
            continue
        # Key: value
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            current_key = key
            if val:
                meta[key] = val
            else:
                meta[key] = []
        else:
            current_key = None

    return meta, body


def _parse_frontmatter_paths(raw: str) -> tuple[str, Optional[List[str]]]:
    """Extract content and optional glob patterns from frontmatter `paths:` field."""
    meta, body = _parse_frontmatter(raw)
    paths_val = meta.get("paths")
    if not paths_val:
        return body, None

    if isinstance(paths_val, str):
        patterns = [p.strip() for p in paths_val.split(",") if p.strip()]
    elif isinstance(paths_val, list):
        patterns = [p.strip() for p in paths_val if isinstance(p, str) and p.strip()]
    else:
        return body, None

    # Normalize: remove /** suffix, filter empty / match-all
    patterns = [
        p[:-3] if p.endswith("/**") else p
        for p in patterns
    ]
    patterns = [p for p in patterns if p and p != "**"]

    if not patterns:
        return body, None
    return body, patterns


# ────────────────────────────────────────────────────────────────
# HTML comment stripping
# ────────────────────────────────────────────────────────────────

_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")


def _strip_html_comments(content: str) -> str:
    """Remove block-level HTML comments from markdown content.

    Preserves comments inside fenced code blocks.
    """
    if "<!--" not in content:
        return content

    lines = content.splitlines(keepends=True)
    result: list[str] = []
    in_code_block = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track fenced code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            i += 1
            continue

        if in_code_block:
            result.append(line)
            i += 1
            continue

        # Replace HTML comments outside code blocks
        result.append(_HTML_COMMENT_RE.sub("", line))
        i += 1

    return "".join(result)


# ────────────────────────────────────────────────────────────────
# @include extraction
# ────────────────────────────────────────────────────────────────

_INCLUDE_RE = re.compile(r"(?:^|\s)@((?:[^\s\\]|\\ )+)")


def _extract_include_paths(content: str, base_dir: str) -> List[str]:
    """Extract @path references from content and resolve to absolute paths.

    Skips @paths inside fenced code blocks.
    """
    paths: list[str] = []
    in_code_block = False

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        for m in _INCLUDE_RE.finditer(line):
            raw_path = m.group(1)
            if not raw_path:
                continue

            # Strip fragment identifiers (#heading)
            hash_idx = raw_path.find("#")
            if hash_idx != -1:
                raw_path = raw_path[:hash_idx]
            if not raw_path:
                continue

            # Unescape spaces
            raw_path = raw_path.replace("\\ ", " ")

            # Resolve path
            if raw_path.startswith("~/"):
                resolved = os.path.expanduser(raw_path)
            elif raw_path.startswith("/"):
                resolved = raw_path
            elif raw_path.startswith("./"):
                resolved = os.path.join(base_dir, raw_path)
            else:
                # Bare name → relative
                if re.match(r"^[a-zA-Z0-9._-]", raw_path):
                    resolved = os.path.join(base_dir, raw_path)
                else:
                    continue

            resolved = os.path.normpath(resolved)
            paths.append(resolved)

    return paths


# ────────────────────────────────────────────────────────────────
# Core: read and parse a single instruction file
# ────────────────────────────────────────────────────────────────

def _read_instruction_file(
    file_path: str,
    mem_type: MemoryType,
) -> Optional[tuple[InstructionFile, List[str]]]:
    """Read and parse a single instruction file.

    Returns (InstructionFile, include_paths) or None if the file
    doesn't exist or is empty.
    """
    # Skip non-text files
    ext = os.path.splitext(file_path)[1].lower()
    if ext and ext not in TEXT_FILE_EXTENSIONS:
        logger.debug("Skipping non-text @include: %s", file_path)
        return None

    try:
        raw = Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if not raw.strip():
        return None

    # Parse frontmatter
    body, globs = _parse_frontmatter_paths(raw)

    # Strip HTML comments
    body = _strip_html_comments(body)

    if not body.strip():
        return None

    # Extract @includes
    include_paths = _extract_include_paths(body, os.path.dirname(file_path))

    info = InstructionFile(
        path=file_path,
        type=mem_type,
        content=body.strip(),
        globs=globs,
    )
    return info, include_paths


# ────────────────────────────────────────────────────────────────
# Recursive file processor (handles @includes)
# ────────────────────────────────────────────────────────────────

def _process_file(
    file_path: str,
    mem_type: MemoryType,
    processed: set[str],
    depth: int = 0,
    parent: Optional[str] = None,
) -> List[InstructionFile]:
    """Recursively process an instruction file and its @includes.

    Returns files in order: [main_file, ...included_files].
    """
    normalized = os.path.normpath(file_path)
    if normalized in processed or depth >= MAX_INCLUDE_DEPTH:
        return []

    processed.add(normalized)

    result = _read_instruction_file(file_path, mem_type)
    if result is None:
        return []

    info, include_paths = result
    if parent:
        info.parent = parent

    files: list[InstructionFile] = [info]

    # Process @includes
    for inc_path in include_paths:
        children = _process_file(
            inc_path, mem_type, processed,
            depth=depth + 1, parent=file_path,
        )
        files.extend(children)

    return files


# ────────────────────────────────────────────────────────────────
# Process .claw/rules/ directory
# ────────────────────────────────────────────────────────────────

def _process_rules_dir(
    rules_dir: str,
    mem_type: MemoryType,
    processed: set[str],
) -> List[InstructionFile]:
    """Process all .md files in a .claw/rules/ directory (recursive)."""
    if not os.path.isdir(rules_dir):
        return []

    files: list[InstructionFile] = []
    try:
        for entry in sorted(os.listdir(rules_dir)):
            entry_path = os.path.join(rules_dir, entry)
            if os.path.isdir(entry_path):
                files.extend(_process_rules_dir(entry_path, mem_type, processed))
            elif entry.endswith(".md") and os.path.isfile(entry_path):
                files.extend(_process_file(entry_path, mem_type, processed))
    except OSError:
        pass

    return files


# ────────────────────────────────────────────────────────────────
# Public API: discover all instruction files
# ────────────────────────────────────────────────────────────────

def get_user_claw_dir() -> str:
    """Return the user-level config directory (~/.claw/)."""
    return os.path.join(os.path.expanduser("~"), ".claw")


def discover_instruction_files(cwd: Optional[str] = None) -> List[InstructionFile]:
    """Discover all CLAW.md instruction files from all sources.

    Loading order (lowest → highest priority):
      1. User:    ~/.claw/CLAW.md, ~/.claw/rules/*.md
      2. Project: CLAW.md, .claw/CLAW.md, .claw/rules/*.md
                  (discovered by walking from CWD upward to filesystem root)
      3. Local:   CLAW.local.md (per-directory, not committed)

    Files closer to CWD have higher priority (loaded later).

    Args:
        cwd: Working directory to start discovery from. Defaults to os.getcwd().

    Returns:
        List of InstructionFile objects in loading order.
    """
    cwd = cwd or os.getcwd()
    processed: set[str] = set()
    result: list[InstructionFile] = []

    # ── 1. User-level instructions ──
    user_dir = get_user_claw_dir()

    # ~/.claw/CLAW.md
    user_claw_md = os.path.join(user_dir, "CLAW.md")
    result.extend(_process_file(user_claw_md, "User", processed))

    # ~/.claw/rules/*.md
    user_rules = os.path.join(user_dir, "rules")
    result.extend(_process_rules_dir(user_rules, "User", processed))

    # ── 2. Project-level instructions (walk CWD → root) ──
    dirs: list[str] = []
    current = os.path.abspath(cwd)
    root = os.path.abspath(os.sep)

    while current != root:
        dirs.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Process from root downward to CWD (higher priority = loaded later)
    for d in reversed(dirs):
        # CLAW.md in directory root
        project_path = os.path.join(d, "CLAW.md")
        result.extend(_process_file(project_path, "Project", processed))

        # .claw/CLAW.md
        dot_claw_path = os.path.join(d, ".claw", "CLAW.md")
        result.extend(_process_file(dot_claw_path, "Project", processed))

        # .claw/rules/*.md
        rules_dir = os.path.join(d, ".claw", "rules")
        result.extend(_process_rules_dir(rules_dir, "Project", processed))

        # CLAW.local.md (local, not committed)
        local_path = os.path.join(d, "CLAW.local.md")
        result.extend(_process_file(local_path, "Local", processed))

    count_by_type = {}
    for f in result:
        count_by_type[f.type] = count_by_type.get(f.type, 0) + 1
    if result:
        logger.info(
            "Discovered %d instruction file(s): %s",
            len(result),
            ", ".join(f"{t}={n}" for t, n in count_by_type.items()),
        )

    return result


# ────────────────────────────────────────────────────────────────
# Public API: format into prompt text
# ────────────────────────────────────────────────────────────────

_TYPE_DESCRIPTIONS = {
    "Project": "(project instructions, checked into the codebase)",
    "Local": "(user's private project instructions, not checked in)",
    "User": "(user's private global instructions for all projects)",
}


def format_instructions_prompt(files: List[InstructionFile]) -> str:
    """Format discovered instruction files into a prompt string.

    """
    if not files:
        return ""

    blocks: list[str] = []
    for f in files:
        desc = _TYPE_DESCRIPTIONS.get(f.type, "")
        content = f.content.strip()
        if content:
            blocks.append(f"Contents of {f.path} {desc}:\n\n{content}")

    if not blocks:
        return ""

    return f"{INSTRUCTION_PROMPT}\n\n" + "\n\n".join(blocks)


def get_large_files(files: List[InstructionFile]) -> List[InstructionFile]:
    """Return instruction files exceeding the recommended character limit."""
    return [f for f in files if len(f.content) > MAX_MEMORY_CHARACTER_COUNT]
