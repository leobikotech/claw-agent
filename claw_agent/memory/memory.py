"""
Memory — 文件级持久化记忆
Maps to: src/memdir/memdir.ts, memoryScan.ts, findRelevantMemories.ts

A file-based memory system with:
  - MEMORY.md index + topic files
  - Memory scanning with frontmatter extraction
  - LLM-based relevant memory selection (findRelevantMemories)
  - Build prompt for system prompt injection
"""
from __future__ import annotations
import glob as _glob
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

MEMORY_FILE = "MEMORY.md"          # Index file
MAX_INDEX_LINES = 200              # Maps to MAX_ENTRYPOINT_LINES
MAX_INDEX_BYTES = 25_000           # Maps to MAX_ENTRYPOINT_BYTES
MAX_MEMORY_FILES = 200             # Maps to MAX_MEMORY_FILES in memoryScan.ts
FRONTMATTER_MAX_LINES = 30
MAX_RELEVANT_MEMORIES = 5          # From findRelevantMemories.ts


# ────────────────────────────────────────────────────────────────
# Memory header — maps to MemoryHeader in memoryScan.ts
# ────────────────────────────────────────────────────────────────

@dataclass
class MemoryHeader:
    """Scanned memory file metadata."""
    filename: str       # Relative path within memory_dir
    file_path: str      # Absolute path
    mtime_ms: float     # Modification timestamp (ms)
    description: Optional[str] = None


@dataclass
class MemoryEntry:
    """A single memory entry."""
    title: str
    file: str       # Topic file path relative to memory_dir
    summary: str    # One-line hook for the index


# ────────────────────────────────────────────────────────────────
# Frontmatter parser — maps to utils/frontmatterParser.ts
# ────────────────────────────────────────────────────────────────

def _parse_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML frontmatter from a markdown file."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}

    fm = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm


# ────────────────────────────────────────────────────────────────
# Memory scanning — maps to memoryScan.ts scanMemoryFiles()
# ────────────────────────────────────────────────────────────────

def scan_memory_files(memory_dir: str) -> list[MemoryHeader]:
    """Scan memory directory for .md files, read frontmatter, return headers.
    Maps to: scanMemoryFiles() in memoryScan.ts

    Sorted newest-first, capped at MAX_MEMORY_FILES.
    """
    headers = []
    try:
        for md_path in _glob.glob(os.path.join(memory_dir, "**/*.md"), recursive=True):
            basename = os.path.basename(md_path)
            if basename == MEMORY_FILE:
                continue

            try:
                stat = os.stat(md_path)
                # Read first N lines for frontmatter
                with open(md_path, "r", encoding="utf-8", errors="replace") as f:
                    head = "".join(f.readline() for _ in range(FRONTMATTER_MAX_LINES))

                fm = _parse_frontmatter(head)
                rel_path = os.path.relpath(md_path, memory_dir)

                headers.append(MemoryHeader(
                    filename=rel_path,
                    file_path=md_path,
                    mtime_ms=stat.st_mtime * 1000,
                    description=fm.get("description"),
                ))
            except (OSError, UnicodeDecodeError):
                continue

        # Sort newest-first, cap at limit
        headers.sort(key=lambda h: h.mtime_ms, reverse=True)
        return headers[:MAX_MEMORY_FILES]
    except Exception:
        return []


def format_memory_manifest(memories: list[MemoryHeader]) -> str:
    """Format memory headers as text manifest for LLM selection.
    Maps to: formatMemoryManifest() in memoryScan.ts
    """
    lines = []
    for m in memories:
        ts = datetime.fromtimestamp(m.mtime_ms / 1000).isoformat()
        if m.description:
            lines.append(f"- {m.filename} ({ts}): {m.description}")
        else:
            lines.append(f"- {m.filename} ({ts})")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# Relevant memory selection — maps to findRelevantMemories.ts
# ────────────────────────────────────────────────────────────────

# The system prompt for the memory selection side-query
SELECT_MEMORIES_SYSTEM_PROMPT = """You are selecting memories that will be useful to an AI coding agent as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected_memories" key containing a list of filenames for the memories that will clearly be useful (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, return an empty list.

Your response MUST be valid JSON, e.g.: {"selected_memories": ["file1.md", "file2.md"]}"""


async def find_relevant_memories(
    query: str,
    memory_dir: str,
    provider: Any,  # LLMProvider
    model: str = "",
    already_surfaced: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Find memory files relevant to a query using LLM selection.
    Maps to: findRelevantMemories() in findRelevantMemories.ts

    Uses a side LLM call to select the most relevant memories from the
    manifest. Returns list of {"path": ..., "mtime_ms": ...} dicts.
    """
    memories = scan_memory_files(memory_dir)
    if already_surfaced:
        memories = [m for m in memories if m.file_path not in already_surfaced]

    if not memories:
        return []

    manifest = format_memory_manifest(memories)

    try:
        import json
        response = await provider.chat(
            messages=[
                {"role": "system", "content": SELECT_MEMORIES_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nAvailable memories:\n{manifest}"},
            ],
            tools=None,
            model=model,
            max_tokens=256,
            temperature=0.0,
        )

        if not response.content:
            return []

        # Parse JSON response
        text = response.content.strip()
        # Handle markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        parsed = json.loads(text)
        selected_filenames = parsed.get("selected_memories", [])

        # Map back to full paths
        by_filename = {m.filename: m for m in memories}
        results = []
        for fn in selected_filenames[:MAX_RELEVANT_MEMORIES]:
            m = by_filename.get(fn)
            if m:
                results.append({"path": m.file_path, "mtime_ms": m.mtime_ms})

        return results
    except Exception as e:
        logger.warning(f"findRelevantMemories failed: {e}")
        return []


# ────────────────────────────────────────────────────────────────
# Memory class — main interface
# ────────────────────────────────────────────────────────────────

class Memory:
    """File-based persistent memory / 基于文件的持久化记忆
    Maps to: memdir.ts buildMemoryPrompt, loadMemoryPrompt

    Structure:
        memory_dir/
        ├── MEMORY.md          ← Index (one-line pointers)
        ├── user_preferences.md
        ├── project_context.md
        └── ...
    """

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

    @property
    def index_path(self) -> str:
        return os.path.join(self.memory_dir, MEMORY_FILE)

    def load_index(self) -> str:
        """Load MEMORY.md index with truncation.
        Maps to: truncateEntrypointContent() in memdir.ts
        """
        if not os.path.exists(self.index_path):
            return ""
        with open(self.index_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")
        if len(lines) > MAX_INDEX_LINES:
            content = "\n".join(lines[:MAX_INDEX_LINES])
            content += f"\n\n> WARNING: MEMORY.md has {len(lines)} lines (limit: {MAX_INDEX_LINES}). Truncated."

        if len(content.encode()) > MAX_INDEX_BYTES:
            content = content[:MAX_INDEX_BYTES]
            content += "\n\n> WARNING: MEMORY.md exceeds byte limit. Truncated."

        return content

    def save(self, title: str, content: str, filename: Optional[str] = None) -> str:
        """Save a memory (topic file + index entry).
        Maps to: two-step save in buildMemoryLines (write file → update index)
        """
        # Generate filename from title if not provided
        if not filename:
            filename = title.lower().replace(" ", "_").replace("/", "_") + ".md"

        file_path = os.path.join(self.memory_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}\n")

        # Update index — append one-line pointer
        summary = content.split("\n")[0][:100]
        entry = f"- [{title}]({filename}) — {summary}\n"
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(entry)

        return file_path

    def search(self, query: str) -> list[tuple[str, str]]:
        """Search all memory files for a query.
        Maps to: GrepTool on memory_dir in buildSearchingPastContextSection
        """
        results = []
        query_lower = query.lower()
        for md_path in _glob.glob(os.path.join(self.memory_dir, "**/*.md"), recursive=True):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if query_lower in content.lower():
                    rel = os.path.relpath(md_path, self.memory_dir)
                    for line in content.split("\n"):
                        if query_lower in line.lower():
                            results.append((rel, line.strip()))
                            break
            except Exception:
                continue
        return results

    def forget(self, filename: str) -> bool:
        """Remove a memory file and its index entry."""
        file_path = os.path.join(self.memory_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            if os.path.exists(self.index_path):
                with open(self.index_path, "r") as f:
                    lines = f.readlines()
                with open(self.index_path, "w") as f:
                    for line in lines:
                        if filename not in line:
                            f.write(line)
            return True
        return False

    def scan(self) -> list[MemoryHeader]:
        """Scan all memory files and return headers.
        Maps to: scanMemoryFiles() in memoryScan.ts
        """
        return scan_memory_files(self.memory_dir)

    async def find_relevant(
        self,
        query: str,
        provider: Any,
        model: str = "",
        already_surfaced: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        """Find memories relevant to a query using LLM selection.
        Maps to: findRelevantMemories() in findRelevantMemories.ts
        """
        return await find_relevant_memories(
            query=query,
            memory_dir=self.memory_dir,
            provider=provider,
            model=model,
            already_surfaced=already_surfaced,
        )

    def load_memory_content(self, filepath: str) -> str:
        """Load full content of a memory file."""
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.memory_dir, filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error loading memory: {e}"

    def build_prompt(self) -> str:
        """Build memory prompt for system prompt injection.
        Maps to: buildMemoryPrompt() in memdir.ts
        """
        index = self.load_index()
        return f"""# Persistent Memory
You have file-based memory at: `{self.memory_dir}`

## MEMORY.md
{index if index else "(empty — no memories saved yet)"}

## How to use
- To save: write a topic file, then add one-line pointer to MEMORY.md
- To search: grep memory files for keywords
- To forget: delete the topic file and its index entry
"""
