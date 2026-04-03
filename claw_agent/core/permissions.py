"""
Permissions — 权限门控系统
Maps to: src/Tool.ts (ToolPermissionContext), tools/BashTool/bashPermissions.ts, bashSecurity.ts

Three-tier permission gating: allow, ask (interactive confirm), deny.
Comprehensive dangerous command detection and protected path enforcement.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import os
import re
from typing import Any, Optional

from claw_agent.core.tool import Tool, RiskLevel, PermissionCheck


class PermissionMode(str, Enum):
    """Maps to: PermissionMode in types/permissions.ts"""
    DEFAULT = "default"  # Interactive confirmation
    AUTO = "auto"        # Auto-approve non-destructive
    BYPASS = "bypass"    # Allow everything (dangerous — for trusted automation only)


# ────────────────────────────────────────────────────────────────
# Dangerous command patterns — comprehensive (maps to bashSecurity.ts)
# ────────────────────────────────────────────────────────────────

# Category 1: System-destructive commands
SYSTEM_DESTRUCTIVE = {
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # fork bomb
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "halt",
    "> /dev/sda",
    "format c:",
    "del /f /s /q c:",
}

# Category 2: Git destructive operations
GIT_DESTRUCTIVE_PATTERNS = [
    r"git\s+push\s+.*--force",
    r"git\s+push\s+-f\b",
    r"git\s+reset\s+--hard",
    r"git\s+checkout\s+\.",
    r"git\s+restore\s+\.",
    r"git\s+clean\s+-f",
    r"git\s+branch\s+-D",
    r"git\s+.*--no-verify",
    r"git\s+.*--no-gpg-sign",
    r"git\s+config\b",
]

# Category 3: Package management — dangerous operations
PACKAGE_DANGEROUS = [
    r"pip\s+install\s+--break-system-packages",
    r"npm\s+publish",
    r"npm\s+unpublish",
    r"yarn\s+publish",
]

# Category 4: Container/infra destructive
INFRA_DANGEROUS = [
    r"docker\s+rm\s+-f",
    r"docker\s+system\s+prune",
    r"kubectl\s+delete",
    r"terraform\s+destroy",
]

# Category 5: Credential/config exfiltration
EXFIL_PATTERNS = [
    r"curl\s+.*\|\s*bash",
    r"wget\s+.*\|\s*bash",
    r"curl\s+.*\|\s*sh",
    r"wget\s+.*\|\s*sh",
]

# Compiled regex for git/package/infra/exfil patterns
_DANGEROUS_REGEXES = [
    re.compile(p, re.IGNORECASE)
    for p in (GIT_DESTRUCTIVE_PATTERNS + PACKAGE_DANGEROUS + INFRA_DANGEROUS + EXFIL_PATTERNS)
]


# ────────────────────────────────────────────────────────────────
# Protected files — files that should never be written to
# Maps to: bashPermissions.ts protected path logic
# ────────────────────────────────────────────────────────────────

PROTECTED_PATHS = {
    ".bashrc", ".zshrc", ".bash_profile", ".profile",
    ".gitconfig", ".ssh/", ".gnupg/",
    ".env", ".env.local", ".env.production",
    ".mcp.json",
    "credentials.json", "service-account.json",
}


# ────────────────────────────────────────────────────────────────
# Permission context + checking
# ────────────────────────────────────────────────────────────────

@dataclass
class PermissionContext:
    """Runtime permission state / 运行时权限状态
    Maps to: ToolPermissionContext in Tool.ts
    """
    mode: PermissionMode = PermissionMode.DEFAULT
    allow_rules: dict[str, list[str]] = field(default_factory=dict)
    deny_rules: dict[str, list[str]] = field(default_factory=dict)
    # Tracks consecutive denials (maps to denialTracking.ts)
    denial_count: int = 0


def check_permission(
    tool: Tool,
    arguments: dict[str, Any],
    context: PermissionContext,
) -> PermissionCheck:
    """Check if a tool invocation is permitted / 检查工具调用是否被允许
    Maps to: permissions.ts canUseTool flow
    """
    # Step 1: Blanket deny rules
    if tool.name in context.deny_rules:
        return PermissionCheck(allowed=False, reason=f"Tool '{tool.name}' is denied by policy")

    # Step 2: Always-allow rules
    if tool.name in context.allow_rules:
        return PermissionCheck(allowed=True)

    # Step 3: Mode-based decision
    if context.mode == PermissionMode.BYPASS:
        return PermissionCheck(allowed=True)

    if context.mode == PermissionMode.AUTO:
        # Auto mode: allow read-only, prompt for writes
        if tool.is_read_only:
            return PermissionCheck(allowed=True)
        if tool.risk_level == RiskLevel.LOW:
            return PermissionCheck(allowed=True)
        # Medium/High risk → needs approval
        return PermissionCheck(allowed=False, reason="Auto mode: requires approval for non-read operations")

    # DEFAULT mode: tool-specific check
    return tool.check_permissions(arguments, context)  # type: ignore


def is_dangerous_command(command: str) -> bool:
    """Comprehensive check if a shell command matches dangerous patterns.
    Maps to: bashSecurity.ts dangerous command detector

    Checks:
    1. Exact match against system-destructive commands
    2. Regex match against git/package/infra/exfil patterns
    3. Protected file write detection
    """
    cmd_lower = command.strip().lower()

    # Check system-destructive exact matches
    for pattern in SYSTEM_DESTRUCTIVE:
        if pattern in cmd_lower:
            return True

    # Check regex patterns (git, package, infra, exfil)
    for regex in _DANGEROUS_REGEXES:
        if regex.search(command):
            return True

    return False


def is_protected_file(path: str) -> bool:
    """Check if a file path is in the protected list.
    Maps to: bashPermissions.ts protected path logic
    """
    basename = os.path.basename(path)
    for protected in PROTECTED_PATHS:
        if protected.endswith("/"):
            # Directory pattern
            if f"/{protected}" in path or path.startswith(protected):
                return True
        elif basename == protected:
            return True
    return False
