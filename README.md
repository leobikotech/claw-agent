# Claw Agent — Provider-Agnostic Python Agent Framework

> Reverse-engineered from **Claude Code's** closed-source architecture, rebuilt in clean, readable Python.

<div align="center">
  <img src="https://img.shields.io/badge/Architecture-Claude_Code-blue" alt="Architecture">
  <img src="https://img.shields.io/badge/Python-3.11+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/Provider-Agnostic-success" alt="Provider Agnostic">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

[🇨🇳 简体中文文档](./readme_cn.md)

---

## Why Claw Agent?

Most agent frameworks are either too toy-like or over-encapsulated. This project extracts the architectural essence of Anthropic's Claude Code and rebuilds it as **readable, extensible Python** — every core module under 200 lines.

- **Provider Agnostic** — Same workflow across OpenAI, Claude, Gemini, DeepSeek, MiniMax, Kimi, Qwen
- **Industrial-Grade** — Async multi-agent coordination, security sandbox, auto-compact, persistent memory
- **Transparent** — No black boxes. Standard `asyncio`, clean dataclasses, no magic AST hacks
- **Extensible** — `PromptBuilder` decouples framework guardrails from your domain logic cleanly

---

## Core Features

### Engine Loop & Streaming Re-entry
The async `Engine` orchestrates the LLM ↔ Tool loop. When background workers are still running, the engine **suspends** rather than exiting — awaiting worker notifications via `asyncio.Queue`, then re-entering the LLM loop automatically.

### Auto-Compact
Automatically compresses conversation history when approaching context window limits. Uses LLM-generated summaries to condense tokens into a single message — zero context overflow.

### Session Persistence
Maintains structured markdown notes (`~/.claw/sessions/`) across conversations. A forked sub-agent extracts key context in the background. On compact, session notes replace LLM-generated summaries for faster, higher-fidelity context recovery.

### Auto-Dream Memory Consolidation
Background "headless engine" periodically wakes up to prune, organize, and condense `.md` memory files — keeping context relevant and clean across sessions.

### Multi-Agent Coordinator
Spawn parallel background workers via `asyncio.Queue`. Workers report results back as `<task-notification>` events. The main engine stays responsive throughout.

### Hook System
Full lifecycle hooks: `PRE_TOOL_USE`, `POST_TOOL_USE`, `POST_SAMPLING`, `PRE_COMPACT`, `POST_COMPACT`, `STOP`. Supports both blocking and fire-and-forget modes.

### CLAW.md — Project-Level Instructions
Auto-discovered instruction files injected into the system prompt:

| File | Scope |
|------|-------|
| `~/.claw/CLAW.md` | Global |
| `CLAW.md` / `.claw/CLAW.md` | Project |
| `.claw/rules/*.md` | Modular rules |
| `CLAW.local.md` | Local (gitignored) |

Supports `@include` references, YAML frontmatter, and priority ordering.

### Web Search (Provider-Agnostic)
4 pluggable backends — auto-detected from env vars:

| Backend | Env Var | Notes |
|---------|---------|-------|
| **Tavily** *(recommended)* | `TAVILY_API_KEY` | AI-optimized, 1000 free/month |
| **Brave Search** | `BRAVE_API_KEY` | Privacy-focused, 2000 free/month |
| **SerpAPI** | `SERPAPI_API_KEY` | Google results via API |
| **DuckDuckGo** | *(none)* | Zero-config fallback |

### Built-in Tools
`bash`, `file_read`, `file_edit`, `file_write`, `glob`, `grep`, `web_search`, `web_fetch`, `ask_user`

---

## Quick Start

### Install

```bash
git clone https://github.com/leobikotech/claw-agent.git
cd claw-agent

pip install -e .            # Base (OpenAI-compatible providers)
pip install -e ".[all]"     # + Claude + Gemini SDKs
```

### Set API Keys

```bash
export OPENAI_API_KEY="..."     # or DEEPSEEK_API_KEY, MINIMAX_API_KEY, etc.
export TAVILY_API_KEY="..."     # Optional: enables web search
```

### Run the CLI

```bash
python3 -m claw_agent
```

### Use as a Library

```python
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools

async def main():
    config = Config(provider="openai")
    config.system_prompt = "You are a senior code auditor."
    engine = Engine(config=config, tools=get_default_tools())

    async for event in engine.run("Analyze main.py in the current directory."):
        if event["type"] == "done":
            print(event["content"])

asyncio.run(main())
```

### MCP Integration

```python
from claw_agent.integrations import MCPManager, MCPServerConfig

mcp = MCPManager()
await mcp.connect_all([
    MCPServerConfig(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
])
await mcp.discover_tools_async(engine.registry)
```

---

## Examples

| Script | Description |
|--------|-------------|
| `simple_agent.py` | Minimal agent in 10 lines |
| `memory_example.py` | Auto-Dream memory consolidation |
| `coordinator_example.py` | Multi-agent with background workers |
| `mcp_example.py` | MCP tool server integration |
| `custom_tool.py` | Custom tools via `@tool` decorator |
| `multi_provider.py` | Multi-LLM routing |

---

## Project Structure

```
claw_agent/
├── core/              # Engine loop, hooks, messages, tools, permissions
├── providers/         # LLM providers (OpenAI, Anthropic, Gemini)
├── instructions/      # CLAW.md discovery + PromptBuilder
├── memory/            # Auto-compact, session persistence, dream consolidation
├── tools/             # Built-in tools (bash, file, glob, grep, search, etc.)
├── agents/            # Multi-agent coordinator
├── integrations/      # MCP client
├── config.py          # Configuration
└── __main__.py        # CLI entry point
```

---

## Extending

**Add an LLM Provider** — Inherit `LLMProvider`, implement `chat()`. The Engine handles tool schemas and response parsing.

**Add a Custom Tool** — Use the `@tool` decorator:

```python
from claw_agent import tool

@tool("db_query", description="Query internal DB", parameters={
    "type": "object", "properties": {"sql": {"type": "string"}}
})
async def db_query(args, ctx):
    return run_internal_sql(args['sql'])
```

---

## Star History

<div align="center">
  <a href="https://star-history.com/#leobikotech/claw-agent&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=leobikotech/claw-agent&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=leobikotech/claw-agent&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=leobikotech/claw-agent&type=Date" />
    </picture>
  </a>
</div>

---

## License

MIT
