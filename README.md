# Claw Agent — The Industrial-Grade Python Agent Framework

> **Today's Frameworks Serve Developers 👨‍💻. Tomorrow's Architectures Serve Agents 🤖.**
> *Claw Agent: Bridging the gap between black-box AI tools and transparent, research-ready Agentic Operating Systems.*

<div align="center">
  <img src="https://img.shields.io/badge/Architecture-Claude_Code-blue" alt="Architecture">
  <img src="https://img.shields.io/badge/Language-Python_3.9+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/Provider-Agnostic-success" alt="Provider Agnostic">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

[🇨🇳 简体中文文档 (Chinese Version)](./readme_cn.md)

---

## ⚡ Why Claw Agent?

The market is flooded with out-of-the-box agent frameworks, but they are often either too "toy-like" or over-encapsulated, making it impossible to understand the underlying logic. 

This project was born from a deep parse and rewrite of the **closed-source Anthropic CLI tool `Claude Code`**. We extracted its most powerful architectural advantages and rebuilt them using extremely **elegant, highly readable, and extensible Python code**:

1. **Industrial-Grade Agentic Intelligence**: Inherits the original's powerful asynchronous multi-agent coordination, background memory consolidation (AutoDream), security sandbox mechanisms, and seamless context Auto-Compact.
2. **Reject Black Boxes; Ultimate Readability**: Discards the obscure AST interceptions and massive Generators from the TypeScript original, opting instead for standard Python `asyncio` queues. Core modules are under 200 lines—**you can read it like a textbook**.
3. **Highly Extensible Vertical Foundation**: The pioneering `PromptBuilder` design decouples core security logic from your specific business logic. Whether building a "Medical Code Auditor" or an "Automated Testing Hacker", just inject your business instructions; the foundation remains solid. Includes full `Glob` and `Grep` file tools natively.
4. **Unified Provider Layer**: Provider Agnostic! Just inject an API Key, and you can seamlessly drive the exact same agent workflow using OpenAI, Claude, Gemini, DeepSeek, or MiniMax.

---

## ⚡ Core Architecture

#### 1. Modular PromptBuilder
Say goodbye to thousands of lines of "spaghetti" Prompts. The system is strictly divided into: `System Guardrails` + `Tool Safety Guidelines` + `Tone/Style` + `Business Customization (DOMAIN_INSTRUCTIONS)`. This provides an extremely clean injection experience when building agents.

#### 2. True Async Coordinator with Streaming Re-entry
Unlike the blocking workflows in most Python frameworks, we natively implemented non-blocking streams via `asyncio.Queue`. When the Coordinator spawns a sub-agent task, the LLM is **not blocked** and can immediately handle other sub-tasks. Completed background tasks return to the main loop via `<task-notification>` events in the queue—just like an incoming WeChat popup.

The **Streaming Re-entry** architecture mirrors the TypeScript `QueryEngine`: when background workers are still running but the LLM produces no tool calls, the engine **suspends** instead of exiting—awaiting the event queue for worker notifications, then re-entering the LLM loop. This eliminates premature termination and ensures reliable long-running orchestration.

#### 3. Grounded Auto-Dream & Relevance Injection
More than simple "RAG". Before each query, the engine actively injects previously relevant file memories. In the background, upon hitting thresholds, a "Headless Engine" wakes up to read messy `.md` memory files, prunes outdated facts, organizes timelines, and condenses indices—keeping context pristine. Dream engines support cooperative abort propagation so they don't hang the process on Ctrl+C.

#### 4. Zero Context Overflow via Auto-Compact
As conversations grow long, approaching 128k/200k token limits, the system intercepts and triggers an LLM summarization compression algorithm in the background—condensing massive token counts into a single core message to completely prevent crashes.

#### 5. Background Task Lifecycle Management
All background executions—from spawned workers to `run_in_background` shell commands—are tracked in the engine's **Background Task Registry** (`_bg_tasks`). Every background process produces a `<task-notification>` on completion, and the engine's streaming re-entry loop keeps the main loop alive until all tasks finish. Tools access the live abort signal via `ToolContext.is_aborted`, ensuring cooperative cancellation works even for long-running tool executions.

#### 6. CLAW.md — Project-Level Instructions
Inspired by Claude Code's `CLAUDE.md`, Claw Agent supports **project-level instruction files** that are automatically discovered and injected into the system prompt. This is the most important mechanism for customizing agent behavior per-project:

| File | Priority | Purpose |
|------|----------|---------|
| `~/.claw/CLAW.md` | Global | Private instructions for all projects |
| `~/.claw/rules/*.md` | Global rules | Modular global rules |
| `CLAW.md` | Project | Checked into the codebase |
| `.claw/CLAW.md` | Project | Alternative location |
| `.claw/rules/*.md` | Project rules | Modular project rules (e.g., `security.md`, `style.md`) |
| `CLAW.local.md` | Local | Private per-project (gitignored) |

**Features:**
- **Auto-discovery**: Files are found by walking from CWD upward to the filesystem root
- **@include**: Reference other files with `@./path`, `@~/path`, or `@/absolute/path`
- **Frontmatter**: YAML frontmatter with `paths:` field for conditional rules (only apply to matching files)
- **HTML comment stripping**: `<!-- comments -->` are removed from output
- **Priority ordering**: Later-loaded files (closer to CWD) have higher priority

#### 7. WebSearch — Provider-Agnostic Web Search
Modern agents need real-time information. The original Claude Code relies on Anthropic's native `web_search_20250305` beta—which only works with Claude. Claw Agent rearchitects this as a **provider-agnostic** tool with 4 pluggable search backends:

| Backend | Env Var | Highlights |
|---------|---------|------------|
| **Tavily** (recommended) | `TAVILY_API_KEY` | AI-optimized, returns extracted page content, 1000 free/month |
| **Brave Search** | `BRAVE_API_KEY` | Privacy-focused, 2000 free queries/month |
| **SerpAPI** | `SERPAPI_API_KEY` | Google Search results via API |
| **DuckDuckGo** | *(none needed)* | Zero-config fallback, no API key required |

The backend is auto-detected from environment variables. Results include structured content, markdown-linked sources, and the LLM is prompted to cite them. Domain filtering (`allowed_domains` / `blocked_domains`) works across all backends.

---

## ⚡ Examples

All key features are distilled into ready-to-run example scripts inside the `examples/` directory:

1. **`simple_agent.py`**: Start a minimal agent with just 10 lines of code.
2. **`memory_example.py`**: Demonstrates context auto-injection and background Auto-Dream consolidation.
3. **`coordinator_example.py`**: Advanced Multi-Agent! Spawn parallel workers in the background and receive asynchronous reports.
4. **`mcp_example.py`**: Mount official `server-filesystem` MCP tools in just 4 lines.
5. **`custom_tool.py`**: How to rapidly grant AI agents your local Python functions using `@tool`.
6. **`multi_provider.py`**: Seamless multi-LLM routing within a single codebase.

---

## ⚡ Getting Started

### Step 1: Install & Environment

Clone this repository and install via pip.

```bash
git clone https://github.com/leobikotech/claw-agent.git
cd claw-agent

# Basic Installation: Supports OpenAI specs, DeepSeek, MiniMax, etc.
pip install -e .

# Full Installation: Includes official Claude SDK & Google GenAI
pip install -e ".[all]"
```

Set your preferred LLM API keys via environment variables:
```bash
export MINIMAX_API_KEY="your_minimax_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

**(Optional) Enable Web Search** — set ONE of the following to unlock real-time search:
```bash
# Recommended: Tavily — AI-optimized, returns full page content
export TAVILY_API_KEY="tvly-xxxxxxxxxxxxx"   # Get yours at https://tavily.com

# Alternative: Brave Search — privacy-focused, generous free tier
# export BRAVE_API_KEY="BSAxxxxxxxxxxxxx"    # https://brave.com/search/api

# Alternative: SerpAPI — Google Search results
# export SERPAPI_API_KEY="xxxxxxxxxxxxx"      # https://serpapi.com

# If no key is set, DuckDuckGo is used automatically (free, no signup)
```

### Step 2: Out-Of-The-Box CLI

Comes with a built-in REPL similar to Claude Code:

```bash
python3 -m claw_agent
```

### Step 3: Build Your Private Domain Agent

You can build your own customized application on this skeleton with less than 20 lines of code:

```python
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools

async def main():
    # 1. Automatically detect environment variables and initialize Provider
    config = Config(provider="minimax")  
    
    # 2. Customize your vertical domain logic (Framework guardrails merge automatically)
    config.system_prompt = (
        "You are a senior quantitative trading code auditor. "
        "Focus on avoiding future functions."
    )
    
    # 3. Initialize Engine and start the event loop
    engine = Engine(config=config, tools=get_default_tools())
    
    # 4. Listen to the conversation event stream
    async for event in engine.run("Help me analyze main.py in the current directory."):
        if event["type"] == "done":
            print(event["content"])

asyncio.run(main())
```

### Step 4: Advanced - MCP & Async Coordinator

**Connect external MCP Protocol**:
```python
from claw_agent.integrations import MCPManager, MCPServerConfig

mcp = MCPManager()
# Mount hundreds of official MCP integrations like GitHub
await mcp.connect_all([
    MCPServerConfig(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
])
await mcp.discover_tools_async(engine.registry)
```

**Build your enterprise cluster with Coordinator**:
```python
from claw_agent.agents.coordinator import Coordinator
import asyncio

# Open an async event return queue
event_queue = asyncio.Queue()

# Grant the main Engine the tool/ability to "create sub-agents"
coord_tools = Coordinator.get_coordinator_tools(config, worker_tools, event_queue)
engine = Engine(config=config, tools=coord_tools, event_queue=event_queue)

# When task executing, the AI can independently spawn_worker in the background, 
# while the main program stays clean pulling incoming reports!
```

---

## 📦 Project Structure

```
claw_agent/
├── core/              # Minimal core: engine loop, hooks, messages, tools, permissions
├── providers/         # LLM providers: OpenAI, Anthropic, Gemini (one file per provider)
├── instructions/      # CLAW.md discovery + system prompt builder
├── memory/            # Persistent memory, dream consolidation, auto-compact
├── tools/             # Built-in tools: ask_user, bash, file, glob, grep, web_fetch, web_search
├── agents/            # Multi-agent coordinator + sub-agent
├── integrations/      # External integrations (MCP client)
├── config.py          # Configuration dataclass
└── __main__.py        # CLI REPL entry point
```

---

## 🛠️ Adding Models or Tools?

**Add a new LLM Provider：**
Just inherit `LLMProvider` and implement a simple `chat()` method. The Engine handles the rest (Tool schemas, sanitization, parsing).

**Add your custom Tool：**
Use the `@tool` decorator to turn any Python function into an LLM-ready tool:

```python
from claw_agent import tool

# @tool registers the description and JSON schema parameters automatically.
@tool("db_query", description="Query internal DB", parameters={"type": "object", "properties": {"sql": {"type": "string"}}})
async def db_query(args, ctx):
    # Execute the query securely
    return run_internal_sql(args['sql'])
```

## ⭐ Star History

If Claw Agent helps make your agents more powerful and transparent, give us a star! ⭐

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

## 📄 License
MIT License. Build your true multi-agent moat with confidence!
