# Claw Agent — 供应商无关的 Python 智能体框架

> 逆向自 **Claude Code** 的闭源架构，用干净可读的 Python 重建。

<div align="center">
  <img src="https://img.shields.io/badge/架构-Claude_Code-blue" alt="Architecture">
  <img src="https://img.shields.io/badge/Python-3.11+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/供应商-无关-success" alt="Provider Agnostic">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

[🇺🇸 English Version](./README.md)

---

## 为什么选择 Claw Agent？

大多数智能体框架要么过于玩具化、要么过度封装。本项目提取了 Anthropic Claude Code 的架构精髓，用**可读、可扩展的 Python** 重建——每个核心模块不超过 200 行。

- **供应商无关** — 同一套工作流无缝驱动 OpenAI、Claude、Gemini、DeepSeek、MiniMax、Kimi、Qwen
- **工业级** — 异步多代理协调、安全沙箱、自动压缩、持久化记忆
- **透明** — 无黑盒。标准 `asyncio`，干净的 dataclass，无魔法 AST 拦截
- **可扩展** — `PromptBuilder` 将框架护栏与你的业务逻辑彻底解耦

---

## 核心特性

### Engine 循环与流式重入
异步 `Engine` 编排 LLM ↔ 工具循环。当后台 Worker 仍在运行时，引擎**挂起**而非退出——通过 `asyncio.Queue` 等待 Worker 通知，然后自动重新进入 LLM 循环。

### 自动压缩 (Auto-Compact)
对话接近上下文窗口限制时自动压缩历史。使用 LLM 生成摘要将海量 token 压缩为一条消息——零上下文溢出。

### 会话持久化 (Session Persistence)
在 `~/.claw/sessions/` 维护结构化 Markdown 笔记。后台 forked 子代理自动提取关键上下文。压缩时使用会话笔记替代 LLM 生成的摘要，恢复更快、保真度更高。

### Auto-Dream 记忆巩固
后台"无头引擎"定期唤醒，修剪、整理和压缩 `.md` 记忆文件——跨会话保持上下文相关且整洁。

### 多代理协调器
通过 `asyncio.Queue` 派发并行后台 Worker。Worker 以 `<task-notification>` 事件回报结果。主引擎全程保持响应。

### Hook 系统
完整的生命周期钩子：`PRE_TOOL_USE`、`POST_TOOL_USE`、`POST_SAMPLING`、`PRE_COMPACT`、`POST_COMPACT`、`STOP`。支持阻塞和即发即忘两种模式。

### CLAW.md — 项目级指令
自动发现并注入系统提示词的指令文件：

| 文件 | 范围 |
|------|------|
| `~/.claw/CLAW.md` | 全局 |
| `CLAW.md` / `.claw/CLAW.md` | 项目级 |
| `.claw/rules/*.md` | 模块化规则 |
| `CLAW.local.md` | 本地（不提交） |

支持 `@include` 引用、YAML frontmatter 和优先级排序。

### 网络搜索（供应商无关）
4 个可插拔后端，从环境变量自动检测：

| 后端 | 环境变量 | 说明 |
|------|---------|------|
| **Tavily**（推荐） | `TAVILY_API_KEY` | AI 优化，每月 1000 次免费 |
| **Brave Search** | `BRAVE_API_KEY` | 隐私优先，每月 2000 次免费 |
| **SerpAPI** | `SERPAPI_API_KEY` | Google 结果 API |
| **DuckDuckGo** | *（无需）* | 零配置后备 |

### 内置工具
`bash`、`file_read`、`file_edit`、`file_write`、`glob`、`grep`、`web_search`、`web_fetch`、`ask_user`

---

## 快速开始

### 安装

```bash
git clone https://github.com/leobikotech/claw-agent.git
cd claw-agent

pip install -e .            # 基础（OpenAI 兼容供应商）
pip install -e ".[all]"     # + Claude + Gemini SDK
```

### 设置 API Key

```bash
export OPENAI_API_KEY="..."     # 或 DEEPSEEK_API_KEY、MINIMAX_API_KEY 等
export TAVILY_API_KEY="..."     # 可选：启用网络搜索
```

### 运行 CLI

```bash
python3 -m claw_agent
```

### 作为库使用

```python
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools

async def main():
    config = Config(provider="openai")
    config.system_prompt = "你是一名资深代码审计员。"
    engine = Engine(config=config, tools=get_default_tools())

    async for event in engine.run("分析当前目录下的 main.py"):
        if event["type"] == "done":
            print(event["content"])

asyncio.run(main())
```

### MCP 集成

```python
from claw_agent.integrations import MCPManager, MCPServerConfig

mcp = MCPManager()
await mcp.connect_all([
    MCPServerConfig(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
])
await mcp.discover_tools_async(engine.registry)
```

---

## 示例

| 脚本 | 说明 |
|------|------|
| `simple_agent.py` | 10 行代码的最小智能体 |
| `memory_example.py` | Auto-Dream 记忆巩固 |
| `coordinator_example.py` | 多代理后台 Worker |
| `mcp_example.py` | MCP 工具服务器集成 |
| `custom_tool.py` | 通过 `@tool` 装饰器自定义工具 |
| `multi_provider.py` | 多 LLM 路由 |

---

## 项目结构

```
claw_agent/
├── core/              # 引擎循环、钩子、消息、工具基类、权限
├── providers/         # LLM 供应商（OpenAI、Anthropic、Gemini）
├── instructions/      # CLAW.md 发现 + PromptBuilder
├── memory/            # 自动压缩、会话持久化、Dream 巩固
├── tools/             # 内置工具（bash、文件、glob、grep、搜索等）
├── agents/            # 多代理协调器
├── integrations/      # MCP 客户端
├── config.py          # 配置
└── __main__.py        # CLI 入口
```

---

## 扩展

**新增 LLM 供应商** — 继承 `LLMProvider`，实现 `chat()` 方法。Engine 会处理工具 Schema 和响应解析。

**新增自定义工具** — 使用 `@tool` 装饰器：

```python
from claw_agent import tool

@tool("db_query", description="查询内部数据库", parameters={
    "type": "object", "properties": {"sql": {"type": "string"}}
})
async def db_query(args, ctx):
    return run_internal_sql(args['sql'])
```

---

## Star 历史

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

## 许可证

MIT
