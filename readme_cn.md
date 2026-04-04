# Claw Agent — 工业级 Python 智能体框架

> **今天的框架服务开发者 👨‍💻，明天的架构服务智能体 🤖。**
> *Claw Agent：连接黑盒 AI 工具与透明、可研究的智能体操作系统之间的桥梁。*

<div align="center">
  <img src="https://img.shields.io/badge/架构-Claude_Code-blue" alt="Architecture">
  <img src="https://img.shields.io/badge/语言-Python_3.9+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/供应商-无关-success" alt="Provider Agnostic">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

[🇺🇸 English Version](./README.md)

---

## ⚡ 为什么选择 Claw Agent？

市面上充斥着各种开箱即用的智能体框架，但它们要么过于"玩具化"，要么过度封装，底层逻辑根本无法理解。

本项目诞生于对 **Anthropic 闭源 CLI 工具 `Claude Code`** 的深度解析与重写。我们提取了其最强大的架构优势，并使用极其**优雅、高可读性、易扩展的 Python 代码**进行了重构：

1. **工业级智能体智能**：继承原版强大的异步多代理协调、后台记忆巩固（AutoDream）、安全沙箱机制，以及无缝上下文自动压缩。
2. **拒绝黑盒；极致可读**：摒弃 TypeScript 原版中晦涩的 AST 拦截和庞大的 Generator，转而采用标准 Python `asyncio` 队列。核心模块不超过 200 行——**你可以像读教科书一样阅读它**。
3. **高度可扩展的垂直基座**：首创 `PromptBuilder` 设计，将核心安全逻辑与你的业务逻辑彻底解耦。无论是构建"医疗代码审计员"还是"自动化测试黑客"，只需注入你的业务指令；基座始终坚如磐石。原生包含完整的 `Glob` 和 `Grep` 文件工具。
4. **统一供应商层**：Provider 无关！只需注入一个 API Key，即可使用 OpenAI、Claude、Gemini、DeepSeek 或 MiniMax 无缝驱动完全相同的智能体工作流。

---

## ⚡ 核心架构

#### 1. 模块化 PromptBuilder
告别数千行的"意大利面"式提示词。系统严格划分为：`系统护栏` + `工具安全指引` + `语气/风格` + `业务定制 (DOMAIN_INSTRUCTIONS)`。在构建智能体时提供极其干净的注入体验。

#### 2. 真正的异步协调器 + 流式重入架构
不同于大多数 Python 框架中的阻塞式工作流，我们通过 `asyncio.Queue` 原生实现了非阻塞流。当协调器派发子代理任务时，LLM **不会被阻塞**，可以立即处理其他子任务。完成的后台任务通过队列中的 `<task-notification>` 事件返回主循环——就像收到一条微信弹窗通知。

**流式重入**架构映射原版 TypeScript `QueryEngine`：当后台 Worker 仍在运行，但 LLM 没有产生工具调用时，引擎**挂起**而非退出——等待事件队列中的 Worker 通知，然后重新进入 LLM 循环。这消除了过早终止的问题，确保可靠的长时间运行编排。

#### 3. 接地的 Auto-Dream 与关联注入
不仅仅是简单的 "RAG"。在每次查询之前，引擎会主动注入先前相关的文件记忆。在后台，达到阈值后，一个"无头引擎"会唤醒并阅读凌乱的 `.md` 记忆文件，修剪过时事实，整理时间线，压缩索引——保持上下文始终整洁。Dream 引擎支持协作式中止传播，在 Ctrl+C 时不会挂起进程。

#### 4. 零上下文溢出的 Auto-Compact
当对话变长，接近 128k/200k token 限制时，系统会在后台拦截并触发 LLM 摘要压缩算法——将海量 token 压缩为一条核心消息，彻底防止崩溃。

#### 5. 后台任务生命周期管理
所有后台执行——从派发的 Worker 到 `run_in_background` Shell 命令——都在引擎的**后台任务注册表** (`_bg_tasks`) 中被追踪。每个后台进程在完成时都会产生 `<task-notification>`，引擎的流式重入循环会保持主循环活跃直到所有任务完成。工具通过 `ToolContext.is_aborted` 访问实时中止信号，确保即使在长时间运行的工具执行中也能实现协作式取消。

#### 6. CLAW.md — 项目级指令文件
灵感来自 Claude Code 的 `CLAUDE.md`，Claw Agent 支持**项目级指令文件**，自动发现并注入系统提示词。这是按项目定制智能体行为的最重要机制：

| 文件 | 优先级 | 用途 |
|------|--------|------|
| `~/.claw/CLAW.md` | 全局 | 所有项目通用的私有指令 |
| `~/.claw/rules/*.md` | 全局规则 | 模块化的全局规则 |
| `CLAW.md` | 项目级 | 签入代码库的项目指令 |
| `.claw/CLAW.md` | 项目级 | 替代位置 |
| `.claw/rules/*.md` | 项目规则 | 模块化的项目规则（如 `security.md`、`style.md`） |
| `CLAW.local.md` | 本地 | 项目私有、不提交的本地配置 |

**特性：**
- **自动发现**：从 CWD 向上遍历到文件系统根目录查找文件
- **@include 引用**：支持 `@./path`、`@~/path`、`@/absolute/path` 引用其他文件
- **前置元数据**：支持 YAML frontmatter 的 `paths:` 字段实现条件规则（仅对匹配的文件生效）
- **HTML 注释剥离**：`<!-- comments -->` 会从输出中移除
- **优先级排序**：距离 CWD 越近的文件优先级越高（后加载的覆盖先加载的）

#### 7. WebSearch — 供应商无关的网络搜索
现代智能体需要实时信息。原版 Claude Code 依赖 Anthropic 原生的 `web_search_20250305` beta——仅限 Claude 使用。Claw Agent 将其重新架构为**供应商无关**的工具，配备 4 个可插拔搜索后端：

| 后端 | 环境变量 | 亮点 |
|------|---------|------|
| **Tavily**（推荐） | `TAVILY_API_KEY` | AI 优化，返回提取的页面内容，每月 1000 次免费 |
| **Brave Search** | `BRAVE_API_KEY` | 隐私优先，每月 2000 次免费查询 |
| **SerpAPI** | `SERPAPI_API_KEY` | 通过 API 获取 Google 搜索结果 |
| **DuckDuckGo** | *（无需）* | 零配置后备方案，无需 API Key |

后端从环境变量自动检测。结果包含结构化内容、markdown 超链接来源，并提示 LLM 引用来源。域名过滤（`allowed_domains` / `blocked_domains`）在所有后端通用。

---

## ⚡ 示例

所有核心功能都浓缩在 `examples/` 目录下的即运行示例脚本中：

1. **`simple_agent.py`**：仅用 10 行代码启动一个最小智能体。
2. **`memory_example.py`**：演示上下文自动注入和后台 Auto-Dream 巩固。
3. **`coordinator_example.py`**：高级多代理！在后台派发并行 Worker 并接收异步报告。
4. **`mcp_example.py`**：仅用 4 行代码挂载官方 `server-filesystem` MCP 工具。
5. **`custom_tool.py`**：如何使用 `@tool` 快速赋予 AI 智能体你的本地 Python 函数。
6. **`multi_provider.py`**：在单一代码库中实现无缝多 LLM 路由。

---

## ⚡ 快速开始

### 第一步：安装与环境配置

克隆本仓库并通过 pip 安装。

```bash
git clone https://github.com/leobikotech/claw-agent.git
cd claw-agent

# 基础安装：支持 OpenAI 规范、DeepSeek、MiniMax 等
pip install -e .

# 完整安装：包含官方 Claude SDK 和 Google GenAI
pip install -e ".[all]"
```

设置你偏好的 LLM API Key 环境变量：
```bash
export MINIMAX_API_KEY="your_minimax_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

**（可选）启用网络搜索** — 设置以下任一环境变量即可解锁实时搜索：
```bash
# 推荐：Tavily — AI 优化，返回完整页面内容
export TAVILY_API_KEY="tvly-xxxxxxxxxxxxx"   # 在 https://tavily.com 获取

# 备选：Brave Search — 隐私优先，免费额度充足
# export BRAVE_API_KEY="BSAxxxxxxxxxxxxx"    # https://brave.com/search/api

# 备选：SerpAPI — Google 搜索结果
# export SERPAPI_API_KEY="xxxxxxxxxxxxx"      # https://serpapi.com

# 如果未设置任何 Key，自动使用 DuckDuckGo（免费，无需注册）
```

### 第二步：开箱即用的 CLI

自带类似 Claude Code 的内置 REPL：

```bash
python3 -m claw_agent
```

### 第三步：构建你的私有领域智能体

你可以用不到 20 行代码在此骨架上构建你自己的定制应用：

```python
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools

async def main():
    # 1. 自动检测环境变量并初始化 Provider
    config = Config(provider="minimax")  
    
    # 2. 定制你的垂直领域逻辑（框架护栏自动合并）
    config.system_prompt = (
        "你是一名资深量化交易代码审计员，"
        "重点关注避免使用未来函数。"
    )
    
    # 3. 初始化引擎并启动事件循环
    engine = Engine(config=config, tools=get_default_tools())
    
    # 4. 监听对话事件流
    async for event in engine.run("帮我分析当前目录下的 main.py"):
        if event["type"] == "done":
            print(event["content"])

asyncio.run(main())
```

### 第四步：进阶 - MCP 与异步协调器

**连接外部 MCP 协议**：
```python
from claw_agent.integrations import MCPManager, MCPServerConfig

mcp = MCPManager()
# 挂载数百种官方 MCP 集成（如 GitHub）
await mcp.connect_all([
    MCPServerConfig(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
])
await mcp.discover_tools_async(engine.registry)
```

**使用协调器构建你的企业级集群**：
```python
from claw_agent.agents.coordinator import Coordinator
import asyncio

# 打开异步事件返回队列
event_queue = asyncio.Queue()

# 赋予主引擎"创建子代理"的工具/能力
coord_tools = Coordinator.get_coordinator_tools(config, worker_tools, event_queue)
engine = Engine(config=config, tools=coord_tools, event_queue=event_queue)

# 任务执行时，AI 可以自主在后台 spawn_worker，
# 而主程序保持简洁地拉取收到的报告！
```

---

## 📦 项目结构

```
claw_agent/
├── core/              # 最小核心：引擎循环、钩子、消息、工具基类、权限
├── providers/         # LLM 供应商：OpenAI、Anthropic、Gemini（每个供应商一个文件）
├── instructions/      # CLAW.md 发现 + 系统提示词构建器
├── memory/            # 持久化记忆、Dream 巩固、自动压缩
├── tools/             # 内置工具：ask_user、bash、文件、glob、grep、web_fetch、web_search
├── agents/            # 多代理协调器 + 子代理
├── integrations/      # 外部集成（MCP 客户端）
├── config.py          # 配置数据类
└── __main__.py        # CLI REPL 入口
```

---

## 🛠️ 添加模型或工具？

**新增 LLM Provider：**
只需继承 `LLMProvider` 并实现一个简单的 `chat()` 方法。Engine 会处理其余一切（工具 Schema、清洗、解析）。

**新增自定义工具：**
使用 `@tool` 装饰器将任何 Python 函数转为 LLM 可用的工具：

```python
from claw_agent import tool

# @tool 自动注册描述和 JSON Schema 参数
@tool("db_query", description="查询内部数据库", parameters={"type": "object", "properties": {"sql": {"type": "string"}}})
async def db_query(args, ctx):
    # 安全地执行查询
    return run_internal_sql(args['sql'])
```

## ⭐ Star 历史

如果 Claw Agent 让你的智能体更强大、更透明，请给我们一个 star！⭐

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

## 📄 许可证
MIT 许可证。放心构建你的多代理护城河！
