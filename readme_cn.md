# Claw Agent — 工业级 Python 智能体框架

> **今天的框架服务于开发者 👨‍💻；明天的底座服务于智能体 🤖。**
> *Claw Agent：跨越闭源黑盒与高透明研究级智能体系统的鸿沟。*

<div align="center">
  <img src="https://img.shields.io/badge/Architecture-Claude_Code-blue" alt="Architecture">
  <img src="https://img.shields.io/badge/Language-Python_3.9+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/Provider-Agnostic-success" alt="Provider Agnostic">
</div>

[🇺🇸 English Documentation](./README.md)

---

## ⚡ 为什么选择 Claw Agent？

市面上有无数的开箱即用 Agent 框架，但它们要么过于玩具化，要么过度封装导致您无法理解底层的思考逻辑。

本项目诞生于对 **Anthropic 官方闭源命令行工具 `Claude Code` 源码** 的深度解析与重写。我们提取了其最核心、最惊艳的架构优势，并用极其**优雅、高可读、高可扩展的 Python 代码**进行了重构：

1. **工业级智能体引擎 (Agentic Intelligence)**：继承了原版强大的多代理异步流、后台记忆自我巩固（AutoDream + 实时检索注入）、安全红线机制、以及内置的无缝上下文自动压缩（Auto-Compact）。
2. **拒绝黑盒，极致的解读体验**：抛弃了 TypeScript 原版中极度晦涩的 AST 拦截和庞大的 Generator 强行注入，改用标准的 Python `asyncio` 队列。核心模块单文件不超过 200 行，**您完全可以把它当作一本教科书来阅读**。
3. **高可扩展的垂直领域底座**：首创的 `PromptBuilder` 设计，使得底座安全逻辑与您的垂直业务逻辑完全解耦。无论是开发“医疗代码审计员”、“智能投资分析师”还是“自动化测试黑客”，您只需注入业务指令，底座稳如泰山。系统内置全量 `Glob` 和 `Grep` 文件级搜索与编辑工具。
4. **统一大模型抽象层 (Provider Agnostic)**：自带中间层！只需暴露 API Key，即可毫无违和感地让 MiniMax、OpenAI、Claude、Gemini、DeepSeek、Kimi 等各种模型驱动同一套 Agent 流。

---

## ⚡ 核心架构亮点

#### 1. 模块化提示词大脑 (Modular PromptBuilder)
告别上千行的“面条式” Prompt。系统被严格划分为：`系统底座规则` + `工具使用安全准则` + `输出风格` + `业务定制层(DOMAIN_INSTRUCTIONS)`。您能在开发属于自己的垂直 Agent 时，享受到极其干净的注入体验。

#### 2. 真正的异步多代理编排 (True Async Coordinator)
大部分 Python 框架的“多代理”只是阻塞式的代码执行。我们原生实现了基于 `asyncio.Queue` 的非阻塞流：主节点（Coordinator）派发子任务后，大模型**不阻塞**，可立刻返回处理别的事情。当子任务在后台完成后，结果会以 `<task-notification>` 的形式放入异步队列中，“像微信弹窗一样”唤醒您的主节点继续决策。

#### 3. 会“做梦”的自主记忆库 (Grounded Auto-Dream & Relevance Injection)
这不是简单的“RAG 检索”。引擎除了在每次提问前**自动无缝注入**前置关联记忆，系统还会后台自主判定阈值。在合适时机无感知地唤醒一个内部大模型进程（Headless Engine），读取凌乱的本地 `.md` 记忆文件，剔除过期事实、整理时间线并浓缩索引，永远保持 Agent 上下文清爽。

#### 4. 彻底告别上下文溢出 (Auto-Compact)
当多轮代理长时间在线处理代码，接近 128k 或 200k 极限时，系统自动拦截并在后台触发大模型摘要压缩机制，将数万 Tokens 汇聚为包含核心信息的一条消息，彻底杜绝长时间运行奔溃的痛点。

---

## ⚡ 丰富的示例代码

框架将所有关键特性的使用方法提纯为您可直接运行的示例脚本，尽在 `examples/` 目录下：

1. **`simple_agent.py`**：只需 10 行代码实现基于任意模型的最简 Agent 启动（自动提供文件搜索与 Bash 操作）。
2. **`memory_example.py`**：展示系统如何自动扫描读取本地记忆文件注入到大模型前置上下文，以及触发 Auto-Dream 长时记忆巩固。
3. **`coordinator_example.py`**：高级玩法！利用工具在后台并行派发多个子代理工作流，并在主流程中非阻塞接收事件通知。
4. **`mcp_example.py`**：展示只用 4 行代码挂载官方 `server-filesystem` 等 MCP 协议的能力。
5. **`custom_tool.py`**：示例教学：如何用 `@tool` 快速将您的本地 Python 函数赋予给 AI 代理使用。
6. **`multi_provider.py`**：代码无需更改，自由跨步切换各家顶级大模型的调用。

---

## ⚡ 极速上手指南

### 步骤一：安装与环境准备

克隆本仓库后，使用 `pip` 安装。

```bash
git clone https://github.com/leobikotech/claw-agent.git
cd Claw-Agent

# 基础安装：支持 OpenAI 协议以及适配了全系国产大模型（MiniMax, Kimi, DeepSeek 等）
pip install -e .

# 完全体安装：额外支持官方 Claude SDK 和 Google GenAI
pip install -e ".[all]"
```

设置您偏好的大模型 API Key（通过环境变量自动感应）：
```bash
export MINIMAX_API_KEY="your_minimax_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

### 步骤二：开箱即用的 CLI 终端

内置了一个精简版的类似 `Claude Code` 的交互式 REPL：

```bash
python3 -m claw_agent
```

### 步骤三：开发您的私有垂直 Agent

基于这份底座框架搭建自己的程序只需要不到 20 行代码：

```python
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools

async def main():
    # 1. 自动感应环境变量并初始化 Provider
    config = Config(provider="minimax")  
    
    # 2. 定制您的垂直业务逻辑（框架安全底座规则会被自动合并保护）
    config.system_prompt = (
        "你是一名资深量化交易代码审计员，"
        "重点审计策略以避免未来函数的使用。"
    )
    
    # 3. 初始化 Engine 并启动循环
    engine = Engine(config=config, tools=get_default_tools())
    
    # 4. 监听对话事件流
    async for event in engine.run("帮我分析当前目录下的交易策略 main.py"):
        if event["type"] == "done":
            print(event["content"])

asyncio.run(main())
```

### 步骤四：高阶玩法 —— 接入外接 MCP 与群智网络

**接入外部 MCP (Model Context Protocol)**:
```python
from claw_agent.core.mcp_client import MCPManager, MCPServerConfig

mcp = MCPManager()
# 挂载 Github 等数百个完全遵照标准的官方 MCP 集成
await mcp.connect_all([
    MCPServerConfig(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
])
await mcp.discover_tools_async(engine.registry)
```

**打造您的企业集群流大脑（开启异步 Coordinator）**:
```python
from claw_agent.agents.coordinator import Coordinator
import asyncio

# 开启事件回流队列
event_queue = asyncio.Queue()

# 将“创造子集 Agent 的能力”作为高阶工具赋予给主引擎
coord_tools = Coordinator.get_coordinator_tools(config, worker_tools, event_queue)
engine = Engine(config=config, tools=coord_tools, event_queue=event_queue)

# 当大模型执行主任务时，它可以自主 spawn_worker 在后台并行干活，
# 主程序则一直跑在这个干净的非阻塞循环里收接回传报告！
```

---

## 🛠️ 添加新模型或工具？

**添加新 Provider：**
只需继承 `LLMProvider` 并实现一个单纯的 `chat()` 方法即可，其余所有（工具定义转换、上下文注入、格式清理）全部由引擎完成。

**添加您的工具：**
使用 `@tool` 装饰器即可将任何本地 Python 函数大模型化：

```python
from claw_agent import tool

# @tool 装饰器自动提取描述及 JSON Schema
@tool("db_query", description="Query internal DB", parameters={"type": "object", "properties": {"sql": {"type": "string"}}})
async def db_query(args, ctx):
    # 此处执行安全查询逻辑
    return run_internal_sql(args['sql'])
```

## ⭐ Star History (星标历史)

如果 Claw Agent 帮助到了你构建更强大的 AI 智能体应用，给我们点个星星吧！⭐

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
MIT License. 放心使用这套框架去构建属于你真正的多代理护城河吧！
