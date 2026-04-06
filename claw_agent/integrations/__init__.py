"""
Integrations — 外部集成

External service integrations that extend agent capabilities.
扩展智能体能力的外部服务集成。
"""
from claw_agent.integrations.mcp_client import MCPManager, MCPServerConfig, MCPBridgeTool

__all__ = ["MCPManager", "MCPServerConfig", "MCPBridgeTool"]
