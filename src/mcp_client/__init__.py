"""MCP client: connects to external MCP servers and exposes their tools to agents."""
from .manager import MCPManager, MCPToolError, mcp_manager

__all__ = ["MCPManager", "MCPToolError", "mcp_manager"]
