"""
Tools module - web search for the agents (file access / URL fetching go through MCP)
"""

from .web_search import WebSearchTool, web_search_tool

__all__ = ["WebSearchTool", "web_search_tool"]
