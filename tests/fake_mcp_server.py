"""Tiny stdio MCP server used by the tests (no npx/uvx/network needed)."""
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

server = Server("fake")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool()
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


@server.tool()
def big() -> str:
    """Return a very long text."""
    return "x" * 50000


@server.tool()
def fetch(url: str) -> str:
    """Pretend to fetch a URL."""
    return f"fetched {url}"


if __name__ == "__main__":
    server.run()
