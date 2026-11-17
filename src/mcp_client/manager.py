"""
MCP client manager.

The MCP SDK is async, while the rest of the app (LangGraph nodes, FastAPI
handlers in the thread pool) is synchronous. So the manager runs ONE dedicated
asyncio event loop in a background thread that owns every server connection
(connections must be opened and closed by the same task), and exposes blocking
methods (`call_tool`, `status`, ...) that hop onto that loop.

A server that fails to start never blocks the others or the API.
"""
import asyncio
import json
import logging
import os
import re
import shutil
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SEP = "__"  # qualified tool name: <server>__<tool>


class MCPToolError(Exception):
    """A tool call failed (unknown tool, blocked, timed out, or the server reported an error)."""


def _attr(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    """MCP SDK 2.x uses snake_case fields, 1.x camelCase: accept both."""
    value = getattr(obj, snake, None)
    return value if value is not None else getattr(obj, camel, default)


def _expand(value: str, extra: Dict[str, str]) -> str:
    return _VAR.sub(lambda m: extra.get(m.group(1), os.environ.get(m.group(1), "")), value)


def load_config(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Read the enabled servers from the JSON config ({} if missing/invalid)."""
    path = Path(path or settings.mcp_config_path)
    if not path.exists():
        logger.info(f"MCP config not found: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Invalid MCP config {path}: {e}")
        return {}
    servers = data.get("servers", {})
    return {name: cfg for name, cfg in servers.items()
            if isinstance(cfg, dict) and cfg.get("enabled", True) and cfg.get("command")}


class MCPManager:
    def __init__(self, servers: Optional[Dict[str, Dict[str, Any]]] = None):
        """`servers` overrides the JSON config (used by tests)."""
        self._servers_override = servers
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._started = threading.Event()   # loop is running
        self._ready = threading.Event()     # every server has connected or failed
        self._lock = threading.Lock()

        self._sessions: Dict[str, Any] = {}
        self._tools: Dict[str, Dict[str, Any]] = {}      # qualified name -> info
        self._status: Dict[str, Dict[str, Any]] = {}     # server -> {connected, error, tools}

    # ------------------------------------------------------------ lifecycle

    def start(self, wait: bool = False, timeout: Optional[float] = None) -> None:
        """Connect to all configured servers in the background (idempotent)."""
        with self._lock:
            if self._thread is not None:
                return
            if not settings.mcp_enabled and self._servers_override is None:
                logger.info("MCP disabled (MCP_ENABLED=false)")
                self._ready.set()
                return
            self._thread = threading.Thread(target=self._run_loop, name="mcp-client", daemon=True)
            self._thread.start()
        if wait:
            self._ready.wait(timeout)

    def stop(self) -> None:
        loop, stop = self._loop, self._stop
        if loop and stop and loop.is_running():
            loop.call_soon_threadsafe(stop.set)
        if self._thread:
            self._thread.join(timeout=10)
        self._thread = None
        self._sessions.clear()
        self._tools.clear()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"MCP loop crashed: {e}", exc_info=True)
        finally:
            self._ready.set()
            loop.close()

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        self._started.set()
        servers = self._servers_override if self._servers_override is not None else load_config()
        # Everything below runs in THIS task, so the connections are also closed by it.
        async with AsyncExitStack() as stack:
            for name, cfg in servers.items():
                await self._connect(stack, name, cfg)
            self._ready.set()
            await self._stop.wait()

    async def _connect(self, stack: AsyncExitStack, name: str, cfg: Dict[str, Any]) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._status[name] = {"connected": False, "error": None, "tools": []}
        server_stack = AsyncExitStack()
        try:
            extra = {"WORKSPACE": str(settings.mcp_workspace_dir)}
            command = _expand(cfg["command"], extra)
            command = shutil.which(command) or command  # npx -> npx.cmd on Windows
            params = StdioServerParameters(
                command=command,
                args=[_expand(a, extra) for a in cfg.get("args", [])],
                env={k: _expand(v, extra) for k, v in (cfg.get("env") or {}).items()} or None,
            )

            # asyncio.timeout (not wait_for): the connection's cancel scopes must be entered
            # and exited by the same task, and wait_for would run this in a child task.
            async with asyncio.timeout(settings.mcp_connect_timeout):
                read, write = await server_stack.enter_async_context(stdio_client(params))
                session = await server_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = (await session.list_tools()).tools

            allowed = cfg.get("allowed_tools")
            names: List[str] = []
            for tool in listed:
                if allowed is not None and tool.name not in allowed:
                    continue
                qualified = f"{name}{_SEP}{tool.name}"[:64]
                schema = dict(_attr(tool, "input_schema", "inputSchema") or {})
                schema.setdefault("type", "object")
                schema.setdefault("properties", {})
                self._tools[qualified] = {
                    "server": name,
                    "tool": tool.name,
                    "description": (tool.description or "")[:1000],
                    "schema": schema,
                }
                names.append(tool.name)

            self._sessions[name] = session
            self._status[name].update(connected=True, tools=names)
            logger.info(f"MCP server '{name}' connected with tools: {names}")
            # Hand the session's lifetime to the outer stack (closed on stop())
            stack.push_async_exit(server_stack)
        except BaseException as e:  # noqa: BLE001 - one bad server must not break the rest
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            msg = "timed out" if isinstance(e, asyncio.TimeoutError) else (str(e) or type(e).__name__)
            logger.error(f"MCP server '{name}' failed to start: {msg}")
            self._status[name]["error"] = msg[:300]
            try:
                await server_stack.aclose()
            except BaseException:  # noqa: BLE001
                pass

    # ----------------------------------------------------------------- query

    def has_tools(self) -> bool:
        return bool(self._tools)

    def tool_specs(self) -> List[Dict[str, Any]]:
        """Tools in OpenAI function-calling format (what ChatNVIDIA.bind_tools accepts)."""
        return [
            {"type": "function", "function": {
                "name": qualified, "description": info["description"], "parameters": info["schema"]}}
            for qualified, info in self._tools.items()
        ]

    def describe(self, qualified: str) -> Optional[Dict[str, Any]]:
        return self._tools.get(qualified)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": settings.mcp_enabled or self._servers_override is not None,
            "ready": self._ready.is_set(),
            "servers": {k: dict(v) for k, v in self._status.items()},
            "tool_count": len(self._tools),
        }

    # ------------------------------------------------------------------ call

    def call_tool(self, qualified: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Run a tool and return its text output (blocking, truncated)."""
        info = self._tools.get(qualified)
        if info is None:
            raise MCPToolError(f"Unknown tool: {qualified}")
        if not self._loop or not self._loop.is_running():
            raise MCPToolError("MCP client is not running")
        arguments = arguments or {}

        url = arguments.get("url")
        if isinstance(url, str):
            from src.tools.web_search import _is_public_url  # SSRF guard shared with our scraper
            if not _is_public_url(url):
                raise MCPToolError("Blocked: URL is not a public address")

        future = asyncio.run_coroutine_threadsafe(self._call(info, arguments), self._loop)
        try:
            return future.result(timeout=settings.mcp_tool_timeout + 5)
        except TimeoutError as e:
            future.cancel()
            raise MCPToolError(f"Tool timed out after {settings.mcp_tool_timeout}s") from e

    async def _call(self, info: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        session = self._sessions.get(info["server"])
        if session is None:
            raise MCPToolError(f"Server '{info['server']}' is not connected")
        try:
            result = await asyncio.wait_for(
                session.call_tool(info["tool"], arguments), timeout=settings.mcp_tool_timeout
            )
        except asyncio.TimeoutError as e:
            raise MCPToolError(f"Tool timed out after {settings.mcp_tool_timeout}s") from e
        except Exception as e:
            raise MCPToolError(f"Tool call failed: {e}") from e

        parts: List[str] = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if text is None and getattr(block, "resource", None) is not None:
                text = getattr(block.resource, "text", None)
            parts.append(text if text is not None else f"[{getattr(block, 'type', 'non-text')} content omitted]")
        output = "\n".join(parts).strip() or "(empty result)"
        if len(output) > settings.mcp_max_output_chars:
            output = output[: settings.mcp_max_output_chars] + "\n...[truncated]"
        if _attr(result, "is_error", "isError", False):
            raise MCPToolError(output)
        return output


mcp_manager = MCPManager()
