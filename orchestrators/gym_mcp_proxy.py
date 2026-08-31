#!/usr/bin/env python3
"""stdio MCP shim that fronts a gym's POST-only JSON-RPC endpoint.

The EnterpriseOps-Gym servers expose a minimal POST-only JSON-RPC endpoint (no
GET/SSE stream, no Mcp-Session-Id), which the benchmark's own hand-rolled MCPClient
speaks fine but Claude Code's full Streamable-HTTP MCP client rejects (connection
"failed" -> zero gym tools). This shim speaks the compliant MCP **stdio** transport to
Claude Code and proxies tools/list + tools/call to the gym over plain POST, injecting
the per-task `x-database-id` and context headers.

Configured entirely via env (so the Agent SDK can launch it per session):
  GYM_MCP_URL        base gym url, e.g. http://localhost:8008   (required)
  GYM_MCP_ENDPOINT   path, default /mcp
  GYM_DATABASE_ID    x-database-id header value
  GYM_EXTRA_HEADERS  JSON object of extra headers (e.g. {"x-hr-token-access": "..."})
"""

import asyncio
import json
import os

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

GYM_URL = os.environ["GYM_MCP_URL"].rstrip("/")
ENDPOINT = os.environ.get("GYM_MCP_ENDPOINT", "/mcp")
DB_ID = os.environ.get("GYM_DATABASE_ID", "")
EXTRA_HEADERS = json.loads(os.environ.get("GYM_EXTRA_HEADERS", "{}"))

server = Server("gym-proxy")


def _headers():
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if DB_ID:
        h["x-database-id"] = DB_ID
    h.update(EXTRA_HEADERS)
    return h


async def _rpc(method, params):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{GYM_URL}{ENDPOINT}", json=payload, headers=_headers())
        r.raise_for_status()
        text = r.text
        if "text/event-stream" in r.headers.get("content-type", "") or text.startswith("event:"):
            for line in text.splitlines():
                if line.startswith("data: "):
                    text = line[6:]
                    break
        return json.loads(text)


@server.list_tools()
async def list_tools():
    data = await _rpc("tools/list", {})
    tools = data.get("result", {}).get("tools", [])
    out = []
    for t in tools:
        out.append(
            types.Tool(
                name=t["name"],
                description=t.get("description") or "",
                inputSchema=t.get("inputSchema") or {"type": "object"},
            )
        )
    return out


@server.call_tool()
async def call_tool(name, arguments):
    data = await _rpc("tools/call", {"name": name, "arguments": arguments or {}})
    if data.get("error"):
        return [types.TextContent(type="text", text=json.dumps({"error": data["error"]}))]
    result = data.get("result")
    # MCP CallToolResult shape: {"content": [...], "isError": bool}
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        blocks = []
        for b in result["content"]:
            if isinstance(b, dict) and b.get("type") == "text":
                blocks.append(types.TextContent(type="text", text=b.get("text", "")))
            else:
                blocks.append(types.TextContent(type="text", text=json.dumps(b)))
        return blocks or [types.TextContent(type="text", text="")]
    return [types.TextContent(type="text", text=json.dumps(result))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
