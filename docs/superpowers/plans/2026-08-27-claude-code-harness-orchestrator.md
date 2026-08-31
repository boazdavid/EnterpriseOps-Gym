# Claude Code Harness Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `claude_code` orchestrator that drives the Claude Agent SDK against the same per-task gym database as the native ReAct arm, so both arms run GPT-5.5 and only the harness differs.

**Architecture:** A new `orchestrators/claude_code.py` implements the existing `AgentOrchestrator` interface. Its `execute()` builds a `ClaudeAgentOptions` (HTTP MCP servers carrying the per-task `x-database-id` header, GPT-5.5-via-proxy model/env, oracle/plus_10 tool restriction, built-ins on) and runs `claude_agent_sdk.query()`, then maps the streamed messages into the benchmark's existing result schema. The executor's SQL verifiers score it unchanged.

**Tech Stack:** Python 3.9+, `asyncio`, `claude-agent-sdk` (spawns the Claude Code CLI as a subprocess), `pytest` for tests. Existing benchmark uses `httpx`, LangChain.

**Spec:** `docs/superpowers/specs/2026-08-27-claude-code-harness-vs-native-design.md`

## Global Constraints

- MCP tool name format is `mcp__<gym_name>__<tool>`, where `<gym_name>` is the MCP server key (the benchmark's `mcp_server_name`, e.g. `sn-hr-internal`). The `mcp_servers` config key and the `tool_to_server_mapping` server name MUST be the same string so prefixes line up.
- Model is pinned explicitly: `options.model = "azure/gpt-5.5"` AND `env["ANTHROPIC_MODEL"] = "azure/gpt-5.5"`. An unpinned model silently defaults to Opus 5.
- For the GPT-5.5-via-proxy path, DO NOT set `CLAUDE_CODE_USE_BEDROCK`.
- `permission_mode="bypassPermissions"` (zero prompts; auto-approves MCP + built-ins). Built-in tools are left ENABLED (never added to `disallowed_tools`).
- `execute()` MUST return a dict with keys `final_response`, `conversation_flow`, `tools_used`, `tool_results`, `messages` (matching `orchestrators/react.py`). Extra telemetry goes through `get_result_metadata()`.
- `tools_used` contains MCP tool **base names** (the `mcp__<gym>__` prefix stripped), deduplicated in first-use order — comparable to the native arm. Built-in tool usage is tracked separately in metadata, never mixed into `tools_used`.
- Import `claude_agent_sdk` **lazily inside `execute()`**, never at module top level, so the pure helper functions and their unit tests run without the SDK installed.
- No changes to `benchmark/executor.py`, `benchmark/verifier.py`, Ray configs, `compute_score.py`, or `scan_results.py`.

---

### Task 1: Proxy spike — verify the LiteLLM Anthropic `/v1/messages` route (GATE, no commit)

This is a manual environmental gate, not TDD. It must pass before the orchestrator is worth running (though Tasks 2–5 are pure helpers and can proceed in parallel regardless, since they're reused by the Claude-model fallback too).

**Files:**
- Create (throwaway): `scratch/spike_cc_proxy.py`

**Prerequisites:** Node + Claude Code CLI on PATH; `pip install claude-agent-sdk`; an HR gym server running at `http://localhost:8008` with a seed DB available; the LiteLLM proxy reachable.

- [ ] **Step 1: Create a throwaway HR database for the spike**

Use the existing helper to seed one DB and note the returned id:

```python
# scratch/spike_cc_proxy.py
import asyncio, os
from benchmark.mcp_client import create_database_from_file
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

GYM_URL = "http://localhost:8008"
SEED = "gym_dbs/hr.sql"  # adjust to the actual HR seed file path used by the HR tasks

db_id = create_database_from_file(GYM_URL, SEED)
print("db_id:", db_id)

options = ClaudeAgentOptions(
    model="azure/gpt-5.5",
    env={
        "ANTHROPIC_BASE_URL": "https://ete-litellm.ai-models.vpc-int.res.ibm.com",
        "ANTHROPIC_AUTH_TOKEN": "sk-EUohJ0crvlAVQZMub5fb3g",
        "ANTHROPIC_MODEL": "azure/gpt-5.5",
    },
    mcp_servers={
        "sn-hr-internal": {
            "type": "http",
            "url": f"{GYM_URL}/mcp",
            "headers": {"x-database-id": db_id},
        }
    },
    allowed_tools=["mcp__sn-hr-internal__*"],
    permission_mode="bypassPermissions",
    max_turns=6,
)

async def main():
    saw_tool_use = False
    result = None
    async for msg in query(prompt="List the available HR tools and call one read-only tool once.", options=options):
        cls = type(msg).__name__
        print("MSG:", cls)
        for b in getattr(msg, "content", []) or []:
            if hasattr(b, "name") and hasattr(b, "input"):
                saw_tool_use = True
                print("  TOOL_USE:", b.name)
        if isinstance(msg, ResultMessage):
            result = msg
    print("saw_tool_use:", saw_tool_use, "| result.subtype:", getattr(result, "subtype", None))

asyncio.run(main())
```

- [ ] **Step 2: Run the spike**

Run: `python scratch/spike_cc_proxy.py`
Expected: prints a `db_id`, at least one `TOOL_USE: mcp__sn-hr-internal__...`, and `result.subtype: success`.

- [ ] **Step 3: Record the outcome**

If it fails at `/v1/messages` (404/route missing) or tool translation errors: STOP and report — the fallback is the Claude-model comparison from the spec's §8 (`build_cc_model_env` already supports that branch; only the run config changes). If it succeeds, delete `scratch/spike_cc_proxy.py` and continue. Do not commit the spike script.

---

### Task 2: Tool-surface restriction (`compute_disallowed_tools`)

Reproduces oracle / plus_10 by disallowing every MCP tool the task did not select. The executor has already filtered `available_tools` to the mode's subset; `tool_to_server_mapping` still holds every discovered tool.

**Files:**
- Create: `orchestrators/claude_code.py`
- Test: `tests/orchestrators/test_claude_code.py`

**Interfaces:**
- Produces: `compute_disallowed_tools(available_tools: list[dict], tool_to_server_mapping: dict[str, str]) -> list[str]` — returns sorted `mcp__<server>__<tool>` names for every tool in the mapping that is NOT in `available_tools`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrators/test_claude_code.py
from orchestrators.claude_code import compute_disallowed_tools


def test_disallows_the_complement_of_selected_tools():
    available = [{"name": "get_employee"}, {"name": "list_pto"}]
    mapping = {
        "get_employee": "sn-hr-internal",
        "list_pto": "sn-hr-internal",
        "delete_employee": "sn-hr-internal",
        "approve_pto": "sn-hr-internal",
    }
    result = compute_disallowed_tools(available, mapping)
    assert result == [
        "mcp__sn-hr-internal__approve_pto",
        "mcp__sn-hr-internal__delete_employee",
    ]


def test_returns_empty_when_all_tools_selected():
    available = [{"name": "a"}, {"name": "b"}]
    mapping = {"a": "gym", "b": "gym"}
    assert compute_disallowed_tools(available, mapping) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrators/test_claude_code.py -k disallow -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'compute_disallowed_tools'`.

- [ ] **Step 3: Write minimal implementation**

```python
# orchestrators/claude_code.py
"""Claude Code (Claude Agent SDK) orchestrator for EnterpriseOps-Gym."""

from typing import Any, Dict, List, Tuple


def compute_disallowed_tools(
    available_tools: List[Dict[str, Any]], tool_to_server_mapping: Dict[str, str]
) -> List[str]:
    """MCP tools to hide so the surface matches the task's mode (oracle/plus_N).

    `available_tools` is the executor-filtered subset for this mode;
    `tool_to_server_mapping` holds every tool the servers advertised. We disallow
    the complement so Claude Code sees only the selected MCP tools.
    """
    selected = {t.get("name") for t in available_tools}
    disallowed = [
        f"mcp__{server}__{name}"
        for name, server in tool_to_server_mapping.items()
        if name not in selected
    ]
    return sorted(disallowed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrators/test_claude_code.py -k disallow -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add orchestrators/claude_code.py tests/orchestrators/test_claude_code.py
git commit -m "feat(orchestrator): compute disallowed MCP tools for claude_code mode parity"
```

---

### Task 3: MCP servers config (`build_mcp_servers`)

Builds the SDK `mcp_servers` dict from the benchmark's per-task `MCPClient`s, carrying `x-database-id` and `x-*` context headers. Keyed by gym name so tool prefixes match Task 2.

**Files:**
- Modify: `orchestrators/claude_code.py`
- Test: `tests/orchestrators/test_claude_code.py`

**Interfaces:**
- Consumes: `MCPClient` attributes `base_url`, `mcp_endpoint`, `database_id`, `context` (see `benchmark/mcp_client.py`).
- Produces: `build_mcp_servers(mcp_clients: Dict[str, Any]) -> Dict[str, dict]` — one `{"type":"http","url":...,"headers":{...}}` per gym, keyed by gym name.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/orchestrators/test_claude_code.py
from orchestrators.claude_code import build_mcp_servers


class _FakeClient:
    def __init__(self, base_url, mcp_endpoint, database_id, context):
        self.base_url = base_url
        self.mcp_endpoint = mcp_endpoint
        self.database_id = database_id
        self.context = context


def test_build_mcp_servers_sets_headers_and_url():
    clients = {
        "sn-hr-internal": _FakeClient(
            "http://localhost:8008", "/mcp", "db_123", {"user_id": "u1"}
        )
    }
    servers = build_mcp_servers(clients)
    assert servers == {
        "sn-hr-internal": {
            "type": "http",
            "url": "http://localhost:8008/mcp",
            "headers": {"x-database-id": "db_123", "x-user-id": "u1"},
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrators/test_claude_code.py -k build_mcp_servers -v`
Expected: FAIL with `ImportError: cannot import name 'build_mcp_servers'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to orchestrators/claude_code.py

def _context_headers(context: Dict[str, Any]) -> Dict[str, str]:
    """Mirror MCPClient's context->header convention (user_id -> x-user-id)."""
    headers = {}
    for key, value in (context or {}).items():
        if key.lower().startswith("x-"):
            header_key = key
        else:
            header_key = f"x-{key.lower().replace('_', '-')}"
        headers[header_key] = str(value)
    return headers


def build_mcp_servers(mcp_clients: Dict[str, Any]) -> Dict[str, dict]:
    """One HTTP MCP server entry per gym, carrying the per-task DB + context headers."""
    servers = {}
    for gym_name, client in mcp_clients.items():
        url = f"{client.base_url.rstrip('/')}{client.mcp_endpoint}"
        headers = {"x-database-id": client.database_id}
        headers.update(_context_headers(getattr(client, "context", {})))
        servers[gym_name] = {"type": "http", "url": url, "headers": headers}
    return servers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrators/test_claude_code.py -k build_mcp_servers -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orchestrators/claude_code.py tests/orchestrators/test_claude_code.py
git commit -m "feat(orchestrator): build HTTP MCP server config with per-task db/context headers"
```

---

### Task 4: Model + env mapping (`build_cc_model_env`)

Maps the run's LLM settings to the SDK `model` + per-session `env`. Implements the proxy path (v1) plus `anthropic` and `aws_bedrock` branches for the spec's §8 fallback.

**Files:**
- Modify: `orchestrators/claude_code.py`
- Test: `tests/orchestrators/test_claude_code.py`

**Interfaces:**
- Produces: `build_cc_model_env(provider: str, model: str, api_endpoint: str | None, api_key: str | None, region: str | None) -> Tuple[str, Dict[str, str]]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/orchestrators/test_claude_code.py
from orchestrators.claude_code import build_cc_model_env


def test_proxy_provider_maps_to_anthropic_base_url_and_pins_model():
    model, env = build_cc_model_env(
        "vllm", "azure/gpt-5.5",
        "https://ete-litellm.ai-models.vpc-int.res.ibm.com", "sk-abc", None,
    )
    assert model == "azure/gpt-5.5"
    assert env["ANTHROPIC_BASE_URL"] == "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-abc"
    assert env["ANTHROPIC_MODEL"] == "azure/gpt-5.5"
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_bedrock_provider_sets_bedrock_env():
    model, env = build_cc_model_env(
        "aws_bedrock", "us.anthropic.claude-sonnet-5", None, None, "us-east-1",
    )
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-east-1"
    assert env["ANTHROPIC_MODEL"] == "us.anthropic.claude-sonnet-5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrators/test_claude_code.py -k model_env -v`
Expected: FAIL with `ImportError: cannot import name 'build_cc_model_env'`. (Test node ids also match `provider` — use `-k "proxy_provider or bedrock_provider"` if needed.)

- [ ] **Step 3: Write minimal implementation**

```python
# add to orchestrators/claude_code.py

def build_cc_model_env(
    provider: str, model: str, api_endpoint, api_key, region
) -> Tuple[str, Dict[str, str]]:
    """Translate benchmark LLM settings into Claude Agent SDK (model, env)."""
    provider = (provider or "").lower()
    env: Dict[str, str] = {"ANTHROPIC_MODEL": model}

    if provider == "aws_bedrock":
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        if region:
            env["AWS_REGION"] = region
        return model, env

    if provider == "anthropic":
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        return model, env

    # Proxy path (vllm / openrouter / azureopenai / anything OpenAI-compatible behind
    # a LiteLLM Anthropic /v1/messages route). This is the v1 GPT-5.5 comparison.
    if api_endpoint:
        env["ANTHROPIC_BASE_URL"] = api_endpoint
    if api_key:
        env["ANTHROPIC_AUTH_TOKEN"] = api_key
    return model, env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrators/test_claude_code.py -k "proxy_provider or bedrock_provider" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orchestrators/claude_code.py tests/orchestrators/test_claude_code.py
git commit -m "feat(orchestrator): map benchmark LLM config to Claude Agent SDK model/env"
```

---

### Task 5: Message → result-schema mapper (`messages_to_result`)

Converts the SDK's streamed message list into the benchmark result dict + telemetry. Duck-types on attributes so tests need no real SDK objects.

**Files:**
- Modify: `orchestrators/claude_code.py`
- Test: `tests/orchestrators/test_claude_code.py`

**Interfaces:**
- Produces: `messages_to_result(messages: list, gym_hosts: List[str]) -> Tuple[dict, dict]` — returns `(result, metadata)`. `result` has keys `final_response`, `conversation_flow`, `tools_used`, `tool_results`, `messages`. `metadata` has keys `cc_usage`, `cc_cost_usd`, `cc_subtype`, `builtin_tools_used`, `mcp_bypass_suspected`.
- Block/message duck-typing rules: a message is a *result* if it has both `result` and `subtype` attrs; otherwise iterate its `content`. A block is a *tool_result* if it has `tool_use_id`; a *tool_use* if it has `name` and `input`; a *text* block if it has `text`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/orchestrators/test_claude_code.py
from orchestrators.claude_code import messages_to_result


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _msgs():
    text = _Obj(text="working on it")
    tool_use = _Obj(name="mcp__sn-hr-internal__approve_pto", input={"id": 7}, id="t1")
    builtin_use = _Obj(name="Bash", input={"command": "echo hi"}, id="b1")
    assistant = _Obj(content=[text, tool_use, builtin_use])
    tool_res = _Obj(tool_use_id="t1", content=[{"type": "text", "text": "ok"}])
    user = _Obj(content=[tool_res])
    result = _Obj(result="done", subtype="success",
                  usage={"input_tokens": 10, "output_tokens": 5}, total_cost_usd=0.02)
    return [assistant, user, result]


def test_maps_messages_to_result_schema():
    result, meta = messages_to_result(_msgs(), gym_hosts=["localhost:8008"])
    assert result["final_response"] == "done"
    # MCP base name only, deduped; built-ins excluded from tools_used
    assert result["tools_used"] == ["approve_pto"]
    assert meta["builtin_tools_used"] == ["Bash"]
    assert meta["cc_subtype"] == "success"
    assert meta["cc_cost_usd"] == 0.02
    assert meta["cc_usage"] == {"input_tokens": 10, "output_tokens": 5}
    # correlated tool_result carries the tool name + args
    assert result["tool_results"][0]["tool_name"] == "approve_pto"
    assert result["tool_results"][0]["arguments"] == {"id": 7}


def test_flags_mcp_bypass_when_bash_hits_gym_host():
    tool_use = _Obj(name="Bash", input={"command": "curl http://localhost:8008/api/query"}, id="b1")
    assistant = _Obj(content=[tool_use])
    result = _Obj(result="x", subtype="success", usage={}, total_cost_usd=0.0)
    _, meta = messages_to_result([assistant, result], gym_hosts=["localhost:8008"])
    assert meta["mcp_bypass_suspected"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrators/test_claude_code.py -k "messages_to_result or maps_messages or mcp_bypass" -v`
Expected: FAIL with `ImportError: cannot import name 'messages_to_result'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to orchestrators/claude_code.py
import json


def _is_result(msg) -> bool:
    return hasattr(msg, "result") and hasattr(msg, "subtype")


def _classify_block(b) -> str:
    if hasattr(b, "tool_use_id"):
        return "tool_result"
    if hasattr(b, "name") and hasattr(b, "input"):
        return "tool_use"
    if hasattr(b, "text"):
        return "text"
    return "other"


def _strip_mcp_prefix(name: str) -> Tuple[bool, str]:
    """Return (is_mcp, base_name). 'mcp__gym__tool' -> (True, 'tool')."""
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return True, parts[2]
    return False, name


def messages_to_result(messages: list, gym_hosts: List[str]) -> Tuple[dict, dict]:
    conversation_flow: List[dict] = []
    tools_used: List[str] = []
    builtin_used: List[str] = []
    tool_results: List[dict] = []
    tool_use_by_id: Dict[str, Tuple[str, dict, bool]] = {}  # id -> (base_name, args, is_mcp)
    final_response = ""
    usage = None
    cost = None
    subtype = None
    bypass = False

    for msg in messages:
        if _is_result(msg):
            final_response = getattr(msg, "result", "") or ""
            usage = getattr(msg, "usage", None)
            cost = getattr(msg, "total_cost_usd", None)
            subtype = getattr(msg, "subtype", None)
            continue

        for b in getattr(msg, "content", None) or []:
            kind = _classify_block(b)
            if kind == "text":
                conversation_flow.append({"type": "ai_message", "content": b.text})
            elif kind == "tool_use":
                is_mcp, base = _strip_mcp_prefix(b.name)
                args = b.input if isinstance(b.input, dict) else {}
                tool_use_by_id[getattr(b, "id", "")] = (base, args, is_mcp)
                if is_mcp:
                    if base not in tools_used:
                        tools_used.append(base)
                else:
                    if b.name not in builtin_used:
                        builtin_used.append(b.name)
                    if b.name == "Bash":
                        cmd = json.dumps(args)
                        if any(h and h in cmd for h in gym_hosts):
                            bypass = True
                conversation_flow.append(
                    {"type": "tool_call", "tool_name": base if is_mcp else b.name,
                     "arguments": args, "is_mcp": is_mcp}
                )
            elif kind == "tool_result":
                base, args, is_mcp = tool_use_by_id.get(
                    getattr(b, "tool_use_id", ""), (None, {}, False)
                )
                entry = {
                    "tool_name": base,
                    "arguments": args,
                    "result": {"result": getattr(b, "content", None)},
                    "gym_server": "mcp" if is_mcp else "builtin",
                }
                tool_results.append(entry)
                conversation_flow.append({"type": "tool_result", **entry})

    result = {
        "final_response": final_response,
        "conversation_flow": conversation_flow,
        "tools_used": tools_used,
        "tool_results": tool_results,
        "messages": [str(m) for m in messages],
    }
    metadata = {
        "cc_usage": usage,
        "cc_cost_usd": cost,
        "cc_subtype": subtype,
        "builtin_tools_used": builtin_used,
        "mcp_bypass_suspected": bypass,
    }
    return result, metadata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrators/test_claude_code.py -k "maps_messages or mcp_bypass" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orchestrators/claude_code.py tests/orchestrators/test_claude_code.py
git commit -m "feat(orchestrator): map Agent SDK messages into benchmark result schema"
```

---

### Task 6: `ClaudeCodeOrchestrator.execute()` wiring

Assembles the helpers into the orchestrator. The SDK is imported lazily and `query` is mocked in the test, so no network/SDK is needed to test the wiring.

**Files:**
- Modify: `orchestrators/claude_code.py`
- Test: `tests/orchestrators/test_claude_code.py`

**Interfaces:**
- Consumes: `compute_disallowed_tools`, `build_mcp_servers`, `build_cc_model_env`, `messages_to_result` (Tasks 2–5); base `AgentOrchestrator.__init__` args (`llm_client`, `mcp_clients`, `tool_to_server_mapping`, `available_tools`, `config`, `max_iterations`).
- Produces: `class ClaudeCodeOrchestrator(AgentOrchestrator)` with async `execute() -> dict` and `get_result_metadata() -> dict`. Constructor extra kwargs: `max_seconds: int = 1800`, `permission_mode: str = "bypassPermissions"`.
- The lazy import target is module attribute `orchestrators.claude_code.query` and `orchestrators.claude_code.ClaudeAgentOptions`, assigned inside `_load_sdk()` — the test monkeypatches `_load_sdk` to inject a fake `query` + options class.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/orchestrators/test_claude_code.py
import asyncio
import pytest
from orchestrators.claude_code import ClaudeCodeOrchestrator


class _Cfg:
    system_prompt = "You are an HR agent."
    user_prompt = "Approve PTO request 7."


class _LLM:
    provider = "vllm"
    model = "azure/gpt-5.5"
    custom_api_endpoint = "https://proxy.example"
    api_key = "sk-abc"
    region = None


def test_execute_builds_options_and_maps_result(monkeypatch):
    captured = {}

    class _FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    async def _fake_query(prompt=None, options=None):
        captured["prompt"] = prompt
        for m in _msgs():  # reuse the fake message list from Task 5's test
            yield m

    def _fake_loader():
        import orchestrators.claude_code as mod
        mod.query = _fake_query
        mod.ClaudeAgentOptions = _FakeOptions

    monkeypatch.setattr(ClaudeCodeOrchestrator, "_load_sdk", staticmethod(_fake_loader))

    orch = ClaudeCodeOrchestrator(
        llm_client=_LLM(),
        mcp_clients={"sn-hr-internal": _FakeClient("http://localhost:8008", "/mcp", "db_1", {})},
        tool_to_server_mapping={"approve_pto": "sn-hr-internal", "delete_x": "sn-hr-internal"},
        available_tools=[{"name": "approve_pto"}],
        config=_Cfg(),
        max_iterations=42,
    )
    result = asyncio.run(orch.execute())

    assert result["final_response"] == "done"
    assert result["tools_used"] == ["approve_pto"]
    assert captured["prompt"] == "Approve PTO request 7."
    assert captured["model"] == "azure/gpt-5.5"
    assert captured["system_prompt"] == "You are an HR agent."
    assert captured["max_turns"] == 42
    assert captured["permission_mode"] == "bypassPermissions"
    assert "mcp__sn-hr-internal__delete_x" in captured["disallowed_tools"]
    assert captured["mcp_servers"]["sn-hr-internal"]["headers"]["x-database-id"] == "db_1"
    assert orch.get_result_metadata()["cc_subtype"] == "success"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrators/test_claude_code.py -k execute_builds_options -v`
Expected: FAIL with `AttributeError`/`TypeError` (no `_load_sdk`, `execute` not implemented) or import error for the class members.

- [ ] **Step 3: Write minimal implementation**

```python
# add to orchestrators/claude_code.py
import asyncio
import logging

from .base import AgentOrchestrator

logger = logging.getLogger(__name__)

# Populated lazily by _load_sdk() so unit tests run without claude-agent-sdk installed.
query = None
ClaudeAgentOptions = None


class ClaudeCodeOrchestrator(AgentOrchestrator):
    def __init__(self, *args, max_seconds: int = 1800,
                 permission_mode: str = "bypassPermissions", **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seconds = max_seconds
        self.permission_mode = permission_mode
        self._metadata: Dict[str, Any] = {}

    @staticmethod
    def _load_sdk():
        """Import the Agent SDK into module globals (lazy, so pure helpers stay importable)."""
        global query, ClaudeAgentOptions
        from claude_agent_sdk import query as _q, ClaudeAgentOptions as _opts
        query = _q
        ClaudeAgentOptions = _opts

    def _gym_hosts(self) -> List[str]:
        hosts = []
        for client in self.mcp_clients.values():
            url = client.base_url
            hosts.append(url.split("://", 1)[-1].split("/", 1)[0])
        return hosts

    async def execute(self) -> Dict[str, Any]:
        self._load_sdk()

        model, env = build_cc_model_env(
            self.llm_client.provider, self.llm_client.model,
            getattr(self.llm_client, "custom_api_endpoint", None),
            getattr(self.llm_client, "api_key", None),
            getattr(self.llm_client, "region", None),
        )
        options = ClaudeAgentOptions(
            model=model,
            env=env,
            mcp_servers=build_mcp_servers(self.mcp_clients),
            disallowed_tools=compute_disallowed_tools(
                self.available_tools, self.tool_to_server_mapping
            ),
            system_prompt=self.config.system_prompt,
            permission_mode=self.permission_mode,
            max_turns=self.max_iterations,
        )

        messages: list = []
        subtype_override = None
        try:
            async with asyncio.timeout(self.max_seconds):
                async for msg in query(prompt=self.config.user_prompt, options=options):
                    messages.append(msg)
        except asyncio.TimeoutError:
            subtype_override = "timeout"
            logger.error("Claude Code session timed out after %ss", self.max_seconds)
        except Exception as e:  # SDK raises on max_turns after yielding ResultMessage
            logger.error("Claude Code session raised: %s", e)

        result, metadata = messages_to_result(messages, self._gym_hosts())
        if subtype_override:
            metadata["cc_subtype"] = subtype_override
        self._metadata = metadata
        return result

    def get_result_metadata(self) -> Dict[str, Any]:
        return self._metadata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrators/test_claude_code.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add orchestrators/claude_code.py tests/orchestrators/test_claude_code.py
git commit -m "feat(orchestrator): implement ClaudeCodeOrchestrator.execute via Agent SDK"
```

---

### Task 7: Register `claude_code` in the CLI + declare the dependency

**Files:**
- Modify: `evaluate.py` (imports, `ORCHESTRATOR_MAP`, `--orchestrator` choices)
- Modify: `pyproject.toml` (optional dependency)
- Test: `tests/orchestrators/test_registration.py`

**Interfaces:**
- Consumes: `ClaudeCodeOrchestrator` (Task 6).
- Produces: `ORCHESTRATOR_MAP["claude_code"] is ClaudeCodeOrchestrator`; CLI accepts `--orchestrator claude_code`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrators/test_registration.py
def test_claude_code_registered():
    import evaluate
    from orchestrators.claude_code import ClaudeCodeOrchestrator
    assert evaluate.ORCHESTRATOR_MAP["claude_code"] is ClaudeCodeOrchestrator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrators/test_registration.py -v`
Expected: FAIL with `KeyError: 'claude_code'`.

- [ ] **Step 3: Write minimal implementation**

In `evaluate.py`, add the import next to the other orchestrator imports (near line 37):

```python
from orchestrators.claude_code import ClaudeCodeOrchestrator
```

Add the map entry to `ORCHESTRATOR_MAP` (near line 40):

```python
ORCHESTRATOR_MAP = {
    "react": ReactOrchestrator,
    "planner_react": PlannerReactOrchestrator,
    "decomposing": DecomposingPlannerOrchestrator,
    "claude_code": ClaudeCodeOrchestrator,
}
```

Add `"claude_code"` to the `--orchestrator` `choices` list (near line 288):

```python
        choices=["react", "planner_react", "decomposing", "claude_code"],
```

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
claude_code = ["claude-agent-sdk"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrators/test_registration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py pyproject.toml tests/orchestrators/test_registration.py
git commit -m "feat(cli): register claude_code orchestrator and declare SDK dependency"
```

---

### Task 8: End-to-end single-task integration check (manual gate, no commit)

Depends on Task 1 having passed and an HR gym server running. Confirms the full path produces a scoreable result JSON.

**Prerequisites:** `uv sync --extra claude_code`; Node + Claude Code CLI on PATH; HR gym at `http://localhost:8008`; `conf/llm/gpt-5.5.json` present.

- [ ] **Step 1: Run one HR oracle task through claude_code**

Run a SINGLE task at a time via `--task_id` (or `--limit 1`) so you can debug the
arm cheaply before the full sweep:

```bash
python evaluate.py --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain hr --mode oracle \
    --task_id <one_hr_task_id> \
    --llm_config conf/llm/gpt-5.5.json \
    --orchestrator claude_code \
    --output_folder results/claude_code/gpt-5.5/hr \
    --concurrency 1 --num_runs 1
```
(`--limit 1` runs the first task if you don't have a specific id yet.)
Expected: a `results/claude_code/gpt-5.5/hr/run_1/results_*.json` file is written, with `runs[0].verification_results` populated and no top-level `error`.

- [ ] **Step 2: Score it and eyeball parity with react**

Run:
```bash
python compute_score.py --results_folder results/claude_code/gpt-5.5/hr
```
Expected: the scorer parses the file and prints a success rate without schema errors. Spot-check one result JSON: `tools_used` holds MCP base names; `builtin_tools_used` / `cc_subtype` / `mcp_bypass_suspected` appear in the run object; `verification_results` ran against the task DB.

- [ ] **Step 3: Report readiness**

If parity holds, the arm is ready for the full first-cut run (`--mode oracle plus_10`, `--concurrency 4`, N seeds). Report results; do not commit generated result files.

---

## Self-Review

**1. Spec coverage:**
- §3 arms/controls → Tasks 3 (per-task DB headers), 4 (same model via proxy), 6 (same prompts, built-ins on via `permission_mode`), 2 (oracle/plus_10 surface). ✓
- §3 confounders: capability leak → `builtin_tools_used` (Task 5); MCP-bypass → `mcp_bypass_suspected` (Task 5); proxy fidelity → spike (Task 1). ✓
- §4.3 model pinning / no Bedrock env for proxy → Task 4 + Global Constraints. ✓
- §4.4 tool restriction via `disallowed_tools` → Task 2. ✓
- §4.5 result mapping + tools_used normalization + metadata → Task 5. ✓
- §4.6 wiring + dependency → Task 7. ✓
- §6 spike → Task 1; §7 testing (unit + integration + parity) → Tasks 2–6 unit, Task 8 integration/parity. ✓
- §8 fallback (Claude model) → `build_cc_model_env` bedrock/anthropic branches (Task 4). ✓

**2. Placeholder scan:** No TBD/TODO; every code step has concrete code. The only intentionally environment-specific value is the HR seed file path in Task 1 (`gym_dbs/hr.sql`), flagged inline to adjust to the real path.

**3. Type consistency:** `compute_disallowed_tools(available_tools, tool_to_server_mapping)`, `build_mcp_servers(mcp_clients)`, `build_cc_model_env(provider, model, api_endpoint, api_key, region)`, `messages_to_result(messages, gym_hosts) -> (result, metadata)` are used with the same names/signatures in Task 6. Result keys (`final_response`/`conversation_flow`/`tools_used`/`tool_results`/`messages`) and metadata keys (`cc_usage`/`cc_cost_usd`/`cc_subtype`/`builtin_tools_used`/`mcp_bypass_suspected`) are consistent across Tasks 5–6 and the Global Constraints. The Task 6 test reuses `_msgs()` and `_FakeClient` defined in the Task 5/Task 3 tests (same file). ✓
