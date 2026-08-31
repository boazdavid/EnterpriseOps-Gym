"""Claude Code (Claude Agent SDK) orchestrator for EnterpriseOps-Gym.

Drives the Claude Agent SDK against the same per-task gym database as the native
ReAct arm, so both arms run the same model and only the harness differs. The SDK
is imported lazily inside ``execute()`` so the pure helper functions (and their
unit tests) work without ``claude-agent-sdk`` installed.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

from .base import AgentOrchestrator

logger = logging.getLogger(__name__)

# Appended to the task system prompt only when continual shared memory is enabled.
MEMORY_INSTRUCTION = (
    "\n\n---\nCONTINUAL MEMORY: You have a persistent memory file (your Claude config "
    "CLAUDE.md) shared across tasks in this environment. At the start, consult it for "
    "reusable facts/procedures. When you learn something durable and reusable (a working "
    "approach, an entity id, a gotcha), append a concise note to it with the Write/Edit "
    "tools. Keep it short and general — do not store task-specific secrets."
)

# Procedural-only memory (Arm 2a): generalizable how-to, NOT task-specific facts/IDs.
PROCEDURAL_MEMORY_INSTRUCTION = (
    "\n\n---\nCONTINUAL MEMORY: You have a persistent memory file (your Claude config "
    "CLAUDE.md) shared across tasks in this environment. At the start, consult it. Record "
    "ONLY reusable PROCEDURES — generalizable how-to steps, tool-usage patterns, and gotchas "
    "that apply to a class of tasks. Do NOT store task-specific facts, entity IDs, names, or "
    "data values (they don't generalize). Keep entries short and procedural."
)

# Populated lazily by ClaudeCodeOrchestrator._load_sdk() so the pure helpers below
# stay importable without the SDK installed.
query = None
ClaudeAgentOptions = None


# ============================================================================
# TOOL SURFACE
# ============================================================================


def compute_disallowed_tools(
    available_tools: List[Dict[str, Any]], tool_to_server_mapping: Dict[str, str]
) -> List[str]:
    """MCP tools to hide so the surface matches the task's mode (oracle/plus_N).

    ``available_tools`` is the executor-filtered subset for this mode;
    ``tool_to_server_mapping`` holds every tool the servers advertised. We disallow
    the complement so Claude Code sees only the selected MCP tools.
    """
    selected = {t.get("name") for t in available_tools}
    disallowed = [
        f"mcp__{server}__{name}"
        for name, server in tool_to_server_mapping.items()
        if name not in selected
    ]
    return sorted(disallowed)


# ============================================================================
# MCP SERVER CONFIG
# ============================================================================


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


_PROXY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_mcp_proxy.py")


def build_mcp_servers(mcp_clients: Dict[str, Any]) -> Dict[str, dict]:
    """One **stdio** MCP server per gym, launching the gym_mcp_proxy shim.

    The gym is a POST-only JSON-RPC endpoint that Claude Code's Streamable-HTTP client
    rejects; the shim speaks compliant stdio to Claude Code and proxies to the gym over
    POST, injecting the per-task x-database-id + context headers. Keyed by gym name so
    tool names are mcp__<gym_name>__<tool> (consistent with compute_disallowed_tools).
    """
    import sys

    servers = {}
    for gym_name, client in mcp_clients.items():
        servers[gym_name] = {
            "type": "stdio",
            "command": sys.executable,
            "args": [_PROXY_PATH],
            "env": {
                "GYM_MCP_URL": client.base_url.rstrip("/"),
                "GYM_MCP_ENDPOINT": client.mcp_endpoint,
                "GYM_DATABASE_ID": client.database_id or "",
                "GYM_EXTRA_HEADERS": json.dumps(_context_headers(getattr(client, "context", {}))),
            },
        }
    return servers


# ============================================================================
# MODEL / ENV MAPPING
# ============================================================================


def build_cc_model_env(
    provider: str, model: str, api_endpoint, api_key, region, custom_headers=None
) -> Tuple[str, Dict[str, str]]:
    """Translate benchmark LLM settings into Claude Agent SDK (model, env).

    ``custom_headers`` (proxy path only) is an optional value for the
    ``ANTHROPIC_CUSTOM_HEADERS`` env var — some Anthropic-compatible proxies require
    an extra header (e.g. a gateway token) in addition to the provider credential.
    """
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
    if custom_headers:
        env["ANTHROPIC_CUSTOM_HEADERS"] = custom_headers
    return model, env


# ============================================================================
# MESSAGE -> RESULT MAPPING
# ============================================================================


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


def _as_usage_dict(u) -> dict:
    """Coerce an SDK usage value (dict or object) into a plain dict."""
    if u is None:
        return {}
    if isinstance(u, dict):
        return u
    if hasattr(u, "__dict__"):
        return dict(vars(u))
    return {}


def normalize_usage(usage) -> dict:
    """Normalize an Anthropic-style usage into a trajectory-friendly usage_metadata.

    ``input_tokens`` is the FULL input incl. cached (matching LangChain's convention so
    it reads naturally alongside the ReAct arm); the cache split is under
    ``input_token_details``, and raw Anthropic buckets are preserved for cost accounting.
    """
    u = _as_usage_dict(usage)
    cache_read = u.get("cache_read_input_tokens", 0) or 0
    cache_creation = u.get("cache_creation_input_tokens", 0) or 0
    in_uncached = u.get("input_tokens", 0) or 0
    in_total = in_uncached + cache_read + cache_creation
    out_total = u.get("output_tokens", 0) or 0
    return {
        "input_tokens": in_total,
        "output_tokens": out_total,
        "total_tokens": in_total + out_total,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        "raw": {
            "input_tokens": in_uncached,
            "output_tokens": out_total,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }


def messages_to_result(messages: list, gym_hosts: List[str]) -> Tuple[dict, dict]:
    """Convert the SDK's streamed message list into the benchmark result dict + telemetry."""
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

        # Group one conversation_flow "ai_message" per assistant message, carrying its
        # text, tool_calls, per-turn usage_metadata and response_metadata — so the saved
        # trajectory holds token/cost info per turn (mirrors the ReAct arm).
        texts: List[str] = []
        calls: List[dict] = []
        for b in getattr(msg, "content", None) or []:
            kind = _classify_block(b)
            if kind == "text":
                texts.append(b.text)
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
                calls.append(
                    {"name": base if is_mcp else b.name, "args": args, "is_mcp": is_mcp}
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

        if texts or calls:
            ai_entry = {
                "type": "ai_message",
                "content": "\n".join(texts),
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
            }
            msg_usage = getattr(msg, "usage", None)
            if msg_usage is not None:
                ai_entry["usage_metadata"] = normalize_usage(msg_usage)
            resp_meta = {}
            if getattr(msg, "model", None) is not None:
                resp_meta["model"] = msg.model
            if getattr(msg, "stop_reason", None) is not None:
                resp_meta["stop_reason"] = msg.stop_reason
            if resp_meta:
                ai_entry["response_metadata"] = resp_meta
            conversation_flow.append(ai_entry)

    result = {
        "final_response": final_response,
        "conversation_flow": conversation_flow,
        "tools_used": tools_used,
        "tool_results": tool_results,
        "messages": [str(m) for m in messages],
    }
    metadata = {
        "cc_usage": usage,
        "cc_cost_usd_reported": cost,  # Claude Code's own estimate — mispriced (Opus rates) for unrecognized models
        "cc_subtype": subtype,
        "builtin_tools_used": builtin_used,
        "mcp_bypass_suspected": bypass,
    }
    return result, metadata


# ============================================================================
# COST (CACHE-AWARE)
# ============================================================================

# Per-1M-token list prices, chosen by model-id substring. Where cache_read/cache_write
# are omitted they default to 0.1x / 1.25x input. Env vars override any field. Sonnet-5
# rates are the proxy's actual Bedrock price map (us.anthropic.claude-sonnet-5).
DEFAULT_PRICE_INPUT_PER_1M = 1.25
DEFAULT_PRICE_OUTPUT_PER_1M = 10.0
MODEL_PRICE_TABLE = [
    ("sonnet", {"input": 1.52, "output": 7.6, "cache_read": 0.152, "cache_write": 1.9}),
    ("opus", {"input": 5.0, "output": 25.0}),
    ("haiku", {"input": 1.0, "output": 5.0}),
    ("gpt-5", {"input": 1.25, "output": 10.0}),  # gpt-5 / gpt-5.5 (approximate)
]


def default_rates_for_model(model) -> Dict[str, float]:
    m = (model or "").lower()
    for key, rates in MODEL_PRICE_TABLE:
        if key in m:
            return dict(rates)
    return {"input": DEFAULT_PRICE_INPUT_PER_1M, "output": DEFAULT_PRICE_OUTPUT_PER_1M}


def price_rates(model=None) -> Dict[str, float]:
    """Cost rates (USD per 1M tokens). Auto-selected by model id; env vars override:
    CC_PRICE_{INPUT,OUTPUT,CACHE_READ,CACHE_WRITE}_PER_1M."""
    base = default_rates_for_model(model)
    inp = float(os.environ.get("CC_PRICE_INPUT_PER_1M", base["input"]))
    out = float(os.environ.get("CC_PRICE_OUTPUT_PER_1M", base["output"]))
    cache_read = float(
        os.environ.get("CC_PRICE_CACHE_READ_PER_1M", base.get("cache_read", inp * 0.1))
    )
    cache_write = float(
        os.environ.get("CC_PRICE_CACHE_WRITE_PER_1M", base.get("cache_write", inp * 1.25))
    )
    return {"input": inp, "output": out, "cache_read": cache_read, "cache_write": cache_write}


def compute_cache_aware_cost(usage: dict, rates: Dict[str, float]) -> Dict[str, Any]:
    """Cache-aware USD cost from an Anthropic-style usage object.

    Bills the (uncached) ``input_tokens`` at the input rate, ``output_tokens`` at the
    output rate, and — crucially — ``cache_read_input_tokens`` at the discounted cached
    rate rather than the full input rate (list price overstates agent-loop cost ~10x
    when cache reads are billed as fresh input).
    """
    u = usage or {}

    def g(k):
        return u.get(k, 0) or 0

    inp, out = g("input_tokens"), g("output_tokens")
    cache_read, cache_write = g("cache_read_input_tokens"), g("cache_creation_input_tokens")
    cost = (
        inp * rates["input"]
        + out * rates["output"]
        + cache_read * rates["cache_read"]
        + cache_write * rates["cache_write"]
    ) / 1_000_000
    return {
        "cost_usd": cost,
        "cost_rates_per_1m": rates,
        "usage_totals": {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
            "total_tokens": inp + out + cache_read + cache_write,
        },
    }


# ============================================================================
# ORCHESTRATOR
# ============================================================================


class ClaudeCodeOrchestrator(AgentOrchestrator):
    def __init__(
        self,
        *args,
        max_seconds: int = 1800,
        permission_mode: str = "bypassPermissions",
        auto_memory_dir: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_seconds = max_seconds
        self.permission_mode = permission_mode
        # Opt-in continual memory: when set, every task session shares this directory as
        # its Claude config home (CLAUDE_CONFIG_DIR), so CLAUDE.md AND auto-memory both
        # accumulate here across tasks and stay isolated from the real ~/.claude. Decoupled
        # from the per-task scratch cwd, so repo isolation is preserved. None => stateless.
        self.auto_memory_dir = auto_memory_dir or os.environ.get("CC_AUTO_MEMORY_DIR")
        self._metadata: Dict[str, Any] = {}

    def _memory_config(self) -> Tuple[Any, Any]:
        """Return (settings_json, config_dir) for shared continual memory, or (None, None).

        config_dir becomes CLAUDE_CONFIG_DIR (isolates & shares all Claude Code state
        away from the real ~/.claude); auto-memory is pinned to <config_dir>/memory. The
        reliable, headless-safe memory affordance is a seeded <config_dir>/CLAUDE.md the
        model edits (see _seed_memory_file + MEMORY_INSTRUCTION).
        """
        if not self.auto_memory_dir:
            return None, None
        config_dir = self.auto_memory_dir
        mem_dir = os.path.join(config_dir, "memory")
        os.makedirs(mem_dir, exist_ok=True)
        settings = json.dumps(
            {"autoMemoryEnabled": True, "autoMemoryDirectory": mem_dir}
        )
        return settings, config_dir

    @staticmethod
    def _seed_memory_file(config_dir: str, procedural: bool = False) -> str:
        """Create the shared CLAUDE.md memory file if absent; return its path."""
        path = os.path.join(config_dir, "CLAUDE.md")
        if not os.path.exists(path):
            body = (
                "Reusable, generalizable PROCEDURES and tool-usage patterns learned while "
                "solving earlier tasks. Append concise procedures here — no task-specific "
                "facts or IDs.\n"
                if procedural else
                "Durable, reusable facts and procedures learned while solving earlier "
                "tasks in this environment. Append concise new learnings here.\n"
            )
            with open(path, "w") as f:
                f.write("# Agent Memory\n\n" + body)
        return path

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
            self.llm_client.provider,
            self.llm_client.model,
            getattr(self.llm_client, "custom_api_endpoint", None),
            getattr(self.llm_client, "api_key", None),
            getattr(self.llm_client, "region", None),
            custom_headers=os.environ.get("ANTHROPIC_CUSTOM_HEADERS"),
        )
        procedural = os.environ.get("CC_MEMORY_MODE", "").lower() == "procedural"
        settings, config_dir = self._memory_config()
        system_prompt = self.config.system_prompt
        stateless_cfg = None
        if config_dir:
            # Continual: shared, isolated config dir + seeded memory + instruction.
            env["CLAUDE_CONFIG_DIR"] = config_dir
            self._seed_memory_file(config_dir, procedural=procedural)
            system_prompt = system_prompt + (
                PROCEDURAL_MEMORY_INSTRUCTION if procedural else MEMORY_INSTRUCTION
            )
        else:
            # Stateless: a fresh throwaway config dir per task — no memory, isolated from
            # the real ~/.claude (also excludes the user's plugins/CLAUDE.md for a clean,
            # reproducible tool surface). Deleted in finally.
            stateless_cfg = tempfile.mkdtemp(prefix="cc_cfg_")
            env["CLAUDE_CONFIG_DIR"] = stateless_cfg
        # Isolate the session in an empty scratch dir so Claude Code's built-in
        # filesystem tools (Bash/Read/Glob/...) cannot read the benchmark repo — the
        # seed SQL files are ground-truth answer keys and other tasks/domains live
        # there too. Without this the agent explores the repo instead of the gym.
        scratch_dir = tempfile.mkdtemp(prefix="cc_task_")
        options = ClaudeAgentOptions(
            model=model,
            env=env,
            cwd=scratch_dir,
            settings=settings,
            mcp_servers=build_mcp_servers(self.mcp_clients),
            disallowed_tools=compute_disallowed_tools(
                self.available_tools, self.tool_to_server_mapping
            ),
            system_prompt=system_prompt,
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
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            if stateless_cfg:
                shutil.rmtree(stateless_cfg, ignore_errors=True)

        result, metadata = messages_to_result(messages, self._gym_hosts())
        # Prepend the prompts so the saved trajectory is self-contained (mirrors ReAct).
        result["conversation_flow"] = [
            {"type": "system_message", "content": self.config.system_prompt},
            {"type": "user_message", "content": self.config.user_prompt},
        ] + result["conversation_flow"]
        if subtype_override:
            metadata["cc_subtype"] = subtype_override
        # Authoritative cache-aware cost (Claude Code's own cc_cost_usd_reported is
        # mispriced for non-Claude / unrecognized models).
        metadata.update(
            compute_cache_aware_cost(
                metadata.get("cc_usage"), price_rates(self.llm_client.model)
            )
        )
        self._metadata = metadata
        return result

    def get_result_metadata(self) -> Dict[str, Any]:
        return self._metadata
