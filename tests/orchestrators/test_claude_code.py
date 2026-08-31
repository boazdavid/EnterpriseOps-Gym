import asyncio
import os

from orchestrators.claude_code import (
    build_cc_model_env,
    build_mcp_servers,
    compute_cache_aware_cost,
    compute_disallowed_tools,
    messages_to_result,
    ClaudeCodeOrchestrator,
)


# ---------------------------------------------------------------------------
# compute_disallowed_tools
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_mcp_servers
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, base_url, mcp_endpoint, database_id, context):
        self.base_url = base_url
        self.mcp_endpoint = mcp_endpoint
        self.database_id = database_id
        self.context = context


def test_build_mcp_servers_launches_stdio_shim_with_db_and_headers():
    import json as _json
    clients = {
        "sn-hr-internal": _FakeClient(
            "http://localhost:8008", "/mcp", "db_123", {"user_id": "u1"}
        )
    }
    servers = build_mcp_servers(clients)
    s = servers["sn-hr-internal"]
    assert s["type"] == "stdio"
    assert s["args"][0].endswith("gym_mcp_proxy.py")
    assert s["env"]["GYM_MCP_URL"] == "http://localhost:8008"
    assert s["env"]["GYM_MCP_ENDPOINT"] == "/mcp"
    assert s["env"]["GYM_DATABASE_ID"] == "db_123"
    assert _json.loads(s["env"]["GYM_EXTRA_HEADERS"]) == {"x-user-id": "u1"}


# ---------------------------------------------------------------------------
# build_cc_model_env
# ---------------------------------------------------------------------------


def test_proxy_provider_maps_to_anthropic_base_url_and_pins_model():
    model, env = build_cc_model_env(
        "vllm",
        "azure/gpt-5.5",
        "https://ete-litellm.ai-models.vpc-int.res.ibm.com",
        "sk-abc",
        None,
    )
    assert model == "azure/gpt-5.5"
    assert env["ANTHROPIC_BASE_URL"] == "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-abc"
    assert env["ANTHROPIC_MODEL"] == "azure/gpt-5.5"
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_proxy_provider_forwards_custom_headers():
    _, env = build_cc_model_env(
        "vllm", "gpt-5.5", "https://proxy/anthropic", "sk-abc", None,
        custom_headers="x-context-guru-token: cg_live_xyz",
    )
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-context-guru-token: cg_live_xyz"


def test_bedrock_provider_sets_bedrock_env():
    model, env = build_cc_model_env(
        "aws_bedrock", "us.anthropic.claude-sonnet-5", None, None, "us-east-1"
    )
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-east-1"
    assert env["ANTHROPIC_MODEL"] == "us.anthropic.claude-sonnet-5"


# ---------------------------------------------------------------------------
# messages_to_result
# ---------------------------------------------------------------------------


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
    result = _Obj(
        result="done",
        subtype="success",
        usage={"input_tokens": 10, "output_tokens": 5},
        total_cost_usd=0.02,
    )
    return [assistant, user, result]


def test_maps_messages_to_result_schema():
    result, meta = messages_to_result(_msgs(), gym_hosts=["localhost:8008"])
    assert result["final_response"] == "done"
    # MCP base name only, deduped; built-ins excluded from tools_used
    assert result["tools_used"] == ["approve_pto"]
    assert meta["builtin_tools_used"] == ["Bash"]
    assert meta["cc_subtype"] == "success"
    assert meta["cc_cost_usd_reported"] == 0.02
    assert meta["cc_usage"] == {"input_tokens": 10, "output_tokens": 5}
    # correlated tool_result carries the tool name + args
    assert result["tool_results"][0]["tool_name"] == "approve_pto"
    assert result["tool_results"][0]["arguments"] == {"id": 7}


def test_ai_message_carries_per_turn_usage_and_response_metadata():
    assistant = _Obj(
        content=[
            _Obj(text="calling tool"),
            _Obj(name="mcp__gym__do_it", input={"x": 1}, id="t1"),
        ],
        usage={"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 60},
        model="aws/claude-sonnet-5",
        stop_reason="tool_use",
    )
    result, _ = messages_to_result([assistant], gym_hosts=[])
    ai = [e for e in result["conversation_flow"] if e["type"] == "ai_message"][0]
    # full input incl. cached; cache split preserved
    assert ai["usage_metadata"]["input_tokens"] == 160
    assert ai["usage_metadata"]["input_token_details"]["cache_read"] == 60
    assert ai["usage_metadata"]["output_tokens"] == 40
    assert ai["response_metadata"] == {"model": "aws/claude-sonnet-5", "stop_reason": "tool_use"}
    assert ai["tool_calls"] == [{"name": "do_it", "args": {"x": 1}}]


def test_flags_mcp_bypass_when_bash_hits_gym_host():
    tool_use = _Obj(
        name="Bash", input={"command": "curl http://localhost:8008/api/query"}, id="b1"
    )
    assistant = _Obj(content=[tool_use])
    result = _Obj(result="x", subtype="success", usage={}, total_cost_usd=0.0)
    _, meta = messages_to_result([assistant, result], gym_hosts=["localhost:8008"])
    assert meta["mcp_bypass_suspected"] is True


# ---------------------------------------------------------------------------
# compute_cache_aware_cost
# ---------------------------------------------------------------------------


def test_cache_read_billed_at_cached_rate_not_full_input():
    rates = {"input": 1.25, "output": 10.0, "cache_read": 0.125, "cache_write": 1.5625}
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
    }
    info = compute_cache_aware_cost(usage, rates)
    # 1.25 (input) + 10.0 (output) + 0.125 (cache_read at cached rate, NOT 1.25) = 11.375
    assert round(info["cost_usd"], 4) == 11.375
    assert info["usage_totals"]["cache_read_input_tokens"] == 1_000_000
    assert info["usage_totals"]["total_tokens"] == 3_000_000


def test_sonnet_rates_match_proxy_price_map():
    from orchestrators.claude_code import price_rates
    r = price_rates("aws/claude-sonnet-5")
    assert r == {"input": 1.52, "output": 7.6, "cache_read": 0.152, "cache_write": 1.9}
    # 8 input + 13 output tokens -> matches x-litellm-response-cost-original
    info = compute_cache_aware_cost({"input_tokens": 8, "output_tokens": 13}, r)
    assert round(info["cost_usd"], 8) == 0.00011096


def test_cache_aware_cost_handles_missing_and_none_usage():
    rates = {"input": 1.25, "output": 10.0, "cache_read": 0.125, "cache_write": 1.5625}
    assert compute_cache_aware_cost(None, rates)["cost_usd"] == 0.0
    assert compute_cache_aware_cost({"output_tokens": 2_000_000}, rates)["cost_usd"] == 20.0


# ---------------------------------------------------------------------------
# ClaudeCodeOrchestrator.execute
# ---------------------------------------------------------------------------


class _Cfg:
    system_prompt = "You are an HR agent."
    user_prompt = "Approve PTO request 7."


class _LLM:
    provider = "vllm"
    model = "azure/gpt-5.5"
    custom_api_endpoint = "https://proxy.example"
    api_key = "sk-abc"
    region = None


def _mk_orch(**over):
    kw = dict(
        llm_client=_LLM(),
        mcp_clients={"g": _FakeClient("http://localhost:8008", "/mcp", "db_1", {})},
        tool_to_server_mapping={"a": "g"},
        available_tools=[{"name": "a"}],
        config=_Cfg(),
        max_iterations=5,
    )
    kw.update(over)
    return ClaudeCodeOrchestrator(**kw)


def test_memory_config_shares_dir_and_isolates_config(tmp_path):
    import json as _json
    shared = str(tmp_path / "exp_mem")
    orch = _mk_orch(auto_memory_dir=shared)
    settings, config_dir = orch._memory_config()
    assert config_dir == shared  # becomes CLAUDE_CONFIG_DIR (isolates from ~/.claude)
    s = _json.loads(settings)
    assert s["autoMemoryEnabled"] is True
    assert s["autoMemoryDirectory"] == os.path.join(shared, "memory")
    assert os.path.isdir(os.path.join(shared, "memory"))  # created


def test_memory_config_none_when_stateless(monkeypatch):
    monkeypatch.delenv("CC_AUTO_MEMORY_DIR", raising=False)
    assert _mk_orch()._memory_config() == (None, None)


def test_execute_builds_options_and_maps_result(monkeypatch):
    captured = {}

    class _FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    async def _fake_query(prompt=None, options=None):
        captured["prompt"] = prompt
        for m in _msgs():
            yield m

    def _fake_loader():
        import orchestrators.claude_code as mod

        mod.query = _fake_query
        mod.ClaudeAgentOptions = _FakeOptions

    monkeypatch.setattr(ClaudeCodeOrchestrator, "_load_sdk", staticmethod(_fake_loader))

    orch = ClaudeCodeOrchestrator(
        llm_client=_LLM(),
        mcp_clients={
            "sn-hr-internal": _FakeClient("http://localhost:8008", "/mcp", "db_1", {})
        },
        tool_to_server_mapping={
            "approve_pto": "sn-hr-internal",
            "delete_x": "sn-hr-internal",
        },
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
    # session runs in an isolated scratch cwd, not the benchmark repo
    assert captured["cwd"] and captured["cwd"] != "."
    assert "mcp__sn-hr-internal__delete_x" in captured["disallowed_tools"]
    assert captured["mcp_servers"]["sn-hr-internal"]["env"]["GYM_DATABASE_ID"] == "db_1"
    assert orch.get_result_metadata()["cc_subtype"] == "success"
    # stateless: isolated throwaway config dir (not the shared/continual one), no memory
    assert captured["env"]["CLAUDE_CONFIG_DIR"]  # a throwaway temp dir
    assert captured["settings"] is None
    assert "CONTINUAL MEMORY" not in captured["system_prompt"]


def test_execute_continual_memory_isolates_and_instructs(monkeypatch, tmp_path):
    captured = {}

    class _FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    async def _fake_query(prompt=None, options=None):
        for m in _msgs():
            yield m

    def _fake_loader():
        import orchestrators.claude_code as mod
        mod.query = _fake_query
        mod.ClaudeAgentOptions = _FakeOptions

    monkeypatch.setattr(ClaudeCodeOrchestrator, "_load_sdk", staticmethod(_fake_loader))
    shared = str(tmp_path / "exp_home")
    orch = _mk_orch(auto_memory_dir=shared)
    asyncio.run(orch.execute())

    assert captured["env"]["CLAUDE_CONFIG_DIR"] == shared          # isolated from ~/.claude
    assert "CONTINUAL MEMORY" in captured["system_prompt"]          # instruction appended
    assert os.path.isfile(os.path.join(shared, "CLAUDE.md"))        # memory file seeded
    import json as _json
    assert _json.loads(captured["settings"])["autoMemoryDirectory"] == os.path.join(shared, "memory")


# ---------------------------------------------------------------------------
# Arm 2a / 3a — parallel hint (CC_PARALLEL_HINT) + procedural memory (CC_MEMORY_MODE)
# ---------------------------------------------------------------------------


def _run_capture(monkeypatch, orch):
    captured = {}

    class _FO:
        def __init__(self, **kw):
            captured.update(kw)

    async def _fq(prompt=None, options=None):
        for m in _msgs():
            yield m

    def _loader():
        import orchestrators.claude_code as mod
        mod.query = _fq
        mod.ClaudeAgentOptions = _FO

    monkeypatch.setattr(ClaudeCodeOrchestrator, "_load_sdk", staticmethod(_loader))
    asyncio.run(orch.execute())
    return captured


def test_procedural_memory_mode_uses_procedural_instruction_and_seed(monkeypatch, tmp_path):
    monkeypatch.delenv("CC_PARALLEL_HINT", raising=False)
    monkeypatch.setenv("CC_MEMORY_MODE", "procedural")
    shared = str(tmp_path / "procmem")
    cap = _run_capture(monkeypatch, _mk_orch(auto_memory_dir=shared))
    assert "ONLY reusable PROCEDURES" in cap["system_prompt"]
    # not the fact-based Arm-2 instruction
    assert "do not store task-specific secrets" not in cap["system_prompt"]
    assert "no task-specific" in open(os.path.join(shared, "CLAUDE.md")).read()
