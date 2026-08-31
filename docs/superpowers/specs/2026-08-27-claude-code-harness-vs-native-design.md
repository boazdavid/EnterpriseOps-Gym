# Claude Code harness vs. native ReAct on EnterpriseOps-Gym — design

**Date:** 2026-08-27
**Status:** Draft for review

## 1. Question

Holding the *model* fixed, how much does the **agent harness** change task success on
EnterpriseOps-Gym? We compare the benchmark's native ReAct orchestrator against the
**Claude Code** harness (Claude Agent SDK), with both arms driving the **same GPT-5.5
deployment** so that the only variable is the scaffolding — the agent loop, its context
management, and its built-in tools.

## 2. How the benchmark works (load-bearing facts)

These properties of the existing code are what make the comparison clean:

- **Per-task database via HTTP header.** Each run creates a fresh DB
  (`create_database_from_file`) and every MCP tool call carries an `x-database-id` header
  plus `x-*` context headers (`benchmark/mcp_client.py`). State is keyed entirely on that
  header.
- **Verification is decoupled from the agent.** `BenchmarkExecutor._run_verifiers`
  (`benchmark/executor.py`) runs SQL `database_state` checks against the final DB state for
  a given `database_id`, regardless of *how* the agent reached that state. The same
  verifiers score any harness.
- **The orchestrator is pluggable.** `AgentOrchestrator` (`orchestrators/base.py`) is
  handed `llm_client`, `mcp_clients` (already carrying the per-task `database_id`, URL,
  endpoint, and context), `available_tools`, `config`, and `max_iterations`; it must return
  `{final_response, conversation_flow, tools_used, tool_results, messages}` and may add
  telemetry via `get_result_metadata()`. `evaluate.py` maps `--orchestrator` names to
  classes in `ORCHESTRATOR_MAP`.
- **Gym servers are streamable-HTTP MCP servers.** For HR the single gym is
  `sn-hr-internal` at `http://localhost:8008` (no secondary knowledge server).

**Consequence:** we can add a Claude Code orchestrator that operates on the *same* per-task
DB (same `x-database-id`) and let the *existing, unchanged* verifiers score it. Nothing in
`executor.py`, Ray, `compute_score.py`, or `scan_results.py` needs to change.

## 3. Experiment design

### Arms (independent variable = harness only)
- **A — Native ReAct** (baseline): existing `--orchestrator react`, GPT-5.5.
- **B — Claude Code** (`--orchestrator claude_code`): new orchestrator driving the Claude
  Agent SDK, GPT-5.5 **through an Anthropic-compatible proxy**, with MCP tools **plus**
  Claude Code's built-in tools ("as shipped").

### Held constant across both arms
- **Model + deployment:** GPT-5.5 on the same LiteLLM proxy
  (`https://ete-litellm.ai-models.vpc-int.res.ibm.com`, model `azure/gpt-5.5`). Arm A via
  the OpenAI-compatible endpoint (`conf/llm/gpt-5.5.json`, provider `vllm`); Arm B via the
  proxy's Anthropic `/v1/messages` endpoint (`ANTHROPIC_BASE_URL`).
- **Per-task DB:** same fresh DB and `x-database-id` in both arms.
- **Prompts:** same `system_prompt` + `user_prompt` from each task config.
- **MCP tool surface per mode:** `oracle` and `plus_10` replicated in both arms (see §4.4).
- **Task set / domain / seeds / verifiers:** HR domain, same task set, same `--num_runs`,
  same SQL verifiers.

### Deliberately different
Only the harness: native ReAct loop vs. Claude Code's loop + always-on built-ins
(Bash/Read/Write/Edit/Glob/Grep/WebSearch/WebFetch).

### Scope (first cut)
- **Domain:** HR.
- **Model:** GPT-5.5.
- **Modes:** `oracle`, `plus_10`.
- **Runs:** N seeds (default per existing convention; e.g. 3), seed-major ordering.

### Metrics (reuse existing scoring)
Overall task success (all verifiers pass), verifier-level pass rate, pass@1, mean
turns/tools, per-arm deltas per mode — via `compute_score.py` / `scan_results.py` on the
unchanged results schema. Cost/turns are secondary descriptive stats (the loops differ);
success is the headline. Token cost accounted cache-aware (bill `cache_read` at the cached
rate).

### Confounders (documented; partly mitigated)
1. **Capability leak from built-ins.** Arm B has Bash/Read/Write etc.; Arm A does not. So a
   B>A delta is "better loop *and* more tools," not loop alone. Mitigation: record
   `builtin_tools_used` per task so we can quantify how often built-ins were actually used;
   an optional **Arm B′ (MCP-only)** is noted in §8 to isolate loop-effect from tool-effect.
2. **MCP-bypass integrity risk.** With Bash on, B could `curl` the gym HTTP API directly and
   mutate the DB outside MCP, or hit the verifier's query path. Mitigations: (a) run each
   session in a scratch `cwd` and pass the gym URL only through the SDK's programmatic
   `mcp_servers` (never in the prompt, never as a `.mcp.json` in cwd); (b) post-hoc flag any
   task whose transcript shows a Bash call referencing the gym host
   (`mcp_bypass_suspected`). Hard network isolation is possible but out of scope for v1.
3. **Proxy fidelity.** Claude Code emits Claude-shaped requests (system-prompt injection,
   tool schemas, possibly `cache_control`/thinking). Driving GPT-5.5 through an
   Anthropic→OpenAI translation (LiteLLM) can be flakier than native. Mitigations: full
   plain-string system-prompt override (no Claude Code preset dynamics), thinking left off,
   and a mandatory proxy spike before any real runs (§6).

## 4. Architecture

### 4.1 New component
`orchestrators/claude_code.py` → `ClaudeCodeOrchestrator(AgentOrchestrator)`. It implements
`execute()` by driving the Claude Agent SDK (`claude_agent_sdk.query`) instead of the
LLM↔tool loop. It reuses everything the base class already receives; **no change to the
executor's flow**.

### 4.2 Data flow (per task)
1. Executor creates the fresh DB and sets `database_id` on each `MCPClient` (unchanged).
2. Executor constructs `ClaudeCodeOrchestrator` with the usual kwargs (unchanged call site).
3. Orchestrator builds `ClaudeAgentOptions`:
   - `mcp_servers`: one HTTP entry per gym from `mcp_clients` — `{"type":"http", "url":
     base_url+mcp_endpoint, "headers": {"x-database-id": db_id, **context_headers}}`.
   - `model` + `env`: GPT-5.5-via-proxy config (see §4.3).
   - `system_prompt`: full-string override = `config.system_prompt`.
   - tool controls: see §4.4.
   - `permission_mode="bypassPermissions"` (zero prompts; auto-approves MCP + built-ins —
     the honest "as shipped" condition).
   - `max_turns` = `self.max_iterations` (50).
4. Orchestrator runs `query(prompt=config.user_prompt, options=...)` inside
   `asyncio.timeout(...)`, draining the async iterator fully (never `break` early).
5. Orchestrator maps the streamed messages into the result dict (§4.5) and returns.
6. Executor runs the **existing** SQL verifiers against the same `database_id` (unchanged).

### 4.3 Model / proxy configuration
Derived from the run's `LLMConfig` (the orchestrator can read `llm_client` /
`llm_config`). For GPT-5.5-via-proxy:
- `ANTHROPIC_BASE_URL = <llm_api_endpoint>` (the LiteLLM proxy).
- `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) = `<llm_api_key>`.
- `ANTHROPIC_MODEL = azure/gpt-5.5` and `options.model = azure/gpt-5.5` — **pinned
  explicitly** (an unpinned model silently defaults to Opus 5 → wrong model + cost).
- **Do not** set `CLAUDE_CODE_USE_BEDROCK`.
- Passed via the per-session `env=` option (merged over process env) so concurrent sessions
  don't race on global `os.environ`.

A small mapping function translates an `LLMConfig` into `(model, env)` for the SDK. For a
Claude model target (the §8 alternative) it would instead set Bedrock/Anthropic env; for v1
we implement the proxy path.

### 4.4 Tool surface (reproducing oracle / plus_10)
- MCP tool name format is `mcp__<server>__<tool>`; `mcp__<server>__*` is a wildcard.
- `allowed_tools` only *auto-approves*; it does **not** hide tools. To actually restrict the
  MCP surface to a task's `selected_tools` (oracle) or its plus_10 set, we **`disallowed_tools`**
  the complement: `disallowed_tools = {mcp__<server>__<name> for every server tool not in
  selected_tools}`. The full per-server tool list is obtained via the existing
  `MCPClient.list_tools()`.
- Built-ins are **left enabled** (not disallowed) per the "as shipped" choice.
- `plus_10` set is taken from the task config exactly as the native arm receives it (the
  dataset's `mode` split already encodes the distractor set in `selected_tools`).

### 4.5 Result mapping (Agent SDK → benchmark schema)
Drain messages; map:
- `AssistantMessage.content` → `TextBlock` into `conversation_flow` ai_messages;
  `ToolUseBlock(name,input,id)` → `tool_calls` and appended to `tools_used`.
- `UserMessage.content` → `ToolResultBlock(tool_use_id,content)` → `tool_results` +
  `conversation_flow` tool_result entries.
- `ResultMessage` → `final_response` = `.result`; telemetry: `.usage` (incl. cache fields),
  `.total_cost_usd`, `.subtype` (e.g. `error_max_turns`).
- **`tools_used` normalization:** strip the `mcp__<server>__` prefix from MCP tool names so
  MCP tool stats stay comparable to Arm A; record built-in tool usage separately in
  `builtin_tools_used` (surfaced via `get_result_metadata()`, not mixed into domain-tool
  counts).
- `tool_results` mapping is best-effort (for logging/analysis only) — verification does
  **not** depend on it.
- `get_result_metadata()` returns: `cc_usage`, `cc_cost_usd`, `cc_subtype`,
  `builtin_tools_used`, `mcp_bypass_suspected`.

### 4.6 Wiring
- Register `"claude_code": ClaudeCodeOrchestrator` in `ORCHESTRATOR_MAP` and add it to the
  `--orchestrator` choices in `evaluate.py`.
- Add `claude-agent-sdk` as an optional extra in `pyproject.toml` (e.g. `[project.optional-
  dependencies] claude_code = ["claude-agent-sdk"]`).
- No changes to `executor.py`, `verifier.py`, Ray configs, `compute_score.py`,
  `scan_results.py`.

### 4.7 Run command
```bash
python evaluate.py --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain hr --mode oracle plus_10 \
    --llm_config conf/llm/gpt-5.5.json \
    --orchestrator claude_code \
    --output_folder results/claude_code/gpt-5.5/hr \
    --concurrency 4 --num_runs 3
```
Native baseline is the same command with `--orchestrator react` and a `results/react/...`
output folder.

**Running one task at a time.** `evaluate.py` accepts `--task_id <id> [<id> ...]` (match
against the task config basename) and `--limit N` to run a single task or a small batch —
useful for debugging the Claude Code arm and controlling cost before the full sweep. The
existing `skip_sample` logic means re-running a folder resumes only the not-yet-completed
tasks.

## 5. Prerequisites
- **Node + Claude Code CLI** installed and on PATH (the Agent SDK spawns the CLI as a
  subprocess per session).
- `claude-agent-sdk` Python package installed.
- LiteLLM proxy reachable and — critically — exposing a working Anthropic `/v1/messages`
  route that maps to `azure/gpt-5.5` with tool-calling translated.

## 6. Opening spike (do first, before any orchestrator code)
Verify the proxy speaks Anthropic Messages with tools: a minimal `claude_agent_sdk.query`
against `ANTHROPIC_BASE_URL=<proxy>`, `model=azure/gpt-5.5`, one trivial HTTP MCP tool
(pointed at a throwaway HR DB), asserting we get a `ToolUseBlock` and a `ResultMessage`. If
`/v1/messages` isn't exposed or tool translation fails, resolve that (enable the route, or
fall back to the §8 Claude-model comparison) before building further.

## 7. Testing strategy
- **Unit (no network):** the `LLMConfig → (model, env)` mapper; the tool-surface computation
  (`selected_tools` → `disallowed_tools`); the message→result-schema mapper against a
  recorded/faked Agent SDK message list (assert `conversation_flow`, `tools_used`
  normalization, `builtin_tools_used`, `mcp_bypass_suspected`).
- **Integration (1 task):** run a single HR/oracle task end-to-end through
  `claude_code`; assert the result JSON matches the schema the verifiers/scoring expect and
  that verifiers execute against the task DB.
- **Parity check:** run the same single task through `react` and `claude_code`; confirm both
  produce scoreable outputs and the scorer treats them identically.

## 8. Out of scope / future
- **Arm B′ (MCP-only Claude Code)** to separate loop-effect from built-in-tool-effect
  (disallow built-ins).
- **Same-model-with-a-Claude-model** comparison (cleanest harness isolation, no proxy
  fragility) as a separate run.
- Additional domains (incl. multi-server `hybrid`) and additional modes
  (`plus_5`/`plus_15`).
- Hard network isolation of the Claude Code arm.

## 9. Risks
- **Proxy fidelity** (see §3.3, §6) — the highest-risk item; gated by the spike.
- **Subprocess pressure** — each session is a Node subprocess; keep concurrency modest
  (mirror the repo's low-concurrency convention where needed).
- **MCP first-turn connect blocks** up to `MCP_TIMEOUT` (30s default); size proxy/gym
  capacity for the chosen concurrency.

## 10. Open questions
- Exact `--num_runs` / seed count for the first cut (assumed 3).
- Whether to pin a specific Claude Code CLI / `claude-agent-sdk` version for
  reproducibility (recommended).
