# Running the Claude Code vs. native harness comparison

How to run EnterpriseOps-Gym with the **Claude Code** harness (`--orchestrator
claude_code`) and compare it against the native **ReAct** arm on the *same* model,
tasks, and databases. Design rationale lives in
`docs/superpowers/specs/2026-08-27-claude-code-harness-vs-native-design.md`; the build
plan in `docs/superpowers/plans/2026-08-27-claude-code-harness-orchestrator.md`.

Both arms hit the same per-task database (via the `x-database-id` header) and are
scored by the same SQL verifiers — only the agent harness differs.

## 1. Prerequisites

- **Gym server** for the domain running locally. HR: `sn-hr-internal` on
  `http://localhost:8008` (see README §Gym Servers for the docker image). Check:
  `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8008/` → `200`.
- **Seed databases** available on disk. Task configs reference seeds under
  `Domain Wise DBs and Task-DB Mappings/<domain>/dbs/…`; run `unzip gym_dbs.zip` if not
  already extracted.
- **Node + Claude Code CLI** on PATH, and the SDK: `uv pip install claude-agent-sdk`
  (or `uv sync --extra claude_code`). The SDK spawns the Claude Code CLI as a
  subprocess per task.
- **LiteLLM/contextguru proxy** reachable, exposing an Anthropic `/v1/messages` route.
  Verified: `https://contextguru.vpc.cloud9.ibm.com/anthropic/v1/messages` serves both
  `gpt-5.5` and `aws/claude-sonnet-5` with correct Anthropic tool-call translation.

## 2. Credentials

Credentials live in `try_wikis/.env` (do not commit; do not echo values):

- `OPENAI_API_KEY` — the proxy provider key (sent as the Anthropic auth token).
- `ANTHROPIC_CUSTOM_HEADERS` — `x-context-guru-token: <token>`. **Required in addition
  to** the provider key: the `/anthropic` route returns 401 if either is missing.

The Claude Code arm reads `ANTHROPIC_CUSTOM_HEADERS` from the environment and forwards
it. Export it before running (see commands below). The provider key + endpoint come
from the LLM config (§3).

> Note: `cd`-ing into `try_wikis` activates a direnv that breaks PATH — read the `.env`
> by absolute path; don't `cd` there.

## 3. LLM configs (Claude Code arm)

The Claude Code arm uses the config's `llm_api_endpoint` as `ANTHROPIC_BASE_URL` and
`llm_api_key` as the auth token; `llm_model` is pinned as the model. Point configs at
the proxy's `/anthropic` route:

- `conf/llm/gpt-5.5-cc.json` — `provider=vllm`, `model=gpt-5.5`,
  `endpoint=https://contextguru.vpc.cloud9.ibm.com/anthropic`.
- `conf/llm/sonnet-5.json` — `provider=vllm`, `model=aws/claude-sonnet-5`, same endpoint.

Put the proxy key in these files' `llm_api_key` (they are gitignored/uncommitted by
convention here). The `x-context-guru-token` is **not** in the config — it is supplied
via `ANTHROPIC_CUSTOM_HEADERS` at run time.

## 4. Run the Claude Code arm

```bash
# from the repo root
export ANTHROPIC_CUSTOM_HEADERS="$(grep -E '^ANTHROPIC_CUSTOM_HEADERS=' \
  /Users/davidboaz/Documents/GitHub/try_wikis/.env | sed -E 's/^[^=]+=//; s/^"//; s/"$//')"

python evaluate.py --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
  --domain hr --mode oracle \
  --llm_config conf/llm/sonnet-5.json \
  --orchestrator claude_code \
  --output_folder results/claude_code/sonnet-5/hr/oracle \
  --concurrency 4 --num_runs 1
```

Notes:
- **Cost rates auto-select by model id** (per 1M tokens): sonnet-5 uses the proxy's
  actual Bedrock map — input 1.52 / output 7.60 / cache-read 0.152 / cache-write 1.90;
  gpt-5/5.5 1.25/10 (approx); opus 5/25; haiku 1/5. Override any with
  `CC_PRICE_{INPUT,OUTPUT,CACHE_READ,CACHE_WRITE}_PER_1M`; where cache rates aren't
  pinned they default to 0.1× / 1.25× input. (These derived multipliers happen to equal
  the sonnet map exactly; `cost_usd` then matches `x-litellm-response-cost-original`.)
- The `[claude-code:unrecognized_model]` log line is harmless — the model is passed
  through to the proxy.
- The session runs in an isolated scratch `cwd`, so Claude Code's built-in
  filesystem tools cannot read the repo (seed SQL = answer keys).

## 5. Run the native ReAct baseline (same model)

```bash
python evaluate.py --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
  --domain hr --mode oracle \
  --llm_config conf/llm/gpt-5.5.json \
  --orchestrator react \
  --output_folder results/react/gpt-5.5/hr/oracle \
  --concurrency 4 --num_runs 1
```

Caveat: the native arm reaches the model via the **OpenAI** route (LangChain
`ChatOpenAI`), which does not currently forward the `x-context-guru-token` header. To
run native through contextguru you must either add a default-header to the `vllm`
branch of `benchmark/llm_client.py`, or point `conf/llm/gpt-5.5.json` at a proxy/route
that doesn't require the extra header. (The Claude Code arm is unaffected — it forwards
the header.)

## 6. Run one task at a time / a train-test partition

- **A single task** (debugging, cost control):
  ```bash
  python evaluate.py … --task_id <task_id> --concurrency 1 --num_runs 1
  # or --limit 1 to just take the first task
  ```
- **A train/test partition.** Generate the split once
  (`data/splits/hr_train_test_split.json`, 3 seeds, 50/50):
  ```bash
  python scripts/make_hr_split.py            # HR, seeds 0-2, 50/50
  ```
  Then select a partition:
  ```bash
  python evaluate.py … \
    --split_file data/splits/hr_train_test_split.json --split test --seed 0
  ```
  `--split {train,test}` + `--seed {0,1,2}`. Task-ids are shared across modes, so the
  same split works for oracle / plus_N. There is no built-in train/test split in the
  dataset (HF splits are *domains*, configs are *tool-modes*) — this file provides one.

## 6b. Parallelism

`--concurrency N` runs N tasks at once. The `claude_code` arm parallelizes safely:
each task gets its own Claude Code CLI subprocess, its own database (fresh DB +
per-task `x-database-id`), its own scratch `cwd`, and its own per-session `env`
(shared `ANTHROPIC_CUSTOM_HEADERS` / `CC_PRICE_*` are inherited, not raced). Verifiers
and DB cleanup are per-task.

Sizing: N = that many Node subprocesses + file descriptors; also bounded by proxy rate
limits, gym-server capacity, and the MCP first-turn connect wait (`MCP_TIMEOUT`, 30s
default). Start at `--concurrency 4`; use `2` for the multi-server `hybrid` domain.
Ray (`ray_experiment_queue.py`) parallelizes across models × domains × modes on top of
this per-domain concurrency.

## 6c. Continual shared memory (opt-in)

By default each task is a fresh, stateless session (empty scratch cwd, no memory). To let
the Claude Code agent **accumulate and reuse memory across tasks**, set
`CC_AUTO_MEMORY_DIR` to a shared directory:

```bash
export CC_AUTO_MEMORY_DIR=results/claude_code/sonnet-5/hr/_memory_seed0
python evaluate.py … --orchestrator claude_code --concurrency 1 …
```

What it does: that dir becomes `CLAUDE_CONFIG_DIR` for every session (isolating all Claude
Code state from the real `~/.claude`), a `CLAUDE.md` memory file is seeded there, and a
memory instruction is appended to the task system prompt telling the agent to consult and
append durable learnings. `autoMemoryDirectory` is also pointed at `<dir>/memory`. Sessions
then share the accumulated `CLAUDE.md` across tasks.

Notes / requirements:
- **Run serially (`--concurrency 1`).** Continual memory is order-dependent, and concurrent
  sessions would race on the shared `CLAUDE.md`. Use a fixed task order (seed-major).
- **Isolation.** State is confined to `CC_AUTO_MEMORY_DIR`; the real `~/.claude` is not
  touched. Claude Code's opaque "auto-memory" (MEMORY.md) does not fire reliably headless —
  the seeded `CLAUDE.md` is the mechanism that works.
- **Scope / leakage.** Use one dir per experiment arm; to avoid train→test leakage, build
  memory on the train split and use a **separate** (or reset) dir for test, or snapshot the
  train-built `CLAUDE.md` and evaluate test read-only. The system prompt gains a memory
  instruction only in this mode (a deliberate deviation for the continual arm).

### Memory content mode (Arm 2a)

Set `CC_MEMORY_MODE=procedural` to record **only reusable procedures** (no task-specific
facts/IDs) — the Arm 2a ablation. Default (unset) is facts+procedures (Arm 2).

## 6d. Faithful auto-memory emulation (Arm 2b)

`CC_MEMORY_MODE=automem` swaps Arm 2's single-blob `CLAUDE.md` for a faithful emulation of
Claude Code's **native auto-memory** mechanism: a structured `memory/` store (one-fact
topic files + a `MEMORY.md` index). Because the headless Agent SDK does **not** inject the
scaffolding the interactive CLI does (see `docs/claude-code-auto-memory.md`), the
orchestrator injects it manually — the verbatim auto-memory system prompt (write path) plus
the current `MEMORY.md` index, truncated to the feature's **first 200 lines / 25 KB**, in a
`<system-reminder>` block (read path). Topic files are recalled **on demand via `Read`**
(the store is exposed through `add_dirs` so it's reachable from the isolated scratch cwd).

```bash
export CC_AUTO_MEMORY_DIR=<repo>/.cc_automem
CC_MEMORY_MODE=automem python evaluate.py … \
  --orchestrator claude_code --concurrency 1 \
  --split_file data/splits/hr_train_test_split.json --seed 0 …
```

Differences from §6c: **no `CLAUDE.md` is seeded** in this mode (the auto-memory store is
the sole channel, so the arm isolates *memory mechanism* as the only difference vs. Arm 2);
the index injected each session is the live, truncated `MEMORY.md`. Same serial/order
requirements as §6c. Store lands in `<dir>/memory/` (`MEMORY.md` + topic `*.md`). Suggested
result folders: `results/claude_code/sonnet-5/hr/{train,test}_seed0_plus10_continual_automem/run_1/`.

**Verify recall works (do this first, 2-task smoke).** Arm 2's recall was free
(`CLAUDE.md` auto-loads); here it depends on the model calling `Read` on topic files at an
absolute path outside the scratch cwd. After a short run, confirm topic files were written
(`ls <dir>/memory/*.md`) and that later sessions issue `Read` calls hitting
`…/memory/*.md` (grep `conversation_flow` for `Read` on the memory path). If reads are
absent, recall is silently failing — check `add_dirs`/permissions before the full run.

## 7. Reading the output (cost is cache-aware)

Each `results_*.json` run object carries, in addition to the standard fields:

- `cost_usd` — **authoritative, cache-aware** cost. `cache_read` tokens are billed at
  the cached rate (0.1× input by default), not full input. Uses `CC_PRICE_*` rates.
- `cost_rates_per_1m` — the rates used (for transparency).
- `usage_totals` — `input/output/cache_read/cache_creation/total` token buckets.
- `cc_cost_usd_reported` — Claude Code's own estimate. **Unreliable**: mispriced at
  Opus rates ($5/$25) for unrecognized models — use `cost_usd` instead.
- Per assistant turn in `conversation_flow`, an `ai_message` carries `usage_metadata`
  (with `input_token_details.cache_read/cache_creation`) and `response_metadata`
  (`model`, `stop_reason`).
- Harness telemetry: `builtin_tools_used`, `mcp_bypass_suspected`, `cc_subtype`.

## 8. Score

```bash
python compute_score.py --results_folder results/claude_code/sonnet-5/hr
```
Success = all `database_state` verifiers pass, scored on final DB state regardless of
harness. (`scan_results.py`'s aggregate cost is *not* cache-aware — trust the per-run
`cost_usd` field for cost.)

## 9. Known caveats

- **HR is refusal-prone.** With the strict HR policy system prompt, both GPT-5.5 and
  Claude Sonnet-5 tend to *halt* multi-step write tasks ("Administrator authority /
  validate user-role") and call no tools — in both harnesses (native ReAct control
  refuses too). A full HR sweep therefore largely measures refusal rate; to see a real
  harness effect, use tasks/domains where the agents act.
- **Tool surface.** oracle/plus_N is reproduced by disallowing the non-selected MCP
  tools; Claude Code's built-in tools remain enabled ("as shipped"). An MCP-only arm
  (disallow built-ins) is a documented future option to isolate loop-effect from
  tool-effect.
- **Config inheritance (reproducibility).** The bundled Claude Code CLI reads the host's
  global config, so the advertised tool surface can include the user's own tools/plugins
  (`WebSearch`, `WebFetch`, `Cron*`, `Monitor`, `SendMessage`, `Skill`, `LSP`, and any
  installed MCP plugins such as context7). No **memory tool / TodoWrite** is exposed, and
  by default no `CLAUDE.md` is loaded (empty scratch cwd + full system-prompt override), so
  tasks are stateless — unless continual memory is enabled (§6c). For a machine-independent
  surface, pin an allow-list (`mcp__<gym>__*` + a fixed built-in set) and disable plugins.
