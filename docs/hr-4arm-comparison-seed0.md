# HR harness × continual-learning comparison — seed 0

Four agent configurations ("arms") evaluated on the same EnterpriseOps-Gym **HR** task
stream, isolating two axes: **harness** (native ReAct vs. Claude Code) and **continual
learning** (none vs. Claude Code memory vs. external wiki+retrieval). Plus one ablation,
**Arm 2a**, varying Arm 2's *memory content* (procedural-only vs. facts+procedures) — see
[Ablation](#ablation--memory-content-factsprocedures-arm-2-vs-procedural-only-arm-2a).

## Common setup (all arms)

| | |
|---|---|
| Domain | `hr` (`sn-hr-internal`, gym MCP at `http://localhost:8008`) |
| Mode | `plus_10_tools` (oracle tools + 10 distractors; ~22 tools exposed/task) |
| Model | **`aws/claude-sonnet-5`** (Bedrock, via LiteLLM proxies) |
| Task set | seed-0 split of the 102 HR tasks → **51 train + 51 test** |
| Split file | `data/splits/hr_train_test_split.json` (seed `0`; generator `scripts/make_hr_split.py`) |
| Verifiers | per-task SQL `database_state` checks (unchanged; score final DB state) |
| Pricing (per 1M tok) | input $1.52 · output $7.60 · cache-read $0.152 · cache-write $1.90 |

"Success" = all of a task's verifiers pass. "Verifier-level" = fraction of individual
verifiers passed. Costs are **cache-aware**; "full" = list price ignoring cache discount.

---

## Arm 1 — native ReAct (no memory)

- **Harness / orchestrator:** benchmark-native ReAct (`--orchestrator react`).
- **Memory:** none (each task independent).
- **Model route:** contextguru **OpenAI** route (`https://contextguru.vpc.cloud9.ibm.com/openai/v1`) via `conf/llm/sonnet-5-openai.json`; env `LLM_DEFAULT_HEADERS` (context-guru token) + `LLM_INSECURE_TLS=1`.
- **Tasks / order:** seed-0 train (51), then test (51); config-basename sorted within each; **order irrelevant** (stateless, parallel).
- **Result files:**
  - train → `results/react/sonnet-5/hr/train_seed0_plus10/run_1/results_*.json`
  - test → `results/react/sonnet-5/hr/test_seed0_plus10/run_1/results_*.json`

## Arm 3 — Claude Code, stateless (no memory)

- **Harness:** Claude Code via the Claude Agent SDK (`--orchestrator claude_code`), gym tools bridged by the stdio shim `orchestrators/gym_mcp_proxy.py`.
- **Memory:** none — a **fresh throwaway `CLAUDE_CONFIG_DIR` per task** (isolated from `~/.claude`, no seeded memory, no plugins).
- **Model route:** contextguru **Anthropic** route (`…/anthropic`) via `conf/llm/sonnet-5.json`; env `ANTHROPIC_CUSTOM_HEADERS` (guru token). `permission_mode=bypassPermissions`.
- **Tasks / order:** seed-0 train (51), then test (51); sorted; **order irrelevant** (stateless, parallel).
- **Result files:**
  - train → `results/claude_code/sonnet-5/hr/train_seed0_plus10_stateless/run_1/`
  - test → `results/claude_code/sonnet-5/hr/test_seed0_plus10_stateless/run_1/`

## Arm 2 — Claude Code, continual memory via CLAUDE.md

- **Harness:** Claude Code (as arm 3) **+ shared continual memory via `CLAUDE.md`**.
- **Memory mechanism:** the **`CLAUDE.md` files** channel (not the auto-memory feature). Shared, isolated dir `.cc_memory/` used as `CLAUDE_CONFIG_DIR`; a seeded `CLAUDE.md` is auto-loaded into every session and the agent **appends learnings to it via Write/Edit**, driven by a `MEMORY_INSTRUCTION` suffix on the system prompt each session. Memory **carried from train into test** and kept growing (final `.cc_memory/CLAUDE.md` = **205 lines**; per-session snapshots under `.cc_memory/file-history/`).
### Why not Claude Code's native *auto-memory* feature?

We intended to use Claude Code's built-in **auto-memory** (the self-maintained `memory/`
`MEMORY.md` store) but it **does not work when Claude Code is driven headlessly through the
Python Agent SDK** — so this arm uses the `CLAUDE.md` channel instead. Details:

- **The mechanism gap.** Auto-memory has **no dedicated "save" tool** — the agent writes to
  the memory file with ordinary `Write`/`Edit` calls. For that to happen, the agent must be
  *told* the notebook exists (context injected at session start: "you have a `MEMORY.md` at
  `<path>`; record durable learnings there"). The **interactive `claude` CLI injects that
  scaffolding automatically; the headless Agent-SDK path does not.** With no awareness of a
  memory system, the model never writes — even when explicitly asked to "remember … save to
  your memory," it replies *"I have no persistent memory"* or pokes around with `Bash`/`Read`
  and gives up.
- **Observed:** across the ~102-task run the `memory/` store stayed **empty**; a dedicated
  spike reproduced this across **five configurations** — `query()` (one-shot) and
  `ClaudeSDKClient` (persistent); `settings` as a JSON string and as a real scope file with
  `setting_sources=["user"]`; and `permission_mode` `acceptEdits` and `bypassPermissions` —
  all with `autoMemoryEnabled`+`autoMemoryDirectory` set and an explicit "remember" prompt.
  None ever wrote `MEMORY.md`.
- **Ruled out** (so it's specifically the missing context-injection, not these): file-write
  **permission** (allowed via `acceptEdits`/`bypassPermissions`); **`--bare`** (the SDK never
  passes it); **unloaded settings** (used a scope file + `setting_sources`); model refusal.
  Docs imply headless support, so this is likely an undocumented SDK gap (worth a `/feedback`
  / SDK-issue report).
- **Why Arm 2 (`CLAUDE.md`) works instead:** we manually supply the exact piece auto-memory
  needs and doesn't get — we **seed the memory file** *and* **inject the `MEMORY_INSTRUCTION`**
  telling the agent it has the file and to append learnings. So Arm 2 is effectively
  "auto-memory, done explicitly," and it works precisely because it doesn't rely on the SDK to
  inject that awareness. A true auto-memory arm would require driving the *interactive* CLI
  per task (a different harness) — impractical for a 100-task stream — so it's out of scope here.
- **`MEMORY_INSTRUCTION`** (verbatim, appended to `config.system_prompt` every session — [orchestrators/claude_code.py](../orchestrators/claude_code.py)):

  > ---
  > CONTINUAL MEMORY: You have a persistent memory file (your Claude config CLAUDE.md) shared across tasks in this environment. At the start, consult it for reusable facts/procedures. When you learn something durable and reusable (a working approach, an entity id, a gotcha), append a concise note to it with the Write/Edit tools. Keep it short and general — do not store task-specific secrets.

- **Seeded `CLAUDE.md` header** (written once by `_seed_memory_file` if the file is absent; auto-loaded into context each session as the `CLAUDE.md` memory):

  > # Agent Memory
  >
  > Durable, reusable facts and procedures learned while solving earlier tasks in this environment. Append concise new learnings here.
- **Model route:** same as arm 3 (contextguru `…/anthropic`, `conf/llm/sonnet-5.json`, guru header) + env `CC_AUTO_MEMORY_DIR=<repo>/.cc_memory`.
- **Tasks / order:** **serial** (`--concurrency 1`), config-basename sorted; train stream (51) → test stream (51). **Order matters** (continual).
- **Result files:**
  - train → `results/claude_code/sonnet-5/hr/train_seed0_plus10_continual/run_1/`
  - test → `results/claude_code/sonnet-5/hr/test_seed0_plus10_continual/run_1/`
  - memory → `.cc_memory/CLAUDE.md` (+ `.cc_memory/file-history/`)

## Arm 4 — wiki-augmented, index retrieval (continual)

- **Framework:** `try_wikis` continual pipeline (`scripts/continual/run_continual.py`), **prequential** (evaluate task *i* before ingesting its trajectory), **Approach B** (`--learn-from self` — wiki grows from the agent's *own* trajectories).
- **Agent harness:** native ReAct (gym `evaluate.py`) + a `knowledge_server` (port 8766) exposing `get_articles`; the wiki **index/TOC is injected into the prompt** (`--retrieval index`, suffix `system_prompt_knowledge_suffix_index_batch.txt`, `{{INDEX_MD}}` re-filled each task from the wiki as of tasks < *i*), and the agent fetches article bodies on demand.
- **Agent model:** `aws/claude-sonnet-5` via contextguru **OpenAI** route (`conf/llm/sonnet-5-openai.json`, `LLM_DEFAULT_HEADERS` + `LLM_INSECURE_TLS=1`).
- **Wiki builder / extractor model:** `aws/claude-sonnet-5` via **ete-litellm** (`CONTINUAL_BUILDER=openai/aws/claude-sonnet-5`, `OPENAI_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1`, key from `conf/llm/gpt-5.5.json`, `LLM_NO_PREFILL=1`). Final wiki = **16 articles**.
- **Control baseline:** reused **arm 1** trajectories (`prepare_continual --control-src …/react/sonnet-5/hr/train_seed0_plus10/run_1`); test phase ran with `--no-control-gate`.
- **Tasks / order:** **serial**, prequential; manifest order = 51 train (sorted) **then** 51 test (sorted, appended). **Order matters** (continual).
- **Location / result files** (under `/Users/davidboaz/Documents/GitHub/try_wikis/data/continual/hr_sonnet-5_index/`):
  - agent trajectories (train+test, 102) → `knowarm/run_1/results_*.json`
  - per-task log → `continual_log.jsonl` · wiki KB → `wiki/` (`concepts/*.md`, `index.md`)
  - control copy → `control/run_1/` · ingest logs → `ingest_*.log`

---

## Results — seed 0

### Learning curve — performance across the task stream

Interactive chart: **[hr-4arm-learning-curve.html](./hr-4arm-learning-curve.html)** (self-contained;
open in a browser). Per-task verifier pass rate as a trailing moving-average over the
102-task stream (51 train → 51 test, same order for all arms). Features: a **moving-average
window slider**, **click-to-show/hide** each line, distinct dash patterns + direct end-labels
(CVD-safe), a train/test divider, and a hover tooltip. Watch the two continual arms
(CC-continual, wiki) hold up across the train→test boundary while the memoryless arms dip.

### Unified (all 102) — accuracy, turns, cost

| arm | success | verifier | avg turns | avg tool calls | runtime cost: cache-aware (full) | build |
|---|---|---|---|---|---|---|
| 1 native (no mem) | 31/102 (30.4%) | 367/528 (69.5%) | 5.9 | 10.8 | **$10.10** ($18.91) | — |
| 3 CC stateless | 20/102 (19.6%) | 321/528 (60.8%) | 6.0 | 11.4 | $18.39 ($51.06) | — |
| 2 CC continual (proc+facts mem) | 38/102 (37.3%) | 389/528 (73.7%) | 6.6 | 12.1 | $20.61 ($62.51) | $0.46 |
| 2a CC continual (proc) | 28/102 (27.5%) | 380/528 (72.0%) | 6.6 | 12.8 | $20.76 ($59.10) | $0.13 |
| 4 wiki (index continual) | 32/102 (31.4%) | 391/528 (74.1%) | 7.2 | 12.2 | $16.14 ($29.02) | $7.46 |

- **avg tool calls** ≈ 11–12 for all arms (native slightly fewer at 10.8) — confirming they do
  comparable *work*; the difference is turn-packing (native batches into 5.9 turns) and
  per-turn context, not tool volume.

- **"avg turns" = model turns (LLM calls), collapsing parallel tool-call batches.** All four
  arms take ~6–7 turns. 
- "full" = list price with no cache discount (`cache_read` billed at full $1.52 vs $0.152). Cache-aware is the real bill; the CC arms' full/cached gap (~2.8–3.1×) shows how much prompt caching saves them.
- **Wiki cost split** — runtime: train $6.90 / test $9.24 ($16.14 total); build (ingest/extractor): train $3.47 / test $3.99 ($7.46 total, ~32% of the arm's total).
- **CC-continual cost split — build = the memory *write* only ($0.46).** Unlike wiki's separate extractor, CC's memory is inline, but the *write* is isolable: it lands as a distinct near-final turn (median 87% through the session, penultimate; never the first half). Across 102 tasks there are **40 `Edit` write-turns** (on 38 tasks); costing each as its context-read (~51K cache-read tok/turn → $0.31) + the `Edit` output (~19K tok → $0.15) gives **build ≈ $0.46** (full-price $3.26). The **runtime ($20.61)** is the rest — task-solving **plus the carry cost** (the growing `CLAUDE.md` auto-loaded into every turn's prefix, ~$2.22), which is inline and stays in runtime because it can't be cleanly separated. Takeaway: **writing memory is cheap (~$0.46); its real cost is re-reading the KB each session** — reflected in the higher runtime vs. stateless ($20.61 vs $18.39).

### Reads
- **Harness effect (memoryless): native > CC-stateless.** Native 69.5% vs 60.8% verifier at **~half the cost** (both take ~6 turns).
- **Continual learning is the dominant factor.** CC-continual beats CC-stateless by **+17.7pp success / +12.9pp verifier**; wiki beats native by **+1.0pp success / +4.6pp verifier**. It lifts the weak CC harness to the top.
- **Generalization holds on the (harder) test split.** Memoryless arms drop train→test (native −8.2pp, CC-stateless −17.3pp verifier); the continual arms nearly hold (CC-continual 75.1→72.2, wiki 74.3→73.8) — accumulated knowledge offsets the difficulty.
- **Efficiency:** native is cheapest to ~30% success ($0.33/solved); CC-continual is the most cost-effective route to the higher ~37% success / ~74% verifier tier ($0.55/solved); wiki matches the top verifier rate at higher total cost (runtime + $7.46 build).

### Why arm 3 (CC-stateless) trails — not errors, over-halting

CC-stateless is the lowest scorer; the cause is **behavioral, not run failures**:

- **No error artifact.** All 102 tasks completed with `cc_subtype: success` — zero crashes, timeouts, max-turns, or MCP-bypass; MCP tools were used on 98/102 (the 4 zero-tool tasks were deliberate halts). So the score reflects genuine task performance.
- **The Claude Code harness over-halts on the strict HR policy.** Paired vs. native (same model/tasks): **native-solved-only = 15 vs. CC-solved-only = 4** (net −11). CC's losses are frequently *full halts* (`0/8, 0/2, 0/2, 1/5`), not near-misses. Example — a task native solved 5/5, CC-stateless stopped with no changes: *"Operation halted — policy violation detected before any changes were made… Thomas Green's account is currently inactive. Per policy Section 6 … ensure users are active before…"* → 1/5. Given the same policy, native just proceeds and completes the writes. Claude Code's scaffolding + the policy's "halt and cite" clause make it more conservative.
- **No memory to correct it.** The continual arm (same harness) recorded exactly these gotchas in `CLAUDE.md` (tool-enum quirks, "priority defaults to *planning* not *moderate*", entity IDs, and "how to proceed" patterns), which is why **CC-continual reaches 37.3%/73.7% (+17.7pp success over stateless)**. Stateless starts fresh each task and repeats the same conservative halts and schema mistakes.

Net: CC-stateless trails because the memoryless Claude Code loop over-halts on the policy and has no accumulated knowledge to get past it — precisely the deficit the continual memory fills.

### Why the CC arms cost more — heavier per-turn context, not more turns

All four arms take **~6–7 model turns** and the CC arms **do parallelize** tool calls
(batches up to 12–14/turn). So the ~2× cost gap is **not** more
turns or more round-trips — it's how many tokens each turn carries. Per-turn breakdown:

| per turn | native | CC stateless | CC continual | wiki |
|---|--:|--:|--:|--:|
| turns/task | 5.9 | 6.0 | 6.6 | 7.2 |
| fresh input tok/turn | 6,634 | 445 | 408 | 10,212 |
| cache-**read** tok/turn | 10,780 | 41,715 | 51,198 | 12,850 |
| cache-**write** tok/turn | 0 | 8,791 | 9,019 | 0 |
| **$ from cache-write** | $0.00 | **$10.16** | **$11.50** | $0.00 |

Two drivers, both per-turn:

- **A large cached harness prefix, re-read every turn.** Claude Code ships its full context
  each turn — its agent system prompt **plus every built-in tool schema** (Bash/Read/Write/
  Edit/Glob/Grep/WebSearch/… and the context7 plugin) **on top of** the 22 gym MCP tools. That
  drives **~42–51K cache-read tokens/turn** vs native's ~11K (~3–5× the context). Native ReAct
  sends only the task prompt + gym tools.
- **A cache-*creation* bill native doesn't pay.** The CC arms write ~9K cache tokens/turn at
  the 1.25× rate ($1.90/1M) → **~$10–11.50**, which is essentially the whole gap. Native's
  OpenAI-route client writes no cache (cache-read only, at $0.152/1M).

So the earlier "CC does more steps / serializes tools" explanation was wrong (a
message-counting artifact — the SDK surfaces each parallel `tool_use` as its own
`AssistantMessage`). Corrected: **same turns, same parallelism; CC just processes a much
larger cached context per turn and re-creates cache**, which is where the money goes.
**Wiki** sits between — native harness (no cache-write) but a heavier prompt (~10K
input/turn from the injected wiki index).

### Arm 2's memory operations — writes are explicit steps, reads are free

How the continual arm actually touches `CLAUDE.md`, and why memory barely adds to its step count:

| memory op | tool | count | |
|---|---|---|---|
| **write** to `CLAUDE.md` | `Edit` | **40 calls, on 38/102 tasks (~37%)** | explicit steps — the agent appends a learning |
| **read** `CLAUDE.md` | `Read` | 15 calls | occasional explicit re-reads only |
| memory-op steps / task | — | **0.54 avg** | the extra steps memory costs |

- **Writing is an explicit step.** The ~205-line memory was built from **40 `Edit` appends across 38 tasks** — on ~⅓ of tasks the agent judged it had learned something durable and spent a step recording it (all `Edit`, appending to the seeded file; no `Write`/recreate).
- **Reading is mostly *not* a step.** `CLAUDE.md` is **auto-loaded into context at session start** (the CLAUDE.md-files mechanism), so the agent doesn't spend a tool call to consult it in the normal case; the 15 `Read` calls are occasional re-reads, not the primary access path.
- **Memory is nearly free:** writes happen on only ~⅓ of tasks (40 `Edit` appends across 38) and reads are **auto-loaded** into context (no tool call in the normal case), so memory adds little — consistent with arm 2 taking only ~0.6 more turns/task than stateless (**6.6 vs 6.0**). CC's cost is dominated by its per-turn cached context (previous section), not by memory operations.

### Ablation — memory content: facts+procedures (Arm 2) vs procedural-only (Arm 2a)

**Arm 2a** is Arm 2 with one change: `CC_MEMORY_MODE=procedural` swaps the memory instruction
so the agent records **only reusable procedures** (generalizable how-to / tool-usage patterns)
and is told **not** to store task-specific facts, entity IDs, names, or data values. Same
harness/model/tasks/protocol (continual, serial, keeps learning through test). Files:
`results/claude_code/sonnet-5/hr/{train,test}_seed0_plus10_continual_proc/`, memory in
`.cc_memory_proc/`.

| | TRAIN solved | TRAIN verifier | TEST solved | TEST verifier | ALL solved | ALL verifier | cost | memory |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 facts+procedures | 17/51 | 75.1% | **21/51** | **72.2%** | **38/102 (37.3%)** | **73.7%** | $21.07 | 206 lines |
| 2a procedural-only | 20/51 | 77.0% | **8/51** | 66.9% | 28/102 (27.5%) | 72.0% | $20.88 | 78 lines |

**Finding — counter-intuitive: storing *facts* generalizes better than *procedures* here.**
Procedural-only actually **won on train** (20 vs 17 solved) with a **~4× smaller** memory (78 vs
206 lines), but **collapsed on test** — **8 vs 21 solved on the identical 51 test tasks** — and
lost overall (27.5% vs 37.3%). The train→test move is opposite for the two: facts+procedures
**improved** (17→21), procedural-only **fell** (20→8), a strong signal (same tasks) that the
dropped facts were doing the work.

Why: **train and test share the same HR environment** (same org, entities, schema, enum
quirks), so the specific facts Arm 2 stored — entity IDs, `hr_service_id`s, "`priority`
defaults to *planning* not *moderate*", valid enum lists — are **directly reusable** on the
"unseen" test tasks, which touch those same tables. Procedural-only deliberately discarded
exactly the specifics that make the SQL verifiers pass. So the intuition "procedures
generalize, facts don't" is **reversed** in a benchmark where the world is fixed across the
stream. (Cost was ~unchanged, $20.88 vs $21.07, despite the smaller memory — the carry saving
is marginal.) *Single seed; the +13-solved test gap is large but n=51.*

### Caveats
- **Single seed, single order.** Magnitudes need seeds 1–2 for error bars.
- The continual arms kept **learning through the test stream** (stateful arms continue to learn), so later test tasks benefited from more accumulated knowledge — this measures continual improvement, not frozen-KB transfer.
- Costs use one sonnet-5 Bedrock price sheet; the wiki agent runs via the OpenAI route and CC arms via the Anthropic route of the same proxy family.
