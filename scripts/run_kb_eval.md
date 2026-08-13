# run_kb_eval.py — EnterpriseOps eval with a knowledge base

Runs a single **(domain, model)** EnterpriseOps-Gym evaluation with a knowledge
base (wiki) served to the agent. It starts the knowledge MCP server itself,
injects it as a second MCP server, runs the eval, and tears the server down.

**KB arm only** — no control run and no delta. To compare against a no-KB
baseline, run your control separately (or use `scripts/run_5d_matrix.py` in the
`try_wikis` repo, which does paired control+KB deltas).

## What it does

1. **Preflight** (fails fast):
   - A container runtime daemon is reachable (`docker info`, falling back to `podman info`).
   - The domain's MCP container is reachable at the endpoint in
     `conf/ray/domain_conf.json` (e.g. `teams` → `localhost:8002`). **You must
     start that container yourself first** — this script does not start it.
   - `kb_dir/index.md` exists; `conf/llm/<model>.json` exists; the prompt suffix
     and both venvs exist.
2. **Starts the knowledge MCP server** from the `try_wikis` venv:
   `python -m pipeline.runtime.knowledge_server --folder <kb_dir> --tools list_articles,get_article`
   on an auto-picked free port, and waits for it to bind.
3. **Runs the eval** (`evaluate.py`) against `ServiceNow-AI/EnterpriseOps-Gym`
   with `MCP_NAME_2=knowledge-tool-mcp`, `MCP_ENDPOINT_2=http://127.0.0.1:<port>`,
   and `SYSTEM_PROMPT_SUFFIX_FILE=system_prompt_knowledge_suffix_listget.txt`.
4. **Stops the knowledge server** in a `finally` (kill by port), always.

## Usage

```bash
python scripts/run_kb_eval.py \
    --domain teams \
    --model sonnet-5 \
    --kb_dir /path/to/wiki_dir \
    --out_dir results/kb/sonnet-5/teams
```

Preview the exact commands without running anything:

```bash
python scripts/run_kb_eval.py --domain teams --model sonnet-5 \
    --kb_dir /path/to/wiki_dir --out_dir results/kb/sonnet-5/teams --dry-run
```

## Arguments

| Arg | Required | Default | Notes |
|-----|----------|---------|-------|
| `--domain` | yes | — | `teams`, `csm`, `email`, `itsm`, `hr`, `drive` |
| `--model` | yes | — | maps to `conf/llm/<model>.json` |
| `--kb_dir` | yes | — | KB/wiki folder to serve; must contain `index.md` |
| `--out_dir` | yes | — | trajectories output folder |
| `--mode` | no | `plus_10_tools` | tool-set mode |
| `--num_runs` | no | `1` | seeds/runs |
| `--concurrency` | no | `3` | task concurrency |
| `--dry-run` | no | off | preflight + print commands, run nothing |

Fixed by design: dataset is `ServiceNow-AI/EnterpriseOps-Gym`; retrieval mode is
**listget** (`list_articles` + `get_article`); orchestrator is `react`; the KB
server port is always auto-picked.

## Preconditions

- The domain's MCP container is running (this script only checks it, never starts it).
- `try_wikis` is a sibling of this repo, or `$TRY_WIKIS` points at it, and its
  `.venv` is set up (that venv provides `pipeline.runtime.knowledge_server`).

## Output

Trajectories land in `--out_dir/run_1 .. run_<num_runs>`. Score them with:

```bash
python compute_score.py --results_folder <out_dir>
```
