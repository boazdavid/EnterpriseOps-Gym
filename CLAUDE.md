# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EnterpriseOps-Gym is a benchmarking framework for evaluating LLM agents on stateful, multi-step enterprise tasks. It runs agents against containerized MCP servers across 8 domains (Teams, CSM, Email, ITSM, Calendar, HR, Drive, Hybrid) and verifies results via SQL state checks.

## Commands

### Setup
```bash
uv sync --extra all          # Install all provider dependencies
unzip gym_dbs.zip            # Extract seed databases
```

### Running Benchmarks
```bash
# Via Ray (parallel, recommended)
python ray_experiment_queue.py --experiment_config conf/ray/experiment.json

# Direct (single domain/mode)
python evaluate.py \
    --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain teams --mode oracle \
    --llm_config conf/llm/<model>.json \
    --output_folder results/react/<model>/teams/oracle \
    --orchestrator react \
    --concurrency 4 --num_runs 1
```

### Scoring
```bash
python compute_score.py --results_folder results/react/<model>/teams/oracle
python compute_score.py --results_folder results/react/<model>/teams  # all modes
```

### Analysis
```bash
python scan_results.py --folder <results_folder> --output report.csv
```

## Architecture

### Execution Flow
1. `evaluate.py` — entry point; loads HuggingFace dataset, seeds per-task databases, dispatches tasks via `TaskQueueWorker`
2. `benchmark/executor.py` (`BenchmarkExecutor`) — orchestrates a single task: initializes MCP clients, runs the orchestrator loop, then runs verifiers
3. `benchmark/mcp_client.py` — JSON-RPC client for MCP tool servers (one per domain); handles DB creation/deletion per task
4. `benchmark/llm_client.py` — LangChain-based multi-provider LLM wrapper
5. `benchmark/verifier.py` — post-execution SQL verification against final DB state

### Orchestrators (`orchestrators/`)
All extend `AgentOrchestrator` (base.py). The orchestrator drives the LLM ↔ MCP tool-call loop.
- `react.py` — standard ReAct loop
- `planner_react.py` — plan generation then execution
- `decomposing_planner.py` — decomposes task into sub-goals

### Configuration
- `conf/llm/<name>.json` — LLM provider credentials and params (supports arrays for load balancing)
- `conf/ray/experiment.json` — which models × domains × modes to run
- `conf/ray/domain_conf.json` — MCP server endpoints per domain
- `conf/ray/llm_concurrency.json` — per-model task concurrency limits

### Task Modes
- `oracle` — agent gets only the correct tools
- `plus_5_tools` / `plus_10_tools` / `plus_15_tools` — agent gets extra distractor tools

### Results
Each task produces a JSON file in the output folder containing runs with: conversation flow, tools used, verifier results, timing, and token usage.

## Key Design Decisions
- Databases are created fresh per task from SQL snapshots (in `gym_dbs.zip`) and destroyed after verification
- Verification checks final DB state, not action sequences — agents can solve tasks any way they want
- The `hybrid` domain spans multiple MCP servers simultaneously; use lower concurrency (2) for hybrid tasks
- Ray handles experiment-level parallelism; `TaskQueueWorker` handles per-model concurrency within a domain
