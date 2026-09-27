#!/usr/bin/env bash
#
# FULL HR benchmark (102 tasks, plus_10_tools) through the Online Memory Proxy in
# bulk mode — i.e. WITH the wiki injected. gpt-5.5 agent, run in parallel.
#
# Chain:  evaluate.py HR agent ──(OpenAI /chat/completions)──▶ proxy :PORT (bulk, HR wiki)
#                                                       ──▶ ete-litellm  azure/gpt-5.5
#
# This is the "wiki" arm to compare against the no-wiki gpt-5.5 baseline
# (plus_10, 102 tasks: 39.4% pooled / 42.6% mean verifier, ~18.6% success).
#
# Usage:  scripts/run_hr_proxy_bulk_wiki_full.sh
set -euo pipefail

# ---- config ---------------------------------------------------------------
GYM_DIR="/Users/davidboaz/Documents/GitHub/EnterpriseOps-Gym"
TW_DIR="/Users/davidboaz/Documents/GitHub/try_wikis"
TW_PY="$TW_DIR/.venv/bin/python"
GYM_PY="$GYM_DIR/.venv/bin/python"

LIMIT="${1:-}"
WIKI="$TW_DIR/data/continual/hr_gpt-5.5_gpt5.5build_offline_index_consol20/wiki"
PORT=8899
CONCURRENCY=10
if [ -n "$LIMIT" ]; then
    OUT="$GYM_DIR/results/proxy_bulk_smoke_p10/hr";  LIMIT_ARG=(--limit "$LIMIT")
else
    OUT="$GYM_DIR/results/proxy_bulk_wiki/hr";        LIMIT_ARG=()
fi
PROXY_LOG="$(dirname "$OUT")/proxy.log"

UPSTREAM_BASE="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
UPSTREAM_KEY="$("$GYM_PY" -c "import json;print(json.load(open('$GYM_DIR/conf/llm/gpt-5.5.json'))['llm_api_key'])")"

# ---- preflight ------------------------------------------------------------
[ -d "$WIKI" ] || { echo "FATAL: wiki not found: $WIKI" >&2; exit 1; }
mkdir -p "$OUT" "$(dirname "$PROXY_LOG")"

echo "[1/4] HR MCP container (gym-hr :8008) ..."
if ! podman ps --format '{{.Names}}' | grep -qx gym-hr; then
    echo "      starting gym-hr ..."; podman start gym-hr
else
    echo "      already running."
fi

# ---- start proxy ----------------------------------------------------------
# Fail fast if the port is already taken — otherwise a stale proxy (possibly a
# DIFFERENT wiki/mode) would silently serve the run and corrupt the results.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FATAL: port $PORT already in use (stale proxy?). Free it first:" >&2
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
    exit 1
fi

echo "[2/4] starting proxy (bulk) on :$PORT over HR wiki ..."
PROXY_PID=""
# exec replaces the subshell with python, so PROXY_PID is the uvicorn process
# itself (not a parent shell) and cleanup actually kills it.
cleanup() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

( cd "$TW_DIR" && \
  exec env PROXY_PAYLOAD_LOG="$(dirname "$OUT")/payloads.jsonl" OPENAI_BASE_URL="$UPSTREAM_BASE" OPENAI_API_KEY="$UPSTREAM_KEY" UPSTREAM_ALLOWLIST="$UPSTREAM_BASE" \
  "$TW_PY" -c "
import sys, truststore; truststore.inject_into_ssl()
sys.argv = ['proxy', '--wiki-dir', '$WIKI', '--mode', 'bulk', '--port', '$PORT', '--no-learn']
from ojt_agent.proxy.main import main; main()
" ) >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

echo "      waiting for :$PORT (pid $PROXY_PID, log $PROXY_LOG) ..."
for i in $(seq 1 60); do
    kill -0 "$PROXY_PID" 2>/dev/null || { echo "FATAL: proxy exited early"; tail -20 "$PROXY_LOG"; exit 1; }
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then exec 3>&- 3<&-; echo "      up."; break; fi
    sleep 0.5
    [ "$i" = 60 ] && { echo "FATAL: proxy not listening after 30s"; tail -20 "$PROXY_LOG"; exit 1; }
done

# ---- run benchmark --------------------------------------------------------
echo "[3/4] running FULL HR benchmark (102 tasks, plus_10_tools, 1 run, react, concurrency $CONCURRENCY) ..."
cd "$GYM_DIR"
# Agent passes the upstream target+key to the proxy as request headers (x-upstream-url /
# x-upstream-api-key); the vllm provider forwards LLM_DEFAULT_HEADERS verbatim. The proxy
# validates x-upstream-url against UPSTREAM_ALLOWLIST and routes there with the key.
export LLM_DEFAULT_HEADERS="$("$GYM_PY" -c 'import json,sys; print(json.dumps({"x-upstream-url": sys.argv[1], "x-upstream-api-key": sys.argv[2]}))' "$UPSTREAM_BASE" "$UPSTREAM_KEY")"

"$GYM_PY" evaluate.py \
    --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain hr --mode plus_10_tools \
    --orchestrator react \
    --llm_config conf/llm/gpt-5.5-proxy.json \
    --output_folder "$OUT" \
    --num_runs 1 --concurrency "$CONCURRENCY" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

echo "[4/4] done. results in $OUT ; proxy log $PROXY_LOG"
