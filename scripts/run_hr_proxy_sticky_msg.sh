#!/usr/bin/env bash
#
# HR benchmark through the Online Memory Proxy in STICKY_MSG mode.
#
# STICKY_MSG = STICKY, but the pinned bodies are re-injected as a synthetic
# get_memories tool-call/result MESSAGE PAIR (after the first user message) instead of
# as system-prompt text — so persisted knowledge sits in the message context, like a
# native tool result (the "Direct-via-proxy" placement). Same SessionStore + sub-loop.
#
# Chain:  evaluate.py HR agent ─▶ proxy :PORT (sticky_msg, HR wiki) ─▶ ete-litellm azure/gpt-5.5
#
# Usage:  scripts/run_hr_proxy_sticky_msg.sh [LIMIT]
#   LIMIT=3      -> smoke (3 tasks),  output results/proxy_sticky_msg_smoke/hr
#   LIMIT unset  -> full 102-task run, output results/proxy_sticky_msg_wiki/hr
set -euo pipefail

LIMIT="${1:-}"

# ---- config ---------------------------------------------------------------
GYM_DIR="/Users/davidboaz/Documents/GitHub/EnterpriseOps-Gym"
TW_DIR="/Users/davidboaz/Documents/GitHub/try_wikis"
TW_PY="$TW_DIR/.venv/bin/python"
GYM_PY="$GYM_DIR/.venv/bin/python"

WIKI="$TW_DIR/data/continual/hr_gpt-5.5_gpt5.5build_offline_index_consol20/wiki"
PORT=8899
CONCURRENCY=10
MAX_SUBLOOP=4
if [ -n "$LIMIT" ]; then
    OUT="$GYM_DIR/results/proxy_sticky_msg_smoke/hr";  LIMIT_ARG=(--limit "$LIMIT")
else
    OUT="$GYM_DIR/results/proxy_sticky_msg_wiki/hr";    LIMIT_ARG=()
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
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FATAL: port $PORT already in use (stale proxy?). Free it first:" >&2
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
    exit 1
fi

echo "[2/4] starting proxy (sticky_msg, MAX_SUBLOOP=$MAX_SUBLOOP) on :$PORT over HR wiki ..."
PROXY_PID=""
cleanup() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# The agent passes the upstream target+key per request via x-upstream-url / x-upstream-api-key
# (LLM_DEFAULT_HEADERS below), so the proxy routes dynamically. UPSTREAM_ALLOWLIST is the SSRF
# guard: the requested URL is honored only if its origin is listed here. OPENAI_BASE_URL/KEY are
# kept as the fallback default upstream for any request that arrives without the headers.
( cd "$TW_DIR" && \
  exec env PROXY_PAYLOAD_LOG="$(dirname "$OUT")/payloads.jsonl" OPENAI_BASE_URL="$UPSTREAM_BASE" OPENAI_API_KEY="$UPSTREAM_KEY" UPSTREAM_ALLOWLIST="$UPSTREAM_BASE" MAX_SUBLOOP="$MAX_SUBLOOP" \
  "$TW_PY" -c "
import sys, truststore; truststore.inject_into_ssl()
sys.argv = ['proxy', '--wiki-dir', '$WIKI', '--mode', 'sticky_msg', '--port', '$PORT', '--no-learn']
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
# Tell the agent to send the upstream target+key to the proxy as request headers. The vllm
# provider in llm_client.py forwards LLM_DEFAULT_HEADERS verbatim to the proxy on every call;
# the proxy validates x-upstream-url against UPSTREAM_ALLOWLIST and routes there with the key.
export LLM_DEFAULT_HEADERS="$("$GYM_PY" -c 'import json,sys; print(json.dumps({"x-upstream-url": sys.argv[1], "x-upstream-api-key": sys.argv[2]}))' "$UPSTREAM_BASE" "$UPSTREAM_KEY")"

echo "[3/4] running HR benchmark (sticky_msg mode, plus_10_tools, react, concurrency $CONCURRENCY${LIMIT:+, LIMIT=$LIMIT}) ..."
cd "$GYM_DIR"
"$GYM_PY" evaluate.py \
    --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain hr --mode plus_10_tools \
    --orchestrator react \
    --llm_config conf/llm/gpt-5.5-proxy.json \
    --output_folder "$OUT" \
    --num_runs 1 --concurrency "$CONCURRENCY" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

echo "[4/4] done. results in $OUT ; proxy log $PROXY_LOG"
