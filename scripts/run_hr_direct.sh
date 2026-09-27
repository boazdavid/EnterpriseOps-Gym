#!/usr/bin/env bash
#
# "DIRECT" arm: the benchmark's NATIVE knowledge feature — NO proxy, but WITH the
# get_memories tool and the knowledge prompt suffix. The wiki INDEX is embedded in
# the system prompt (via the index_batch suffix), so the agent does NOT call
# list_memories(); it only calls get_memories() to fetch bodies. This mirrors the
# proxy TOOL arm (index in prompt + get_memories), except here retrieval is NATIVE —
# the get_memories calls appear in the agent's own trajectory, not hidden in a proxy.
#
#   evaluate.py HR agent ──(direct)──▶ ete-litellm azure/gpt-5.5
#         │  + 2nd MCP server: knowledge-tool-mcp (get_memories only) over the HR wiki
#         │  + SYSTEM_PROMPT_SUFFIX_FILE = index_batch suffix with {{INDEX_MD}} rendered in
#
# Usage:  scripts/run_hr_direct.sh [LIMIT]
#   LIMIT=3      -> smoke (3 tasks),  output results/direct_smoke/hr
#   LIMIT unset  -> full 102-task run, output results/direct_wiki/hr
set -euo pipefail

LIMIT="${1:-}"

# ---- config ---------------------------------------------------------------
GYM_DIR="/Users/davidboaz/Documents/GitHub/EnterpriseOps-Gym"
TW_DIR="/Users/davidboaz/Documents/GitHub/try_wikis"
TW_PY="$TW_DIR/.venv/bin/python"
GYM_PY="$GYM_DIR/.venv/bin/python"

WIKI="$TW_DIR/data/continual/hr_gpt-5.5_gpt5.5build_offline_index_consol20/wiki"
KPORT=8765
CONCURRENCY=10
SUFFIX_TMPL="$GYM_DIR/system_prompt_knowledge_suffix_index_batch.txt"
if [ -n "$LIMIT" ]; then
    OUT="$GYM_DIR/results/direct_smoke/hr";  LIMIT_ARG=(--limit "$LIMIT")
else
    OUT="$GYM_DIR/results/direct_wiki/hr";    LIMIT_ARG=()
fi
KLOG="$(dirname "$OUT")/kserver.log"
SUFFIX="$(dirname "$OUT")/suffix_index_batch_rendered.txt"   # index_batch with {{INDEX_MD}} filled

# ---- preflight ------------------------------------------------------------
[ -d "$WIKI" ] || { echo "FATAL: wiki not found: $WIKI" >&2; exit 1; }
[ -f "$SUFFIX_TMPL" ] || { echo "FATAL: suffix template not found: $SUFFIX_TMPL" >&2; exit 1; }
[ -f "$WIKI/index.md" ] || { echo "FATAL: wiki has no index.md: $WIKI" >&2; exit 1; }
mkdir -p "$OUT" "$(dirname "$KLOG")"

# ---- render the suffix (evaluate.py appends it verbatim; it does NOT substitute {{INDEX_MD}}) ----
echo "[0/4] rendering index_batch suffix (embedding index.md) -> $SUFFIX ..."
"$GYM_PY" - "$SUFFIX_TMPL" "$WIKI/index.md" "$SUFFIX" <<'PY'
import sys
tmpl, idx, out = sys.argv[1:4]
text = open(tmpl).read()
if "{{INDEX_MD}}" not in text:
    sys.exit(f"template has no {{{{INDEX_MD}}}} placeholder: {tmpl}")
open(out, "w").write(text.replace("{{INDEX_MD}}", open(idx).read()))
PY
grep -q "{{INDEX_MD}}" "$SUFFIX" && { echo "FATAL: placeholder not substituted"; exit 1; }
echo "      rendered ($(wc -l <"$SUFFIX" | tr -d ' ') lines; index embedded, no list_memories)."

echo "[1/4] HR MCP container (gym-hr :8008) ..."
if ! podman ps --format '{{.Names}}' | grep -qx gym-hr; then
    echo "      starting gym-hr ..."; podman start gym-hr
else
    echo "      already running."
fi

# ---- start knowledge MCP server (get_memories ONLY — index is in the prompt) ----
if lsof -nP -iTCP:"$KPORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FATAL: port $KPORT already in use (stale kserver?). Free it first:" >&2
    lsof -nP -iTCP:"$KPORT" -sTCP:LISTEN >&2
    exit 1
fi

echo "[2/4] starting knowledge MCP server (get_memories only) on :$KPORT ..."
KPID=""
cleanup() { [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true; }
trap cleanup EXIT

( cd "$TW_DIR" && exec "$TW_PY" -m ojt_agent.runtime.knowledge_server \
    --folder "$WIKI" --host 127.0.0.1 --port "$KPORT" \
    --tools get_memories ) >"$KLOG" 2>&1 &
KPID=$!

echo "      waiting for :$KPORT (pid $KPID, log $KLOG) ..."
for i in $(seq 1 60); do
    kill -0 "$KPID" 2>/dev/null || { echo "FATAL: kserver exited early"; tail -20 "$KLOG"; exit 1; }
    if (exec 3<>"/dev/tcp/127.0.0.1/$KPORT") 2>/dev/null; then exec 3>&- 3<&-; echo "      up."; break; fi
    sleep 0.5
    [ "$i" = 60 ] && { echo "FATAL: kserver not listening after 30s"; tail -20 "$KLOG"; exit 1; }
done

# ---- run benchmark (direct to gpt-5.5, index-in-prompt + get_memories) ----
echo "[3/4] running HR benchmark DIRECT (index in prompt + native get_memories, plus_10_tools, react, concurrency $CONCURRENCY${LIMIT:+, LIMIT=$LIMIT}) ..."
cd "$GYM_DIR"
MCP_NAME_2="knowledge-tool-mcp" \
MCP_ENDPOINT_2="http://127.0.0.1:$KPORT" \
SYSTEM_PROMPT_SUFFIX_FILE="$SUFFIX" \
LLM_INSECURE_TLS=1 \
"$GYM_PY" evaluate.py \
    --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain hr --mode plus_10_tools \
    --orchestrator react \
    --llm_config conf/llm/gpt-5.5.json \
    --output_folder "$OUT" \
    --num_runs 1 --concurrency "$CONCURRENCY" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

echo "[4/4] done. results in $OUT ; kserver log $KLOG"
