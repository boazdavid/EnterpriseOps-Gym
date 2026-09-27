#!/usr/bin/env bash
#
# HR benchmark through the Online Memory Proxy in STICKY_MSG mode WITH ONLINE LEARNING.
#
# This is run_hr_proxy_sticky_msg.sh + the proxy's `--learn` feature: the proxy still
# INJECTS the wiki (sticky_msg placement) on the way in, and ALSO learns on the way out —
# after a session goes idle for --learn-idle-s seconds, its last conversation is parsed and
# fed to the existing Learner (extract/select), which writes NEW/updated articles into the
# wiki. Because the wiki is read through to disk on every injection, later sessions in the
# same run benefit from what earlier ones taught (online continual learning).
#
# Two adaptations vs the read-only sticky_msg script — both REQUIRED for learning to work:
#   (a) The proxy writes to a THROWAWAY COPY of the wiki ($WORK_WIKI), never the frozen seed.
#   (b) After evaluate.py finishes we WAIT for the idle timers to fire and the learns to
#       complete BEFORE tearing the proxy down (otherwise the trap kills it first and nothing
#       is ever learned).
#
# The learner's extract/select LLM calls reuse the SAME ete-litellm upstream as the agent,
# via ojt_agent's cache client (LLM_CACHE_CLIENT=1 → LangChain ChatOpenAI on OPENAI_BASE_URL).
# Builder model defaults to azure/gpt-5.5 (guaranteed served); override with LEARN_MODEL.
#
# Chain:  evaluate.py HR agent ─▶ proxy :PORT (sticky_msg + learn, WORK_WIKI) ─▶ ete-litellm azure/gpt-5.5
#                                                    │ (idle ≥ IDLE s)
#                                                    └─▶ Learner (extract/select, LEARN_MODEL) ─▶ writes WORK_WIKI
#
# Usage:  scripts/run_hr_proxy_sticky_msg_learn.sh [LIMIT]
#   LIMIT=3      -> smoke (3 tasks),  IDLE=20s,  output results/proxy_sticky_msg_learn_smoke/hr
#   LIMIT unset  -> full 102-task run, IDLE=60s, output results/proxy_sticky_msg_learn_wiki/hr
#
# Env overrides: START_WIKI (empty|seed|<path>; default empty for full run, seed for smoke),
#   LEARN_MODEL (default azure/gpt-5.5), LEARN_IDLE_S, LEARN_CONCURRENCY (max parallel learns,
#   default 4; 1 = strict serial), CONSOLIDATE_EVERY (consolidate after N memory updates,
#   default 20; 0 = off), LEARN_WAIT_MAX (cap on the post-run wait), LLM_TEMPERATURE (default 1),
#   CONCURRENCY.
set -euo pipefail

LIMIT="${1:-}"

# ---- config ---------------------------------------------------------------
GYM_DIR="/Users/davidboaz/Documents/GitHub/EnterpriseOps-Gym"
TW_DIR="/Users/davidboaz/Documents/GitHub/try_wikis"
TW_PY="$TW_DIR/.venv/bin/python"
GYM_PY="$GYM_DIR/.venv/bin/python"

# Frozen seed wiki (read-only source). The proxy learns into a COPY, never this.
SEED_WIKI="$TW_DIR/data/continual/hr_gpt-5.5_gpt5.5build_offline_index_consol20/wiki"
PORT=8899
CONCURRENCY="${CONCURRENCY:-10}"
MAX_SUBLOOP=4
LEARN_MODEL="${LEARN_MODEL:-azure/gpt-5.5}"      # builder model for extract/select (ete-litellm-served)
LEARN_CONCURRENCY="${LEARN_CONCURRENCY:-4}"      # max learns running at once (LLM phases overlap; writes stay serialized)
CONSOLIDATE_EVERY="${CONSOLIDATE_EVERY:-20}"     # run consolidation after every N memory updates (create/update learns); 0 = off

if [ -n "$LIMIT" ]; then
    OUT="$GYM_DIR/results/proxy_sticky_msg_learn_smoke/hr"; LIMIT_ARG=(--limit "$LIMIT")
    IDLE="${LEARN_IDLE_S:-20}"                    # short idle so the smoke learns quickly
    WAIT_MAX="${LEARN_WAIT_MAX:-300}"             # cap the post-run learn wait (s)
    DEFAULT_START="seed"                          # smoke: start from seed to also test inject+skip
else
    OUT="$GYM_DIR/results/proxy_sticky_msg_learn_wiki/hr"; LIMIT_ARG=()
    IDLE="${LEARN_IDLE_S:-60}"
    WAIT_MAX="${LEARN_WAIT_MAX:-1800}"
    DEFAULT_START="empty"                         # full run: TRUE online continual learning from scratch
fi
# Where the run's writable wiki starts: "empty" (grow from nothing), "seed" (copy the frozen
# seed), or an explicit path. Full-benchmark default is empty; smoke default is seed.
START_WIKI="${START_WIKI:-$DEFAULT_START}"
RUN_DIR="$(dirname "$OUT")"
PROXY_LOG="$RUN_DIR/proxy.log"
WORK_WIKI="$RUN_DIR/wiki"                          # writable, per-run copy the learner grows

UPSTREAM_BASE="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
UPSTREAM_KEY="$("$GYM_PY" -c "import json;print(json.load(open('$GYM_DIR/conf/llm/gpt-5.5.json'))['llm_api_key'])")"

# ---- preflight ------------------------------------------------------------
[ -d "$SEED_WIKI" ] || { echo "FATAL: seed wiki not found: $SEED_WIKI" >&2; exit 1; }
# Smoke is disposable and meant to be re-runnable: clear any prior results so the tasks
# actually re-run. evaluate.py skips tasks whose output already exists, so a stale smoke dir
# would drive 0 requests to the proxy → 0 sessions → 0 learns (a silent no-op). The full run
# is NOT cleared (its output-skip is the intended resume behavior).
[ -n "$LIMIT" ] && rm -rf "$OUT"
mkdir -p "$OUT" "$RUN_DIR"

# Build THIS run's writable wiki (never touch the seed). The proxy learns into $WORK_WIKI.
rm -rf "$WORK_WIKI"
case "$START_WIKI" in
    empty)
        # Fresh empty wiki: empty index + empty memories/, no process_index (fresh gate).
        # In sticky_msg, an empty index makes injection a no-op (server forwards original)
        # until the first learn writes an article — exactly the cold-start we want.
        mkdir -p "$WORK_WIKI/memories"; : > "$WORK_WIKI/index.md"
        echo "[1/5] starting from a FRESH EMPTY wiki: $WORK_WIKI (0 articles; grows from the run) ..." ;;
    seed)
        cp -R "$SEED_WIKI" "$WORK_WIKI"
        echo "[1/5] starting from a COPY of the seed wiki: $WORK_WIKI (from $SEED_WIKI) ..." ;;
    *)
        [ -d "$START_WIKI" ] || { echo "FATAL: START_WIKI path not found: $START_WIKI" >&2; exit 1; }
        cp -R "$START_WIKI" "$WORK_WIKI"
        echo "[1/5] starting from custom wiki: $WORK_WIKI (from $START_WIKI) ..." ;;
esac
START_COUNT="$(find "$WORK_WIKI/memories" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')"
# Baseline snapshot of the starting wiki so the end-of-run summary can show exactly what the
# learner changed, regardless of whether we started empty, from seed, or from a custom wiki.
cp -R "$WORK_WIKI" "$RUN_DIR/wiki_start"

echo "[2/5] HR MCP container (gym-hr :8008) ..."
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

echo "[3/5] starting proxy (sticky_msg + LEARN, idle=${IDLE}s, concurrency=$LEARN_CONCURRENCY, consolidate_every=$CONSOLIDATE_EVERY, builder=$LEARN_MODEL) on :$PORT over WORK_WIKI ..."
PROXY_PID=""
cleanup() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Env notes:
#   OPENAI_BASE_URL/KEY  -> agent upstream (gpt-5.5) AND, via the cache client, the learner.
#   LLM_CACHE_CLIENT=1    -> route learner extract/select through LangChain ChatOpenAI on
#                           OPENAI_BASE_URL (so LEARN_MODEL hits ete-litellm regardless of its
#                           provider prefix; the default litellm path would mis-route azure/*).
#   LLM_MAX_TOKENS=8192   -> headroom for extract's structured-JSON output.
#   LLM_TEMPERATURE=1     -> azure/gpt-5.5 rejects temperature=0 (400: "only default (1)
#                           supported"); the learner client defaults to 0, so this is REQUIRED
#                           for the extract/select path to work with gpt-5.5. Override per model.
( cd "$TW_DIR" && \
  exec env PROXY_PAYLOAD_LOG="$RUN_DIR/payloads.jsonl" \
           OPENAI_BASE_URL="$UPSTREAM_BASE" OPENAI_API_KEY="$UPSTREAM_KEY" UPSTREAM_ALLOWLIST="$UPSTREAM_BASE" \
           MAX_SUBLOOP="$MAX_SUBLOOP" \
           LLM_CACHE_CLIENT=1 LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}" \
           LLM_TEMPERATURE="${LLM_TEMPERATURE:-1}" \
  "$TW_PY" -c "
import sys, truststore; truststore.inject_into_ssl()
sys.argv = ['proxy', '--wiki-dir', '$WORK_WIKI', '--mode', 'sticky_msg', '--port', '$PORT',
            '--learn-idle-s', '$IDLE', '--learn-concurrency', '$LEARN_CONCURRENCY',
            '--consolidate-every', '$CONSOLIDATE_EVERY',
            '--learn-model', '$LEARN_MODEL', '--select-model', '$LEARN_MODEL']
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
echo "[4/5] running HR benchmark (sticky_msg+learn, plus_10_tools, react, concurrency $CONCURRENCY${LIMIT:+, LIMIT=$LIMIT}) ..."
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

# ---- wait for idle-learning to fire and finish ----------------------------
# The benchmark is done, so no more requests arrive: every session now goes idle and, IDLE
# seconds later, triggers a learn. We must keep the proxy ALIVE until those learns complete
# (the EXIT trap would otherwise kill it immediately and nothing would be learned). Poll the
# proxy log for "online-learn" completions and stop once the count has been stable for a
# short window (all pending learns done) or WAIT_MAX is hit.
echo "[5/5] waiting for idle-learning (idle=${IDLE}s, cap ${WAIT_MAX}s) ..."
sleep "$((IDLE + 5))"                                  # let the idle timers fire
learns() { local n; n=$(grep -c "online-learn:" "$PROXY_LOG" 2>/dev/null) || n=0; echo "${n:-0}"; }
# A consolidation is in flight if more "start" lines than "done" lines have been logged. It
# emits no "online-learn" lines while running, so the stability check alone could exit mid-pass
# and the trap would kill the proxy — hence the explicit in-flight guard below.
consol_inflight() {
    local s d; s=$(grep -c "online-consolidate: start" "$PROXY_LOG" 2>/dev/null) || s=0
    d=$(grep -c "online-consolidate: done" "$PROXY_LOG" 2>/dev/null) || d=0
    [ "${s:-0}" -gt "${d:-0}" ] && echo 1 || echo 0
}
prev=-1; stable=0; waited=0
while [ "$waited" -lt "$WAIT_MAX" ]; do
    cur="$(learns)"
    if [ "$cur" -eq "$prev" ] && [ "$cur" -gt 0 ] && [ "$(consol_inflight)" = 0 ]; then
        stable=$((stable + 1))
        [ "$stable" -ge 3 ] && break                  # ~30s stable AND no consolidation running
    else
        stable=0
    fi
    prev="$cur"; sleep 10; waited=$((waited + 10))
done

echo
echo "===== online-learning summary ====="
echo "learns logged:        $(learns)   (proxy.log 'online-learn:' lines)"
echo "  by action:"
grep -o "online-learn:.*action=[a-z]*" "$PROXY_LOG" 2>/dev/null | grep -o "action=[a-z]*" | sort | uniq -c | sed 's/^/    /' || true
echo "consolidations run:   $(grep -c 'online-consolidate: done' "$PROXY_LOG" 2>/dev/null || echo 0)  (every $CONSOLIDATE_EVERY memory updates)"
NEW_COUNT="$(find "$WORK_WIKI/memories" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')"
echo "start wiki:           $START_WIKI"
echo "wiki articles:        start=$START_COUNT → after=$NEW_COUNT  (Δ$((NEW_COUNT - START_COUNT)))"
echo "changed/added files vs start:"
diff -rq "$RUN_DIR/wiki_start" "$WORK_WIKI" 2>/dev/null | grep -v "process_index.jsonl\|wiki.log" | sed 's/^/    /' || true
echo
echo "done. results in $OUT ; grown wiki in $WORK_WIKI ; proxy log $PROXY_LOG"
