#!/usr/bin/env python3
"""Run an EnterpriseOps-Gym evaluation for one (domain, model) with a knowledge base.

Given a KB/wiki directory, this script:
  1. Preflights the container runtime and the domain's MCP container.
  2. Starts the knowledge MCP server (from the try_wikis venv) serving the KB
     via list_articles + get_article on an auto-picked free port.
  3. Runs evaluate.py against ServiceNow-AI/EnterpriseOps-Gym with the KB injected
     as a second MCP server, writing trajectories to --out_dir.
  4. Always stops the knowledge server when the run ends.

KB arm only -- no control run, no delta. See scripts/run_kb_eval.md.
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path

GYM = Path(__file__).resolve().parent.parent
# try_wikis holds the knowledge_server module + its own venv; override with $TRY_WIKIS.
TRY_WIKIS = Path(os.environ.get("TRY_WIKIS", GYM.parent / "try_wikis")).expanduser()
PY_TW = TRY_WIKIS / ".venv/bin/python"
PY_GYM = GYM / ".venv/bin/python"
DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
SUFFIX_FILE = GYM / "system_prompt_knowledge_suffix_listget.txt"
DOMAIN_CONF = GYM / "conf/ray/domain_conf.json"
KSERVER_TOOLS = "list_articles,get_article"


def up(host, port):
    s = socket.socket(); s.settimeout(1)
    try:
        s.connect((host, int(port))); return True
    except Exception:
        return False
    finally:
        s.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def runtime_ok():
    """True if a container runtime daemon (docker or podman) is reachable."""
    for rt in ("docker", "podman"):
        try:
            if subprocess.run([rt, "info"], capture_output=True).returncode == 0:
                return rt
        except FileNotFoundError:
            continue
    return None


def domain_endpoint(domain):
    conf = json.loads(DOMAIN_CONF.read_text())
    if domain not in conf:
        raise SystemExit(f"unknown domain '{domain}'; known: {', '.join(sorted(conf))}")
    ep = conf[domain]["MCP_ENDPOINT"]              # e.g. http://localhost:8002
    hostport = ep.split("://", 1)[1]
    host, port = hostport.split(":")
    return host, int(port)


def preflight(domain, kb_dir, llm_config):
    problems = []
    rt = runtime_ok()
    if not rt:
        problems.append("no container runtime reachable (tried `docker info`, `podman info`)")
    host, port = domain_endpoint(domain)
    if not up(host, port):
        problems.append(f"domain '{domain}' MCP container not reachable on {host}:{port} "
                        f"-- start its container first")
    if not (kb_dir / "index.md").exists():
        problems.append(f"kb_dir has no index.md: {kb_dir}")
    if not llm_config.exists():
        problems.append(f"missing llm config: {llm_config}")
    if not SUFFIX_FILE.exists():
        problems.append(f"missing prompt suffix: {SUFFIX_FILE}")
    if not PY_TW.exists():
        problems.append(f"try_wikis venv python not found: {PY_TW} (set $TRY_WIKIS)")
    if not PY_GYM.exists():
        problems.append(f"gym venv python not found: {PY_GYM}")
    return problems, rt


def start_kserver(kb_dir, port, log_path):
    cmd = [str(PY_TW), "-m", "pipeline.runtime.knowledge_server",
           "--folder", str(kb_dir), "--host", "127.0.0.1", "--port", str(port),
           "--tools", KSERVER_TOOLS]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=str(TRY_WIKIS), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(80):                            # ~40s to bind
        if up("127.0.0.1", port):
            time.sleep(1.0)
            if proc.poll() is not None:
                raise RuntimeError(f"kserver died but :{port} held by a stale server; see {log_path}")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"kserver exited early; see {log_path}")
        time.sleep(0.5)
    raise RuntimeError(f"kserver did not bind on :{port}; see {log_path}")


def stop_kserver(port):
    pids = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True).stdout.split()
    for pid in pids:
        subprocess.run(["kill", "-9", pid], capture_output=True)


def eval_cmd(args, out_dir):
    llm_config = f"conf/llm/{args.model}.json"
    return [str(PY_GYM), "evaluate.py",
            "--hf_dataset", DATASET,
            "--domain", args.domain,
            "--mode", args.mode,
            "--llm_config", llm_config,
            "--orchestrator", "react",
            "--output_folder", str(out_dir),
            "--concurrency", str(args.concurrency),
            "--num_runs", str(args.num_runs)]


def eval_env(port):
    env = dict(os.environ)
    env["MCP_NAME_2"] = "knowledge-tool-mcp"
    env["MCP_ENDPOINT_2"] = f"http://127.0.0.1:{port}"
    env["SYSTEM_PROMPT_SUFFIX_FILE"] = str(SUFFIX_FILE)
    return env


def main():
    p = argparse.ArgumentParser(description="Run EnterpriseOps-Gym eval with a knowledge base (KB arm).")
    p.add_argument("--domain", required=True, help="e.g. teams, csm, email, itsm, hr, drive")
    p.add_argument("--model", required=True, help="maps to conf/llm/<model>.json")
    p.add_argument("--kb_dir", required=True, type=Path, help="KB/wiki folder to serve (must contain index.md)")
    p.add_argument("--out_dir", required=True, type=Path, help="trajectories output folder")
    p.add_argument("--mode", default="plus_10_tools", help="tool-set mode (default: plus_10_tools)")
    p.add_argument("--num_runs", default=1, type=int, help="number of seeds/runs (default: 1)")
    p.add_argument("--concurrency", default=3, type=int, help="task concurrency (default: 3)")
    p.add_argument("--dry-run", action="store_true", help="preflight + print commands, do not run")
    args = p.parse_args()

    kb_dir = args.kb_dir.expanduser().resolve()
    llm_config = GYM / f"conf/llm/{args.model}.json"

    problems, rt = preflight(args.domain, kb_dir, llm_config)
    if problems:
        print("PREFLIGHT PROBLEMS:")
        for x in problems:
            print("  -", x)
        if not args.dry_run:
            raise SystemExit(1)
    else:
        print(f"[preflight] ok (runtime={rt})")

    port = free_port()
    log_path = f"/tmp/kb_eval_{args.domain}_{args.model}_kserver.log"

    if args.dry_run:
        print("DRY RUN")
        print(f"  kserver: {PY_TW} -m pipeline.runtime.knowledge_server --folder {kb_dir} "
              f"--host 127.0.0.1 --port <free> --tools {KSERVER_TOOLS}  (cwd={TRY_WIKIS})")
        print(f"  eval:    {' '.join(eval_cmd(args, args.out_dir))}  (cwd={GYM})")
        print(f"  env:     MCP_NAME_2=knowledge-tool-mcp MCP_ENDPOINT_2=http://127.0.0.1:<free> "
              f"SYSTEM_PROMPT_SUFFIX_FILE={SUFFIX_FILE}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kserver] starting on :{port} (log {log_path})")
    proc = start_kserver(kb_dir, port, log_path)
    try:
        cmd = eval_cmd(args, args.out_dir)
        print(f"[eval] {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(GYM), env=eval_env(port)).returncode
    finally:
        print(f"[kserver] stopping :{port}")
        stop_kserver(port)
    print(f"=== DONE (eval rc={rc}) -> {args.out_dir} ===")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
