#!/usr/bin/env python3
"""Scan result JSON files in a folder and produce a CSV report."""

import argparse
import csv
import json
import os
import sys


def process_file(filepath):
    with open(filepath) as f:
        data = json.load(f)

    runs = data.get("runs", [])
    n = len(runs)
    if n == 0:
        return None

    passed = sum(1 for r in runs if r.get("overall_success"))

    durations = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []
    turns_list = []
    messages_list = []

    knowledge_calls_list = []
    max_msg_tokens_list = []
    costs = []
    KB_TOOLS = {"get_article", "list_articles"}

    for run in runs:
        durations.append(run.get("execution_time_ms", 0) / 1000.0)

        conv = run.get("conversation_flow", [])
        knowledge_calls = sum(
            1 for m in conv
            if m.get("type") == "tool_result" and m.get("tool_name") in KB_TOOLS
        )
        knowledge_calls_list.append(knowledge_calls)
        ai_messages = sum(
            1 for m in conv
            if m.get("type") == "ai_message"
            and not any(
                tc.get("name") in KB_TOOLS or tc.get("function", {}).get("name") in KB_TOOLS
                for tc in m.get("tool_calls", [])
            )
        )
        messages_list.append(ai_messages)

        user_turns = sum(1 for m in conv if m.get("type") == "user_message")
        turns_list.append(user_turns)

        run_prompt = 0
        run_completion = 0
        run_total = 0
        run_cost = 0.0
        run_max_msg_tokens = 0

        for msg in conv:
            if msg.get("type") == "ai_message":
                usage = msg.get("usage_metadata", {})
                in_t = usage.get("input_tokens", 0)
                out_t = usage.get("output_tokens", 0)
                total_t = usage.get("total_tokens", 0)
                run_prompt += in_t
                run_completion += out_t
                run_total += total_t
                if total_t > run_max_msg_tokens:
                    run_max_msg_tokens = total_t

                # Default pricing (GPT-5 approximate): $1.25/1M input, $10/1M output
                cost = (in_t * 1.25 + out_t * 10.0) / 1_000_000
                run_cost += cost

        prompt_tokens_list.append(run_prompt)
        completion_tokens_list.append(run_completion)
        total_tokens_list.append(run_total)
        max_msg_tokens_list.append(run_max_msg_tokens)
        costs.append(run_cost)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "file": os.path.basename(filepath),
        "n": n,
        "passed": passed,
        "pass@1": passed / n if n > 0 else 0,
        "avg_duration_s": avg(durations),
        "avg_cost": avg(costs),
        "avg_prompt_tokens": avg(prompt_tokens_list),
        "avg_completion_tokens": avg(completion_tokens_list),
        "avg_total_tokens": avg(total_tokens_list),
        "avg_turns": avg(turns_list),
        "avg_ai_messages": avg(messages_list),
        "avg_knowledge_calls": avg(knowledge_calls_list),
        "avg_max_msg_tokens": avg(max_msg_tokens_list),
    }


def aggregate_directory(file_rows):
    """Aggregate per-file results into a single directory-level row."""
    n = sum(r["n"] for r in file_rows)
    passed = sum(r["passed"] for r in file_rows)

    def weighted_avg(key):
        total = sum(r[key] * r["n"] for r in file_rows)
        return total / n if n > 0 else 0

    return {
        "n": n,
        "passed": passed,
        "pass@1": passed / n if n > 0 else 0,
        "avg_duration_s": weighted_avg("avg_duration_s"),
        "avg_cost": weighted_avg("avg_cost"),
        "avg_prompt_tokens": weighted_avg("avg_prompt_tokens"),
        "avg_completion_tokens": weighted_avg("avg_completion_tokens"),
        "avg_total_tokens": weighted_avg("avg_total_tokens"),
        "avg_turns": weighted_avg("avg_turns"),
        "avg_ai_messages": weighted_avg("avg_ai_messages"),
        "avg_knowledge_calls": weighted_avg("avg_knowledge_calls"),
        "avg_max_msg_tokens": weighted_avg("avg_max_msg_tokens"),
    }


def main():
    parser = argparse.ArgumentParser(description="Scan result JSON files and produce a CSV report")
    parser.add_argument("folder", help="Folder containing result JSON files")
    parser.add_argument("-o", "--output", default=None, help="Output CSV file (default: stdout)")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        os.path.join(dirpath, f)
        for dirpath, _, filenames in os.walk(folder)
        for f in filenames
        if f.endswith(".json")
    )
    if not files:
        print(f"No JSON files found in {folder}", file=sys.stderr)
        sys.exit(1)

    from collections import defaultdict
    dir_results = defaultdict(list)

    for filepath in files:
        relpath = os.path.relpath(filepath, folder)
        reldir = os.path.dirname(relpath)
        try:
            row = process_file(filepath)
            if row:
                dir_results[reldir].append(row)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: skipping {relpath}: {e}", file=sys.stderr)

    rows = []
    for dirpath in sorted(dir_results.keys()):
        agg = aggregate_directory(dir_results[dirpath])
        agg["directory"] = dirpath
        rows.append(agg)

    fieldnames = [
        "directory", "n", "passed", "pass@1",
        "avg_duration_s", "avg_cost",
        "avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens",
        "avg_turns", "avg_ai_messages", "avg_knowledge_calls",
        "avg_max_msg_tokens",
    ]

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.output:
        out.close()
        print(f"Report written to {args.output} ({len(rows)} directories)", file=sys.stderr)


if __name__ == "__main__":
    main()
