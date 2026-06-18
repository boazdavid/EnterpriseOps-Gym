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
    costs = []

    for run in runs:
        durations.append(run.get("execution_time_ms", 0) / 1000.0)

        conv = run.get("conversation_flow", [])
        messages_list.append(len(conv))

        user_turns = sum(1 for m in conv if m.get("type") == "user_message")
        turns_list.append(user_turns)

        run_prompt = 0
        run_completion = 0
        run_total = 0
        run_cost = 0.0

        for msg in conv:
            if msg.get("type") == "ai_message":
                usage = msg.get("usage_metadata", {})
                pt = usage.get("input_tokens", 0)
                ct = usage.get("output_tokens", 0)
                tt = usage.get("total_tokens", 0)
                run_prompt += pt
                run_completion += ct
                run_total += tt

                resp_meta = msg.get("response_metadata", {})
                token_usage = resp_meta.get("token_usage", {})
                cached = token_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                non_cached_prompt = pt - cached

                # Default pricing (GPT-5 approximate): $2/1M input, $8/1M cached input, $10/1M output
                cost = (non_cached_prompt * 2.0 + cached * 8.0 + ct * 10.0) / 1_000_000
                run_cost += cost

        prompt_tokens_list.append(run_prompt)
        completion_tokens_list.append(run_completion)
        total_tokens_list.append(run_total)
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
        "avg_messages": avg(messages_list),
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
        "avg_messages": weighted_avg("avg_messages"),
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
        "avg_turns", "avg_messages",
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
