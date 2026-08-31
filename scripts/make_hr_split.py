#!/usr/bin/env python3
"""Generate a deterministic train/test split of HR task IDs.

Writes a JSON file with, for each seed, a 50/50 train/test partition of the HR
task_ids (task_ids are shared across tool-set modes, so one split serves oracle /
plus_N alike). Consumed by evaluate.py via --split_file / --split / --seed.

Usage:
    python scripts/make_hr_split.py                 # HR, seeds 0-2, 50/50
    python scripts/make_hr_split.py --domain hr --seeds 0 1 2 --train_frac 0.5 \
        --out data/splits/hr_train_test_split.json
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default="ServiceNow-AI/EnterpriseOps-Gym")
    ap.add_argument("--domain", default="hr")
    ap.add_argument("--mode", default="oracle",
                    help="Config to read task_ids from (task_ids are shared across modes).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--out", default="data/splits/hr_train_test_split.json")
    args = ap.parse_args()

    ds = load_dataset(args.hf_dataset, args.mode, split=args.domain)
    # Sort for a stable base ordering independent of dataset row order.
    task_ids = sorted({row["task_id"] for row in ds})
    n = len(task_ids)
    n_train = round(n * args.train_frac)

    seeds = {}
    for seed in args.seeds:
        shuffled = list(task_ids)
        random.Random(seed).shuffle(shuffled)
        train = sorted(shuffled[:n_train])
        test = sorted(shuffled[n_train:])
        assert set(train).isdisjoint(test)
        assert len(train) + len(test) == n
        seeds[str(seed)] = {"train": train, "test": test}

    out = {
        "hf_dataset": args.hf_dataset,
        "domain": args.domain,
        "source_mode": args.mode,
        "num_tasks": n,
        "train_frac": args.train_frac,
        "n_train": n_train,
        "n_test": n - n_train,
        "seeds": seeds,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}: {n} HR tasks -> {n_train} train / {n - n_train} test "
          f"per seed {args.seeds}")


if __name__ == "__main__":
    main()
