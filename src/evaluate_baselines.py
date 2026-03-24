# src/evaluate_baselines.py
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.utils import ensure_dir, load_yaml, read_jsonl


def index_tests_by_task(test_rows: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    tests: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in test_rows:
        tests[r["task_id"]][int(r["cand_id"])] = r
    return tests


def index_candidates_by_task(cand_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    cands: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in cand_rows:
        cands[r["task_id"]].append(r)
    # sort by cand_id for stability
    for k in list(cands.keys()):
        cands[k] = sorted(cands[k], key=lambda x: int(x["cand_id"]))
    return cands


def pass_at_1_first_sample(tasks: List[str], tests: Dict[str, Dict[int, Dict[str, Any]]]) -> float:
    solved = 0
    for t in tasks:
        r0 = tests[t].get(0)
        if r0 is not None and bool(r0["passed"]):
            solved += 1
    return solved / len(tasks) if tasks else 0.0


def pass_at_1_best_of_n(tasks: List[str], tests: Dict[str, Dict[int, Dict[str, Any]]]) -> float:
    solved = 0
    for t in tasks:
        if any(bool(r["passed"]) for r in tests[t].values()):
            solved += 1
    return solved / len(tasks) if tasks else 0.0


def pass_at_1_random(
    tasks: List[str],
    tests: Dict[str, Dict[int, Dict[str, Any]]],
    n_candidates: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    solved = 0
    for t in tasks:
        cid = rng.randrange(n_candidates)
        r = tests[t].get(cid)
        if r is not None and bool(r["passed"]):
            solved += 1
    return solved / len(tasks) if tasks else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--tests", type=str, required=True)
    parser.add_argument("--random_seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = cfg["paths"]["results_dir"]
    ensure_dir(out_dir)

    cand_rows = read_jsonl(args.candidates)
    test_rows = read_jsonl(args.tests)

    cands_by_task = index_candidates_by_task(cand_rows)
    tests_by_task = index_tests_by_task(test_rows)

    tasks = sorted(set(cands_by_task.keys()))
    if not tasks:
        raise ValueError("No tasks found in candidates file.")

    n_candidates = int(cand_rows[0].get("n_candidates", 1))

    # baselines
    first_p1 = pass_at_1_first_sample(tasks, tests_by_task)
    best_p1 = pass_at_1_best_of_n(tasks, tests_by_task)

    seeds = [int(s.strip()) for s in args.random_seeds.split(",") if s.strip()]
    rand_scores = [pass_at_1_random(tasks, tests_by_task, n_candidates, s) for s in seeds]
    rand_mean = sum(rand_scores) / len(rand_scores) if rand_scores else 0.0

    # task difficulty stats
    pass_fractions: List[float] = []
    for t in tasks:
        passed = sum(1 for r in tests_by_task[t].values() if bool(r["passed"]))
        pass_fractions.append(passed / n_candidates)

    avg_pass_frac = sum(pass_fractions) / len(pass_fractions) if pass_fractions else 0.0

    print(f"Tasks evaluated: {len(tasks)}")
    print(f"N candidates per task: {n_candidates}")
    print(f"pass@1 (first-sample): {first_p1:.3f}")
    print(f"pass@1 (random, mean over {len(seeds)} seeds): {rand_mean:.3f}  (scores={rand_scores})")
    print(f"pass@1 (test-all best-of-N): {best_p1:.3f}")
    print(f"Avg pass fraction per task (passed/N): {avg_pass_frac:.3f}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"baselines_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"tasks,{len(tasks)}\n")
        f.write(f"n_candidates,{n_candidates}\n")
        f.write(f"pass_at_1_first_sample,{first_p1}\n")
        f.write(f"pass_at_1_random_mean,{rand_mean}\n")
        f.write(f"pass_at_1_best_of_n,{best_p1}\n")
        f.write(f"avg_pass_fraction,{avg_pass_frac}\n")

    print(f"[OK] Wrote baseline summary to: {csv_path}")


if __name__ == "__main__":
    main()