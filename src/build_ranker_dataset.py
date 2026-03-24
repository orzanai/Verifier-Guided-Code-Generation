# src/build_ranker_dataset.py
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.utils import ensure_dir, load_humaneval, load_yaml, read_jsonl, write_jsonl


def group_by_task(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def build_prompt_map() -> Dict[str, str]:
    tasks = load_humaneval(max_tasks=None, task_indices=None)
    return {t["task_id"]: t["prompt"] for t in tasks}


def make_text_input(prompt: str, code: str) -> str:
    # Cross-encoder input: task spec + candidate code
    return f"[PROMPT]\n{prompt}\n\n[CODE]\n{code}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--tests", type=str, required=True)

    parser.add_argument("--pairs_per_task", type=int, default=120)
    parser.add_argument("--train_frac", type=float, default=0.75)
    parser.add_argument("--val_frac", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rng = random.Random(args.seed)

    prompt_map = build_prompt_map()

    cand_rows = read_jsonl(args.candidates)
    test_rows = read_jsonl(args.tests)

    c_by_task = group_by_task(cand_rows, "task_id")

    passed_map: Dict[Tuple[str, int], bool] = {}
    for r in test_rows:
        passed_map[(r["task_id"], int(r["cand_id"]))] = bool(r["passed"])

    task_ids = sorted(c_by_task.keys())

    # split by task_id
    rng.shuffle(task_ids)
    n = len(task_ids)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_tasks = set(task_ids[:n_train])
    val_tasks = set(task_ids[n_train:n_train + n_val])
    test_tasks = set(task_ids[n_train + n_val:])

    def build_pairs(task_set: set) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        for tid in task_set:
            prompt = prompt_map.get(tid, "")
            cands = sorted(c_by_task[tid], key=lambda x: int(x["cand_id"]))

            pass_list = []
            fail_list = []
            for c in cands:
                cid = int(c["cand_id"])
                if passed_map.get((tid, cid), False):
                    pass_list.append(c)
                else:
                    fail_list.append(c)

            if len(pass_list) == 0 or len(fail_list) == 0:
                continue

            # sample capped pairs per task (with replacement)
            for _ in range(args.pairs_per_task):
                p = rng.choice(pass_list)
                f = rng.choice(fail_list)
                pairs.append(
                    {
                        "task_id": tid,
                        "a_text": make_text_input(prompt, p["code"]),
                        "b_text": make_text_input(prompt, f["code"]),
                        "label": 1,
                        "a_cand_id": int(p["cand_id"]),
                        "b_cand_id": int(f["cand_id"]),
                    }
                )
        return pairs

    train_pairs = build_pairs(train_tasks)
    val_pairs = build_pairs(val_tasks)
    test_pairs = build_pairs(test_tasks)

    out_dir = cfg["paths"]["ranker_data_dir"]
    ensure_dir(out_dir)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    train_path = os.path.join(out_dir, f"pairs_prompt_train_{ts}.jsonl")
    val_path = os.path.join(out_dir, f"pairs_prompt_val_{ts}.jsonl")
    test_path = os.path.join(out_dir, f"pairs_prompt_test_{ts}.jsonl")

    write_jsonl(train_path, train_pairs)
    write_jsonl(val_path, val_pairs)
    write_jsonl(test_path, test_pairs)

    print(f"Tasks total: {n} | train={len(train_tasks)} val={len(val_tasks)} test={len(test_tasks)}")
    print(f"Pairs: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"[OK] Wrote:\n  {train_path}\n  {val_path}\n  {test_path}")


if __name__ == "__main__":
    main()