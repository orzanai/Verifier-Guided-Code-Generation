# src/run_tests.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

from src.utils import ensure_dir, load_humaneval, load_yaml, read_jsonl, write_jsonl


def build_task_map() -> Dict[str, Dict[str, Any]]:
    tasks = load_humaneval(max_tasks=None, task_indices=None)
    return {t["task_id"]: t for t in tasks}


def candidate_defines_entrypoint(code: str, entry_point: str) -> bool:
    pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code, flags=re.MULTILINE) is not None


def make_program(task: Dict[str, Any], cand_code: str) -> str:
    entry_point = task["entry_point"]
    prompt = task["prompt"]

    if candidate_defines_entrypoint(cand_code, entry_point):
        return cand_code.strip() + "\n"
    else:
        return prompt.rstrip() + "\n" + cand_code.strip() + "\n"


def run_one_candidate(task: Dict[str, Any], program: str, timeout_s: float) -> Tuple[bool, str, int]:
    test_code = task["test"]
    entry_point = task["entry_point"]
    runner = f"""
# --- candidate program ---
{program}
# --- tests ---
{test_code}
# --- run ---
check({entry_point})
"""

    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "run.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(runner)

        try:
            completed = subprocess.run(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                text=True,
            )
            runtime_ms = int((time.time() - start) * 1000)

            if completed.returncode == 0:
                return True, "pass", runtime_ms

            err = (completed.stderr or "").lower()
            if "syntaxerror" in err:
                return False, "syntax_error", runtime_ms
            return False, "exception", runtime_ms

        except subprocess.TimeoutExpired:
            runtime_ms = int((time.time() - start) * 1000)
            return False, "timeout", runtime_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates JSONL")
    parser.add_argument("--timeout_s", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_map = build_task_map()

    cand_rows = read_jsonl(args.candidates)

    out_dir = cfg["paths"]["tests_dir"]
    ensure_dir(out_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"humaneval_tests_{timestamp}.jsonl")

    results: List[Dict[str, Any]] = []

    for r in cand_rows:
        task_id = r["task_id"]
        cand_id = r["cand_id"]
        cand_code = r["code"]

        if task_id not in task_map:
            results.append(
                {
                    "task_id": task_id,
                    "cand_id": cand_id,
                    "passed": False,
                    "error_type": "unknown_task_id",
                    "runtime_ms": 0,
                }
            )
            continue

        task = task_map[task_id]
        program = make_program(task, cand_code)

        passed, error_type, runtime_ms = run_one_candidate(task, program, timeout_s=float(args.timeout_s))

        results.append(
            {
                "task_id": task_id,
                "cand_id": cand_id,
                "passed": bool(passed),
                "error_type": error_type,
                "runtime_ms": int(runtime_ms),
            }
        )

    write_jsonl(out_path, results)
    print(f"[OK] Wrote {len(results)} test results to: {out_path}")


if __name__ == "__main__":
    main()