# src/evaluate_ranker_selection.py
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import load_humaneval, load_yaml, read_jsonl


def build_prompt_map() -> Dict[str, str]:
    tasks = load_humaneval(max_tasks=None, task_indices=None)
    return {t["task_id"]: t["prompt"] for t in tasks}


def group_candidates(cand_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_task = defaultdict(list)
    for r in cand_rows:
        by_task[r["task_id"]].append(r)
    for t in list(by_task.keys()):
        by_task[t] = sorted(by_task[t], key=lambda x: int(x["cand_id"]))
    return by_task


def build_pass_map(test_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], bool]:
    m = {}
    for r in test_rows:
        m[(r["task_id"], int(r["cand_id"]))] = bool(r["passed"])
    return m


def make_text_input(prompt: str, code: str) -> str:
    return f"[PROMPT]\n{prompt}\n\n[CODE]\n{code}\n"


class Ranker(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.scorer(cls).squeeze(-1)


@torch.no_grad()
def score_candidates(
    model: Ranker,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    candidates: List[Dict[str, Any]],
    max_length: int,
    batch_size: int = 8,
) -> List[float]:
    texts = [make_text_input(prompt, c["code"]) for c in candidates]
    scores: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        s = model(enc["input_ids"], enc["attention_mask"])
        scores.extend(s.detach().cpu().tolist())

    return scores


def pass_at_1_first(tasks: List[str], pass_map: Dict[Tuple[str, int], bool]) -> float:
    return sum(1 for t in tasks if pass_map.get((t, 0), False)) / len(tasks)


def pass_at_1_best_of_n(tasks: List[str], by_task: Dict[str, List[Dict[str, Any]]], pass_map: Dict[Tuple[str, int], bool]) -> float:
    solved = 0
    for t in tasks:
        cands = by_task[t]
        if any(pass_map.get((t, int(c["cand_id"])), False) for c in cands):
            solved += 1
    return solved / len(tasks)


def pass_at_1_random(tasks: List[str], by_task: Dict[str, List[Dict[str, Any]]], pass_map: Dict[Tuple[str, int], bool], seed: int) -> float:
    rng = random.Random(seed)
    solved = 0
    for t in tasks:
        cands = by_task[t]
        pick = rng.randrange(len(cands))
        cid = int(cands[pick]["cand_id"])
        if pass_map.get((t, cid), False):
            solved += 1
    return solved / len(tasks)


def eval_ranker_top1(tasks: List[str], by_task: Dict[str, List[Dict[str, Any]]], pass_map: Dict[Tuple[str, int], bool],
                     prompt_map: Dict[str, str], model: Ranker, tokenizer: AutoTokenizer, device: torch.device,
                     max_length: int) -> float:
    solved = 0
    for t in tqdm(tasks, desc="Ranker top-1"):
        cands = by_task[t]
        scores = score_candidates(model, tokenizer, device, prompt_map[t], cands, max_length=max_length)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        cid = int(cands[best_idx]["cand_id"])
        if pass_map.get((t, cid), False):
            solved += 1
    return solved / len(tasks)


def eval_ranker_topk_tests(tasks: List[str], by_task: Dict[str, List[Dict[str, Any]]], pass_map: Dict[Tuple[str, int], bool],
                           prompt_map: Dict[str, str], model: Ranker, tokenizer: AutoTokenizer, device: torch.device,
                           max_length: int, k: int) -> Tuple[float, float]:
    solved = 0
    tests_used = 0

    for t in tqdm(tasks, desc=f"Ranker top-{k}+tests"):
        cands = by_task[t]
        scores = score_candidates(model, tokenizer, device, prompt_map[t], cands, max_length=max_length)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kk = min(k, len(order))
        tests_used += kk

        chosen_cid = int(cands[order[0]]["cand_id"])  # fallback
        for idx in order[:kk]:
            cid = int(cands[idx]["cand_id"])
            if pass_map.get((t, cid), False):
                chosen_cid = cid
                break

        if pass_map.get((t, chosen_cid), False):
            solved += 1

    return solved / len(tasks), tests_used / len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--tests", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--encoder_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--k_list", type=str, default="1,2,4")
    parser.add_argument("--random_seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    _ = load_yaml(args.config)  # kept for consistency

    cand_rows = read_jsonl(args.candidates)
    test_rows = read_jsonl(args.tests)

    by_task = group_candidates(cand_rows)
    pass_map = build_pass_map(test_rows)
    prompt_map = build_prompt_map()

    tasks = sorted(by_task.keys())

    # load ranker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = Ranker(args.encoder_name).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # baselines 
    first = pass_at_1_first(tasks, pass_map)
    best = pass_at_1_best_of_n(tasks, by_task, pass_map)
    seeds = [int(x.strip()) for x in args.random_seeds.split(",") if x.strip()]
    rand_scores = [pass_at_1_random(tasks, by_task, pass_map, s) for s in seeds]
    rand_mean = sum(rand_scores) / len(rand_scores)

    print(f"Tasks evaluated: {len(tasks)}")
    print(f"N candidates per task: {len(by_task[tasks[0]])}")
    print(f"pass@1 first-sample: {first:.3f}")
    print(f"pass@1 random mean: {rand_mean:.3f} (scores={rand_scores})")
    print(f"pass@1 test-all best-of-N: {best:.3f}")

    # Ranker methods
    top1 = eval_ranker_top1(tasks, by_task, pass_map, prompt_map, model, tokenizer, device, args.max_length)
    print(f"pass@1 ranker top-1: {top1:.3f}")

    for k in [int(x.strip()) for x in args.k_list.split(",") if x.strip()]:
        p1, avg_tests = eval_ranker_topk_tests(tasks, by_task, pass_map, prompt_map, model, tokenizer, device, args.max_length, k=k)
        print(f"pass@1 ranker top-{k}+tests: {p1:.3f} | avg tests used per task: {avg_tests:.2f}")


if __name__ == "__main__":
    main()