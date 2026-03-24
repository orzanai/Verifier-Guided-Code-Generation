# src/error_analysis_shortlist.py
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.utils import load_humaneval, read_jsonl


def build_prompt_map() -> Dict[str, str]:
    tasks = load_humaneval(max_tasks=None, task_indices=None)
    return {t["task_id"]: t["prompt"] for t in tasks}


def group_candidates(cand_rows):
    by_task = defaultdict(list)
    for r in cand_rows:
        by_task[r["task_id"]].append(r)
    for t in by_task:
        by_task[t] = sorted(by_task[t], key=lambda x: int(x["cand_id"]))
    return by_task


def build_pass_map(test_rows) -> Dict[Tuple[str, int], bool]:
    m = {}
    for r in test_rows:
        m[(r["task_id"], int(r["cand_id"]))] = bool(r["passed"])
    return m


def make_text(prompt: str, code: str) -> str:
    return f"[PROMPT]\n{prompt}\n\n[CODE]\n{code}\n"


class Ranker(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.scorer(cls).squeeze(-1)


@torch.no_grad()
def score_all(model, tok, device, prompt, cands, max_length=512, batch_size=8) -> List[float]:
    texts = [make_text(prompt, c["code"]) for c in cands]
    scores = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i+batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        s = model(enc["input_ids"], enc["attention_mask"])
        scores.extend(s.cpu().tolist())
    return scores


def simulate_topk(pass_map, task_id, cands, order, k: int) -> bool:
    kk = min(k, len(order))
    for idx in order[:kk]:
        cid = int(cands[idx]["cand_id"])
        if pass_map.get((task_id, cid), False):
            return True
    return False


def first_passing_rank(pass_map, task_id, cands, order) -> int:
    # returns 1-based position of first passing candidate in the ranked list, or 0 if none
    for pos, idx in enumerate(order, start=1):
        cid = int(cands[idx]["cand_id"])
        if pass_map.get((task_id, cid), False):
            return pos
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder_name", default="microsoft/codebert-base")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    cand_rows = read_jsonl(args.candidates)
    test_rows = read_jsonl(args.tests)

    by_task = group_candidates(cand_rows)
    pass_map = build_pass_map(test_rows)
    prompt_map = build_prompt_map()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.encoder_name)
    model = Ranker(args.encoder_name).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tasks = sorted(by_task.keys())

    no_pass = []
    miss_top1 = []
    miss_top4 = []

    for t in tasks:
        cands = by_task[t]
        any_pass = any(pass_map.get((t, int(c["cand_id"])), False) for c in cands)
        if not any_pass:
            no_pass.append(t)
            continue

        scores = score_all(model, tok, device, prompt_map[t], cands, max_length=args.max_length)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        top1_ok = simulate_topk(pass_map, t, cands, order, k=1)
        top4_ok = simulate_topk(pass_map, t, cands, order, k=4)

        if not top1_ok:
            miss_top1.append((t, first_passing_rank(pass_map, t, cands, order)))
        if not top4_ok:
            miss_top4.append((t, first_passing_rank(pass_map, t, cands, order)))

    print("No passing candidate (best-of-N fails):", len(no_pass))
    print("Best-of-N passes but ranker top-1 fails:", len(miss_top1))
    print("Best-of-N passes but ranker top-4 fails:", len(miss_top4))

    print("\n--- Missed by top-1 (task_id, first_passing_rank) ---")
    for t, r in miss_top1[:30]:
        print(t, r)

    print("\n--- Missed by top-4 (task_id, first_passing_rank) ---")
    for t, r in miss_top4[:30]:
        print(t, r)


if __name__ == "__main__":
    main()