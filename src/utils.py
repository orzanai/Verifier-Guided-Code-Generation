# src/utils.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_humaneval(max_tasks: Optional[int], task_indices: Optional[List[int]]) -> List[Dict[str, Any]]:
    """
    Loads HumanEval tasks via HuggingFace datasets.

    Each item contains:
      - task_id
      - prompt (includes function signature + docstring)
      - canonical_solution
      - test
      - entry_point
    """
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval")["test"]  # standard HF mirror
    items = [ds[i] for i in range(len(ds))]

    if task_indices is not None and len(task_indices) > 0:
        items = [items[i] for i in task_indices]

    if max_tasks is not None:
        items = items[: max_tasks]

    return items


def build_chat_prompt_for_qwen(prompt: str) -> str:
    """
    Qwen Instruct models expect a chat format. We'll keep it minimal:
    system: strict instruction about output format
    user: the HumanEval prompt
    """
    system = (
        "You are a helpful assistant that writes correct Python code.\n"
        "Return only the completed Python code. Do not add explanations."
    )
    user = (
        "Complete the following Python function. "
        "Follow the specification in the docstring.\n\n"
        f"{prompt}\n"
    )
    # Transformers will apply chat template if we pass messages, but for simplicity,
    # we can pass messages separately in generate_candidates.py.
    return system, user


def extract_code_from_generation(text: str) -> str:
    """
    Many instruct models wrap code in markdown fences. Strip them if present.
    Keep it conservative: if fences exist, take the first fenced block.
    """
    t = text.strip()

    # Common pattern: ```python ... ```
    if "```" in t:
        parts = t.split("```")
        # parts: [before, lang+code, after, ...]
        # pick the first fenced block content after the opening ```
        if len(parts) >= 2:
            fenced = parts[1]
            # remove possible "python\n"
            fenced = fenced.lstrip()
            if fenced.lower().startswith("python"):
                fenced = fenced[len("python"):].lstrip("\n")
            return fenced.strip()

    return t