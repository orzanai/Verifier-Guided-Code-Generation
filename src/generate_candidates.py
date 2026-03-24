# src/generate_candidates.py
from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    build_chat_prompt_for_qwen,
    ensure_dir,
    extract_code_from_generation,
    load_humaneval,
    load_yaml,
    set_global_seed,
    write_jsonl,
)


def pick_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def normalize_device_map(device_map_value: Any) -> Optional[str]:
    if device_map_value is None:
        return None
    if isinstance(device_map_value, str):
        v = device_map_value.strip().lower()
        if v in ("none", "null"):
            return None
        return device_map_value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    bench = cfg["benchmark"]
    tasks = load_humaneval(
        max_tasks=bench.get("max_tasks", None),
        task_indices=bench.get("task_indices", None),
    )

    out_dir = cfg["paths"]["candidates_dir"]
    ensure_dir(out_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"humaneval_candidates_{timestamp}.jsonl")

    gen_cfg = cfg["generator"]
    model_name = gen_cfg["model_name"]
    device_map = normalize_device_map(gen_cfg.get("device_map", "auto"))
    torch_dtype = pick_torch_dtype(gen_cfg.get("torch_dtype", "float16"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if device_map is None:
        # load normally then move to GPU (more stable than offloading if it fits)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        # let HF handle placement / potential offload
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    model.eval()

    dec = cfg["decoding"]
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    n_candidates = int(cfg["sampling"]["n_candidates"])

    rows: List[Dict[str, Any]] = []

    for task in tqdm(tasks, desc="Generating candidates"):
        task_id = task["task_id"]
        prompt = task["prompt"]

        system_msg, user_msg = build_chat_prompt_for_qwen(prompt)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # apply_chat_template may return a Tensor OR a BatchEncoding (dict-like)
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # works for dict and transformers.BatchEncoding
        if hasattr(model_inputs, "get") and "input_ids" in model_inputs:
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask", None)
        else:
            input_ids = model_inputs
            attention_mask = None

        # hard safety checks: generate() needs tensors here
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids is not a torch.Tensor (got {type(input_ids)}).")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError(f"attention_mask is not a torch.Tensor (got {type(attention_mask)}).")

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        for cand_id in range(n_candidates):
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=bool(dec.get("do_sample", True)),
                    temperature=float(dec.get("temperature", 0.6)),
                    top_p=float(dec.get("top_p", 0.95)),
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = out[0, input_ids.size(1) :]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            code = extract_code_from_generation(gen_text)

            rows.append(
                {
                    "benchmark": "HumanEval",
                    "task_id": task_id,
                    "cand_id": cand_id,
                    "n_candidates": n_candidates,
                    "temperature": float(dec.get("temperature", 0.6)),
                    "top_p": float(dec.get("top_p", 0.95)),
                    "max_new_tokens": max_new_tokens,
                    "model_name": model_name,
                    "code": code,
                }
            )

    write_jsonl(out_path, rows)
    print(f"[OK] Wrote {len(rows)} candidates to: {out_path}")


if __name__ == "__main__":
    main()