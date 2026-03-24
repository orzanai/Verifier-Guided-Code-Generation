# src/train_ranker.py
from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.utils import ensure_dir, load_yaml, read_jsonl


class PairwiseDataset(Dataset):
    def __init__(self, path: str):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {"a_text": r["a_text"], "b_text": r["b_text"]}


@dataclass
class Batch:
    a_input_ids: torch.Tensor
    a_attention_mask: torch.Tensor
    b_input_ids: torch.Tensor
    b_attention_mask: torch.Tensor


def collate_fn(tokenizer: AutoTokenizer, max_length: int):
    def _collate(examples: List[Dict[str, Any]]) -> Batch:
        a = tokenizer([ex["a_text"] for ex in examples], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
        b = tokenizer([ex["b_text"] for ex in examples], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
        return Batch(a["input_ids"], a["attention_mask"], b["input_ids"], b["attention_mask"])
    return _collate


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


def pairwise_loss(sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    # -log sigmoid(sa - sb)
    return -torch.log(torch.sigmoid(sa - sb) + 1e-8).mean()


@torch.no_grad()
def eval_pairwise(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, correct = 0, 0
    for batch in loader:
        sa = model(batch.a_input_ids.to(device), batch.a_attention_mask.to(device))
        sb = model(batch.b_input_ids.to(device), batch.b_attention_mask.to(device))
        correct += (sa > sb).sum().item()
        total += sa.size(0)
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--encoder_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_root = cfg["paths"]["ranker_data_dir"]
    ensure_dir(out_root)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"ranker_model_{ts}")
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    train_ds = PairwiseDataset(args.train_path)
    val_ds = PairwiseDataset(args.val_path)
    test_ds = PairwiseDataset(args.test_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn(tokenizer, args.max_length))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn(tokenizer, args.max_length))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn(tokenizer, args.max_length))

    model = Ranker(args.encoder_name, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    best_val = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                sa = model(batch.a_input_ids.to(device), batch.a_attention_mask.to(device))
                sb = model(batch.b_input_ids.to(device), batch.b_attention_mask.to(device))
                loss = pairwise_loss(sa, sb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_acc = eval_pairwise(model, val_loader, device)
        print(f"[VAL] epoch={epoch} acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            torch.save(
                {"encoder_name": args.encoder_name, "state_dict": model.state_dict(), "max_length": args.max_length},
                best_path,
            )
            print(f"[OK] New best saved: {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("[EARLY STOP] No improvement on validation.")
                break

    # test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_acc = eval_pairwise(model, test_loader, device)
    print(f"[TEST] acc={test_acc:.3f}")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_val_acc={best_val}\n")
        f.write(f"test_pairwise_acc={test_acc}\n")
        f.write(f"train_path={args.train_path}\nval_path={args.val_path}\ntest_path={args.test_path}\n")

    print(f"[OK] Wrote summary: {os.path.join(out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()