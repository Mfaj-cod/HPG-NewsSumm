import argparse
import json
import math
import os
import random
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.HPG import HierarchicalPlannerGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data)}")
    return data


def split_data(
    data: List[Dict], val_split: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    if len(data) < 2 or val_split <= 0:
        return data, []

    idx = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_size = max(1, int(len(data) * val_split))
    val_idx = set(idx[:val_size])

    train_data = [data[i] for i in range(len(data)) if i not in val_idx]
    val_data = [data[i] for i in range(len(data)) if i in val_idx]
    return train_data, val_data


class NewsSumDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_input_length: int,
        max_target_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        doc_text = " ".join(ex["documents"])
        ref_text = ex["summary"]

        x = self.tokenizer(
            doc_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        y = self.tokenizer(
            text_target=ref_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = y["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels": labels,
        }


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> float:
    if len(loader) == 0:
        return float("nan")

    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                out = model(**batch)
                loss = out["loss"]

            total_loss += float(loss.detach().cpu())
            steps += 1

    return total_loss / max(steps, 1)


def save_checkpoint(model, tokenizer, out_dir: str) -> str:
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.generator.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir


def run_evaluation_with_existing_script(
    run_dir: str,
    data_path: str,
    eval_sample: int,
    max_input_length: int,
    max_target_length: int,
) -> None:
    eval_script = os.path.join(ROOT_DIR, "scripts", "run_evaluation.py")
    cmd = [
        sys.executable,
        eval_script,
        "--run_dir",
        run_dir,
        "--data",
        data_path,
        "--sample",
        str(eval_sample),
        "--max_input_length",
        str(max_input_length),
        "--max_target_length",
        str(max_target_length),
    ]
    print("\nRunning evaluation with scripts/run_evaluation.py ...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_data = load_json(args.data)
    if args.sample > 0:
        all_data = all_data[: args.sample]

    train_data, val_data = split_data(all_data, args.val_split, args.seed)
    if len(train_data) == 0:
        raise ValueError("Training split is empty. Increase sample size or reduce val_split.")

    out_dir = os.path.join(args.output_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = HierarchicalPlannerGenerator(
        base_model_name=args.base_model,
        num_segments=args.num_segments,
        num_plan_tokens=args.num_plan_tokens,
        num_heads=args.num_heads,
        planner_layers=args.planner_layers,
        dropout=args.dropout,
        planner_entropy_weight=args.planner_entropy_weight,
        redundancy_weight=args.redundancy_weight,
    ).to(device)

    train_ds = NewsSumDataset(
        train_data,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    val_ds = NewsSumDataset(
        val_data,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = max(
        1, math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    history = []
    best_val = float("inf")
    best_epoch = -1

    print(f"Run directory: {out_dir}")
    print(f"Train size: {len(train_data)} | Val size: {len(val_data)}")
    print("Training HPG...")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', enabled=args.fp16):
                out = model(**batch)
                raw_loss = out["loss"]
                loss = raw_loss / max(args.gradient_accumulation_steps, 1)

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += float(raw_loss.detach().cpu())
            step_count += 1
            pbar.set_postfix(loss=f"{epoch_loss / max(step_count, 1):.4f}")

        # Flush leftover gradients when number of steps is not divisible by grad accumulation.
        if step_count % max(args.gradient_accumulation_steps, 1) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        train_loss = epoch_loss / max(step_count, 1)
        val_loss = evaluate_loss(model, val_loader, device, args.fp16) if len(val_loader) > 0 else float("nan")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        should_save = True
        if not math.isnan(val_loss):
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
            else:
                should_save = not args.save_best_only
        elif args.save_best_only and epoch != args.epochs - 1:
            should_save = False

        if should_save:
            save_checkpoint(model, tokenizer, out_dir)

    # Ensure there is always a checkpoint.
    if not os.path.exists(os.path.join(out_dir, "checkpoint")):
        save_checkpoint(model, tokenizer, out_dir)

    summary = {
        "run_name": args.run_name,
        "base_model": args.base_model,
        "data_path": args.data,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "best_val_loss": None if math.isinf(best_val) else best_val,
        "best_epoch": best_epoch,
        "history": history,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved checkpoint to: {os.path.join(out_dir, 'checkpoint')}")
    print(f"Saved summary to: {os.path.join(out_dir, 'summary.json')}")

    if not args.skip_eval:
        eval_data = args.eval_data if args.eval_data else args.data
        run_evaluation_with_existing_script(
            run_dir=out_dir,
            data_path=eval_data,
            eval_sample=args.eval_sample,
            max_input_length=args.eval_max_input_length,
            max_target_length=args.eval_max_target_length,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enhanced HPG model.")
    parser.add_argument("--data", default="data/enhanced/newssumm_enhanced.json")
    parser.add_argument("--run_name", default="hpg_v2_run_001")
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--base_model", default="allenai/PRIMERA")

    parser.add_argument("--sample", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--val_split", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_best_only", action="store_true")

    # HPG-specific knobs
    parser.add_argument("--num_segments", type=int, default=16)
    parser.add_argument("--num_plan_tokens", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--planner_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--planner_entropy_weight", type=float, default=0.01)
    parser.add_argument("--redundancy_weight", type=float, default=0.03)

    # Evaluation controls
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_data", default="")
    parser.add_argument(
        "--eval_sample",
        type=int,
        default=0,
        help="Passed to run_evaluation.py. Use 0 for full evaluation.",
    )
    parser.add_argument("--eval_max_input_length", type=int, default=4096)
    parser.add_argument("--eval_max_target_length", type=int, default=256)

    args = parser.parse_args()
    main(args)

