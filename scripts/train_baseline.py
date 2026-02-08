# This script trains a model on a single sample from the NewsSumm dataset.

import os
import json
import yaml
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


# Dataset
class NewsSumDataset(Dataset):
    def __init__(self, path, tokenizer, max_in, max_out, sample=None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if sample:
            data = data[:sample]

        self.docs = [" ".join(x["documents"]) for x in data]
        self.sums = [x["summary"] for x in data]

        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        x = self.tokenizer(
            self.docs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_in,
            return_tensors="pt"
        )

        y = self.tokenizer(
            text_target=self.sums[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_out,
            return_tensors="pt"
        )

        item = {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels": y["input_ids"].squeeze(0)
        }

        item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100
        return item


# Utils
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# Training
def main(args):
    config = load_config(args.config)

    run_name = config["experiment"]["name"]
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Saving to:", out_dir)

    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    fp16 = config["training"].get("fp16", False)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"]).to(device)

    dataset = NewsSumDataset(
        config["data"]["train_file"],
        tokenizer,
        config["data"]["max_input_length"],
        config["data"]["max_target_length"],
        sample=args.sample
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    model.train()
    optimizer.zero_grad()

    print("Training...")

    for step, batch in enumerate(tqdm(loader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=fp16):
            out = model(**batch)
            loss = out.loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    # Save checkpoint
    ckpt = os.path.join(out_dir, "checkpoint")
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"loss": float(loss)}, f)

    print("Baseline training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    main(args)
