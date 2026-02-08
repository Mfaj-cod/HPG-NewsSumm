import os
import sys
import json
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.novel_model import HierarchicalPlannerGenerator


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


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)

    run_name = config["experiment"]["name"]
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    fp16 = config["training"].get("fp16", False)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    model = HierarchicalPlannerGenerator(
        base_model_name=config["model"]["base_model"]
    ).to(device)

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

    print("Training Novel model...")

    for step, batch in enumerate(tqdm(loader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=fp16):
            out = model(**batch)
            loss = out["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.generator.model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    print("Novel training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    main(args)
