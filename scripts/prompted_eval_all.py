import os
import json
import torch
from tqdm import tqdm
import evaluate
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# CONFIG
DATA_PATH = "/kaggle/input/newssumm-data/newssumm_processed.json"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3", # done
    "meta-llama/Meta-Llama-3-8B-Instruct", # restricted
    "google/flan-t5-xl", # done
    "google/flan-t5-xxl", # done
    "google/gemma-2-9b-it", # restricted
    "Qwen/Qwen2-7B-Instruct", # done
    "mistralai/Mixtral-8x7B-Instruct-v0.1", # done
]

BATCH_SIZE = 6           # can increase if GPU has memory
MAX_NEW_TOKENS = 128

PROMPT_TEMPLATE = """You are a professional news editor.
Write a concise, factual summary of the following news articles.

Articles:
{docs}

Summary:
"""


# Dataset streaming loader (memory safe)
def stream_json(path):
    with open(path) as f:
        data = json.load(f)
    for item in data:
        yield item


# Load model automatically (seq2seq or decoder)
def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        is_seq2seq = False
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        is_seq2seq = True

    model.eval()

    return tokenizer, model, is_seq2seq


# Batched generation
@torch.no_grad()
def generate_batch(texts, tokenizer, model, max_new_tokens):

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Evaluate single model
def evaluate_model(model_name, sample=None):

    print(f"\n{'='*60}")
    print(f"Running: {model_name}")
    print(f"{'='*60}")

    tokenizer, model, _ = load_model(model_name)

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    preds = []
    refs = []
    summaries = []

    batch_texts = []
    batch_refs = []

    for idx, item in enumerate(tqdm(stream_json(DATA_PATH))):

        if sample and idx >= sample:
            break

        docs = "\n\n".join(item["documents"])
        prompt = PROMPT_TEMPLATE.format(docs=docs)

        batch_texts.append(prompt)
        batch_refs.append(item["summary"])

        if len(batch_texts) == BATCH_SIZE:

            outs = generate_batch(
                batch_texts, tokenizer, model, MAX_NEW_TOKENS
            )

            preds.extend(outs)
            refs.extend(batch_refs)

            for p, r in zip(outs, batch_refs):
                summaries.append({"prediction": p, "reference": r})

            batch_texts, batch_refs = [], []

    # last batch
    if batch_texts:
        outs = generate_batch(batch_texts, tokenizer, model, MAX_NEW_TOKENS)
        preds.extend(outs)
        refs.extend(batch_refs)

    # metrics
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")

    final = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore": sum(bert_scores["f1"]) / len(bert_scores["f1"])
    }

    # save
    safe_name = model_name.split("/")[-1]
    out_dir = f"results/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open(f"{out_dir}/evaluation.json", "w") as f:
        json.dump(final, f, indent=2)

    print("Scores:", final)

    del model
    torch.cuda.empty_cache()


# AUTO RUN ALL MODELS
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    for m in MODELS:
        evaluate_model(m, args.sample)

    print("\n All models finished. Go drink your tea.")
