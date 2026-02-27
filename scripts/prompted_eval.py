import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

import evaluate

HF_TOKEN=os.getenv("HF_TOKEN")

# Prompt template
PROMPT_TEMPLATE = """You are a professional news editor.
Write a concise, factual summary of the following news articles.

Articles:
{docs}

Summary:
"""


# Load model safely (works for both decoder + seq2seq)
def load_model(model_name):

    token = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            token=token,
            device_map="auto"
        )
        is_seq2seq = False
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            token=token,
            device_map="auto"
        )
        is_seq2seq = True

    model.eval()

    return tokenizer, model, is_seq2seq


# Generate summary
@torch.no_grad()
def generate_summary(texts, tokenizer, model, max_new_tokens=128):

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



def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, is_seq2seq = load_model(args.model, device)

    with open(args.data) as f:
        dataset = json.load(f)

    if args.sample:
        dataset = dataset[:args.sample]
        print(f"Using subset of {len(dataset)} samples")


    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    preds = []
    refs = []

    results = []

    for item in tqdm(dataset):

        docs = "\n\n".join(item["documents"])
        prompt = PROMPT_TEMPLATE.format(docs=docs)

        summary = generate_summary([prompt], tokenizer, model, is_seq2seq)[0]

        preds.append(summary)
        refs.append(item["summary"])

        results.append({
            "id": item.get("id", ""),
            "prediction": summary,
            "reference": item["summary"]
        })

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "summaries.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Metrics
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")

    final_scores = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore": sum(bert_scores["f1"]) / len(bert_scores["f1"])
    }

    with open(os.path.join(args.out_dir, "evaluation.json"), "w") as f:
        json.dump(final_scores, f, indent=2)

    print("\nFinal scores:")
    print(final_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample", type=int, default=None, help="Use only first N samples for quick evaluation")
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    main(args)

