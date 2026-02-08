import argparse
import json
import os
from langsmith import evaluate
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import evaluate as hf_evaluate

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = args.run_dir
    ckpt_dir = os.path.abspath(os.path.join(run_dir, "checkpoint"))

    print(f"Loading model from {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device)
    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.sample:
        data = data[:args.sample]

    rouge = hf_evaluate.load("rouge")

    predictions = []
    references = []

    for ex in tqdm(data, desc="Generating summaries"):
        # Join multi-document cluster into a single input
        article_text = " ".join(ex["documents"])

        inputs = tokenizer(
            article_text,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt"
        ).to(device)


        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=args.max_target_length,
                num_beams=4,
                early_stopping=True
            )

        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        ref = ex["summary"]

        predictions.append(pred)
        references.append(ref)

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references
    )

    try:
        bertscore = hf_evaluate.load("bertscore")
        bert_scores = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        bert_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])
    except Exception as e:
        print("BERTScore failed:", e)
        bert_f1 = 0.0

    results = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore": sum(bert_scores["f1"]) / len(bert_scores["f1"])
    }

    print("\nEvaluation results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    out_path = os.path.join(run_dir, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_target_length", type=int, default=256)
    args = parser.parse_args()

    main(args)
