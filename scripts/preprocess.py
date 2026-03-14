# This script preprocesses raw NewsSumm dataset files into a structured JSON format.
import argparse
import json
import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Removing HTML
    text = BeautifulSoup(text, "lxml").get_text()

    # Normalizing whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main(args):
    os.makedirs(args.output, exist_ok=True)
    print("Loading raw dataset...")

    # support XLSX or JSON
    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input)

        processed = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            article = clean_text(str(row.get("article_text", "")))
            summary = clean_text(str(row.get("human_summary", "")))

            if not article or not summary:
                continue

            sample = {
                "cluster_id": f"cluster_{idx}",
                "documents": [article],
                "summary": summary,
                "metadata": {
                    "source": row.get("newspaper_name", ""),
                    "date": row.get("published_date", ""),
                    "title": row.get("headline", ""),
                    "category": row.get("news_category", "")

                }
            }
            processed.append(sample)
    else:
        raise ValueError("Only XLSX input is supported.")

    out_path = os.path.join(args.output, "newssumm_processed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(processed)} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset (xlsx or json)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    args = parser.parse_args()
    main(args)
