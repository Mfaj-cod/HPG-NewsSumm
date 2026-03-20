import argparse
import json
import os
import subprocess
import sys
from typing import List, Tuple

import pandas as pd


def normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch == "_")


def resolve_columns(df: pd.DataFrame, article_col: str = "", summary_col: str = "") -> Tuple[str, str]:
    if article_col:
        if article_col not in df.columns:
            raise ValueError(f"article column not found: {article_col}")
    if summary_col:
        if summary_col not in df.columns:
            raise ValueError(f"summary column not found: {summary_col}")
    if article_col and summary_col:
        return article_col, summary_col

    normalized = {normalize_col_name(c): c for c in df.columns}
    article_candidates = [
        "article_text",
        "articletext",
        "article",
        "content",
        "story",
        "fulltext",
        "body",
        "text",
    ]
    summary_candidates = [
        "human_summary",
        "humansummary",
        "summary",
        "abstract",
        "highlights",
        "shortsummary",
    ]

    if not article_col:
        for cand in article_candidates:
            if cand in normalized:
                article_col = normalized[cand]
                break
    if not summary_col:
        for cand in summary_candidates:
            if cand in normalized:
                summary_col = normalized[cand]
                break

    if not article_col or not summary_col:
        raise ValueError(
            f"Could not resolve article/summary columns from: {list(df.columns)}. "
            "Pass --article_col and --summary_col explicitly."
        )

    return article_col, summary_col


def to_eval_records(df: pd.DataFrame, article_col: str, summary_col: str) -> List[dict]:
    records = []
    for _, row in df.iterrows():
        article = str(row[article_col]).strip()
        summary = str(row[summary_col]).strip()
        if not article or not summary:
            continue
        records.append(
            {
                "documents": [article],
                "summary": summary,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run model evaluation on NewsSumm_Cleaned.xlsx via scripts/run_evaluation.py"
    )
    parser.add_argument("--run_dir", required=True, help="results/<run_name> directory")
    parser.add_argument("--excel", default="data/NewsSumm_Cleaned.xlsx")
    parser.add_argument("--article_col", default="")
    parser.add_argument("--summary_col", default="")
    parser.add_argument("--sample", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--tmp_json", default="")
    parser.add_argument("--keep_tmp_json", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.excel):
        raise FileNotFoundError(f"Excel file not found: {args.excel}")
    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    print(f"Loading Excel: {args.excel}")
    df = pd.read_excel(args.excel, engine="openpyxl")
    article_col, summary_col = resolve_columns(df, args.article_col, args.summary_col)

    df = df.dropna(subset=[article_col, summary_col]).reset_index(drop=True)
    if args.sample > 0:
        df = df.head(args.sample)

    eval_records = to_eval_records(df, article_col, summary_col)
    if not eval_records:
        raise ValueError("No valid rows available after filtering to build evaluation records.")

    tmp_json = args.tmp_json
    if not tmp_json:
        tmp_json = os.path.join(args.run_dir, "eval_from_excel.json")

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2, ensure_ascii=False)

    print(f"Prepared {len(eval_records)} eval records with columns: {article_col}, {summary_col}")
    print(f"Temporary eval JSON: {tmp_json}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_script = os.path.join(repo_root, "scripts", "run_evaluation.py")
    cmd = [
        sys.executable,
        eval_script,
        "--run_dir",
        args.run_dir,
        "--data",
        tmp_json,
        "--sample",
        "0",
        "--max_input_length",
        str(args.max_input_length),
        "--max_target_length",
        str(args.max_target_length),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not args.keep_tmp_json:
        try:
            os.remove(tmp_json)
            print(f"Removed temporary file: {tmp_json}")
        except OSError:
            pass

    print("Excel evaluation complete.")


if __name__ == "__main__":
    main()

