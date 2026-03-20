import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    # Removing HTML
    text = BeautifulSoup(text, "lxml").get_text()
    # Normalizing whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def main():
    input_path = "data/NewsSumm_Dataset.xlsx"
    output_path = "data/NewsSumm_Cleaned.xlsx"

    print("Loading dataset...")
    df = pd.read_excel(input_path)
    original_len = len(df)

    print("Initial rows:", original_len)

    # Renaming columns to standard
    df.columns = [c.strip() for c in df.columns]

    # Dropping rows with missing critical fields
    df = df.dropna(subset=["article_text", "human_summary"])

    print("After dropping missing article/summary:", len(df))

    # Clean text fields
    tqdm.pandas()
    for col in ["headline", "article_text", "human_summary"]:
        df[col] = df[col].progress_apply(clean_text)

    # Dropping empty after cleaning
    df = df[
        (df["article_text"].str.len() > 50) &
        (df["human_summary"].str.len() > 10)
    ]
    print("After removing very short entries:", len(df))

    # Removing duplicates based on article_text
    df = df.drop_duplicates(subset=["article_text"])
    print("After removing duplicates:", len(df))

    # Resetting index
    df = df.reset_index(drop=True)

    # Saving cleaned dataset
    df.to_excel(output_path, index=False)

    print("\n\n")
    print("Cleaning complete.")
    print("Original rows:", original_len)
    print("Final rows:", len(df))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
