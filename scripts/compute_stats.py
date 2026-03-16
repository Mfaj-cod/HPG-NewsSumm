import argparse
import json
import numpy as np

def compute(path_to_dataset):
    with open(path_to_dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_docs = []
    doc_lengths = []
    summary_lengths = []
    length = len(data)

    for item in data:
        docs = item["documents"]
        summary = item["summary"]

        num_docs.append(len(docs))
        for d in docs:
            doc_lengths.append(len(d.split()))

        summary_lengths.append(len(summary.split()))

    return num_docs, doc_lengths, summary_lengths, length

def main(args):
    num_docs, doc_lengths, summary_lengths, length = compute(args.data)

    print("NewsSumm Dataset Statistics:\n")
    print(f"Number of clusters: {length}")
    print(f"Avg documents per cluster: {np.mean(num_docs):.2f}")
    print(f"Avg document length (words): {np.mean(doc_lengths):.2f}")
    print(f"Avg summary length (words): {np.mean(summary_lengths):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the processed JSON file")

    args = parser.parse_args()
    main(args)
