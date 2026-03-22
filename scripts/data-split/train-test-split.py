import argparse
import json
import os
import random
from typing import Any, List, Sequence, Tuple


def load_json(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}, got {type(data)}")
    return data


def parse_test_size(value: str, total: int) -> int:
    """
    Accepts either:
    - float ratio in (0,1), e.g. "0.2"
    - int count >= 1, e.g. "1000"
    """
    value = value.strip()
    if "." in value:
        ratio = float(value)
        if ratio <= 0 or ratio >= 1:
            raise ValueError("When using ratio, --test_size must be between 0 and 1 (exclusive).")
        size = int(total * ratio)
        return max(1, min(size, total - 1))

    count = int(value)
    if count <= 0:
        raise ValueError("When using absolute count, --test_size must be >= 1.")
    if count >= total:
        raise ValueError("--test_size must be smaller than total number of samples.")
    return count


def split_data(data: Sequence[Any], test_size: int, seed: int, shuffle: bool) -> Tuple[List[Any], List[Any]]:
    indices = list(range(len(data)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    test_indices = set(indices[:test_size])
    train = [data[i] for i in range(len(data)) if i not in test_indices]
    test = [data[i] for i in range(len(data)) if i in test_indices]
    return train, test


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSON dataset into train/test sets.")
    parser.add_argument(
        "--input",
        default="data/newssumm_processed/newssumm_processed.json",
        help="Input JSON file (top-level list).",
    )
    parser.add_argument(
        "--test_size",
        default="0.2",
        help='Test split size as ratio (e.g. "0.2") or absolute count (e.g. "500").',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling before split.")
    parser.add_argument(
        "--output_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for train/test files.",
    )
    parser.add_argument("--train_name", default="train.json")
    parser.add_argument("--test_name", default="test.json")
    args = parser.parse_args()

    data = load_json(args.input)
    total = len(data)
    if total < 2:
        raise ValueError("Need at least 2 samples to split train/test.")

    test_size = parse_test_size(args.test_size, total)
    train, test = split_data(data, test_size=test_size, seed=args.seed, shuffle=not args.no_shuffle)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, args.train_name)
    test_path = os.path.join(args.output_dir, args.test_name)

    save_json(train_path, train)
    save_json(test_path, test)

    print(f"Input: {args.input}")
    print(f"Total samples: {total}")
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"Saved train to: {train_path}")
    print(f"Saved test to: {test_path}")


if __name__ == "__main__":
    main()

