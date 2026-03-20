import argparse
import json
import sys


REQUIRED_KEYS = ("documents", "summary")
OPTIONAL_KEYS = ("cluster_id", "metadata", "documents_meta", "id")


def _add_error(errors, idx, msg):
    errors.append(f"ERROR [index {idx}]: {msg}")


def _add_warning(warnings, idx, msg):
    warnings.append(f"WARNING [index {idx}]: {msg}")


def validate_record(record, idx, strict, errors, warnings):
    if not isinstance(record, dict):
        _add_error(errors, idx, f"Record is {type(record).__name__}, expected dict")
        return

    for key in REQUIRED_KEYS:
        if key not in record:
            _add_error(errors, idx, f"Missing required key: '{key}'")

    docs = record.get("documents")
    if "documents" in record:
        if not isinstance(docs, list):
            _add_error(errors, idx, f"'documents' is {type(docs).__name__}, expected list")
        else:
            non_str = [type(d).__name__ for d in docs if not isinstance(d, str)]
            if non_str:
                _add_error(
                    errors,
                    idx,
                    f"'documents' contains non-string items (e.g. {non_str[0]})",
                )
            if strict and len(docs) == 0:
                _add_error(errors, idx, "'documents' is empty")

    summary = record.get("summary")
    if "summary" in record:
        if not isinstance(summary, str):
            _add_error(errors, idx, f"'summary' is {type(summary).__name__}, expected str")
        elif strict and not summary.strip():
            _add_error(errors, idx, "'summary' is empty or whitespace")

    if "metadata" in record and not isinstance(record["metadata"], dict):
        msg = f"'metadata' is {type(record['metadata']).__name__}, expected dict"
        if strict:
            _add_error(errors, idx, msg)
        else:
            _add_warning(warnings, idx, msg)

    if "documents_meta" in record:
        docs_meta = record["documents_meta"]
        if not isinstance(docs_meta, list):
            msg = f"'documents_meta' is {type(docs_meta).__name__}, expected list"
            if strict:
                _add_error(errors, idx, msg)
            else:
                _add_warning(warnings, idx, msg)
        elif isinstance(docs, list) and len(docs_meta) != len(docs):
            msg = "'documents_meta' length does not match 'documents' length"
            if strict:
                _add_error(errors, idx, msg)
            else:
                _add_warning(warnings, idx, msg)


def main():
    parser = argparse.ArgumentParser(description="Validate NewsSumm JSON dataset schema")
    parser.add_argument("--data", required=True, help="Path to dataset JSON file")
    parser.add_argument("--sample", type=int, default=None, help="Validate only the first N records")
    parser.add_argument("--max-errors", type=int, default=20, help="Stop after this many errors")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors and enforce non-empty fields")

    args = parser.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to read JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"ERROR: Top-level JSON is {type(data).__name__}, expected list")
        sys.exit(1)

    if len(data) == 0:
        print("WARNING: Dataset is empty")

    total = len(data)
    limit = total if args.sample is None else min(total, args.sample)

    errors = []
    warnings = []

    for idx in range(limit):
        validate_record(data[idx], idx, args.strict, errors, warnings)
        if len(errors) >= args.max_errors:
            break

    for msg in errors:
        print(msg)
    for msg in warnings:
        print(msg)

    print("\nSummary:")
    print(f"Records checked: {limit}/{total}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Required keys: {', '.join(REQUIRED_KEYS)}")
    print(f"Optional keys: {', '.join(OPTIONAL_KEYS)}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
