"""Create and verify the fixed-data SOTA retraining snapshot.

This script copies manifests only. It never copies images, checkpoints, or
the locked test set, and it refuses to proceed if split leakage is detected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["path", "label", "generator", "difficulty", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def normalize_holdout_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "path": row["path"],
            # The source column is is_real (1=real); project labels use 1=fake.
            "label": str(1 - int(row["is_real"])),
            "generator": row["generator"],
            "difficulty": "0",
            "split": "train",
        }
        for row in rows
    ]


def normalize_reserved_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "path": row["path"],
            "label": str(1 - int(row["is_real"])),
            "generator": row["generator"],
            "difficulty": "0",
            "split": "seen_generator_holdout",
        }
        for row in rows
    ]


def normalize_project_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        label = row.get("label")
        if label in (None, ""):
            # Validated feature indexes use is_real (1=real).
            label = str(1 - int(row["is_real"]))
        normalized.append(
            {
                "path": row["path"],
                "label": str(label),
                "generator": row.get("generator", ""),
                "difficulty": row.get("difficulty", "0"),
                "split": split,
            }
        )
    return normalized


def deduplicate_cross_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Keep the first deterministic manifest row and reject label conflicts."""
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["path"].lower()
        previous = seen.get(key)
        if previous is not None and (
            previous.get("label") != row.get("label")
            or previous.get("generator") != row.get("generator")
        ):
            raise SystemExit(f"conflicting duplicate in cross manifest: {row['path']}")
        seen[key] = row
    return list(seen.values()), len(rows) - len(seen)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train = normalize_holdout_rows(read_rows(args.train))
    val = normalize_project_rows(read_rows(args.val), "val")
    cross_raw = read_rows(args.cross)
    cross, cross_duplicates_removed = deduplicate_cross_rows(cross_raw)
    holdout = normalize_reserved_rows(read_rows(args.holdout))

    train_paths = {row["path"].lower() for row in train}
    val_paths = {row["path"].lower() for row in val}
    holdout_paths = {row["path"].lower() for row in holdout}
    forbidden_test_paths = train_paths | val_paths | holdout_paths
    cross_before_split_filter = len(cross)
    cross = [row for row in cross if row["path"].lower() not in forbidden_test_paths]
    cross_split_overlap_removed = cross_before_split_filter - len(cross)
    cross_paths = {row["path"].lower() for row in cross}
    if len(train_paths) != len(train) or len(val_paths) != len(val) or len(cross_paths) != len(cross):
        raise SystemExit("duplicate path detected inside a manifest")
    for name, left, right in (
        ("train/val", train_paths, val_paths),
        ("train/cross", train_paths, cross_paths),
        ("val/cross", val_paths, cross_paths),
        ("train/holdout", train_paths, holdout_paths),
    ):
        overlap = left & right
        if overlap:
            raise SystemExit(f"{name} overlap: {len(overlap)}")

    output_files = {
        "train_excluding_seen_holdout.csv": train,
        "val.csv": val,
        "cross_generator_test_valid.csv": cross,
        "seen_generator_holdout.csv": holdout,
    }
    for name, rows in output_files.items():
        write_rows(args.output / name, rows)

    metadata = {
        "experiment": "EXP3F_same_data_retrain",
        "created_by": "prepare_same_data_retrain.py",
        "train_n": len(train),
        "val_n": len(val),
        "cross_generator_test_n": len(cross),
        "cross_generator_test_source_n": len(cross_raw),
        "cross_generator_test_duplicates_removed": cross_duplicates_removed,
        "cross_generator_test_split_overlap_removed": cross_split_overlap_removed,
        "seen_generator_holdout_n": len(holdout),
        "labels": {
            "train": Counter(row["label"] for row in train),
            "val": Counter(row["label"] for row in val),
            "cross_generator_test": Counter(row["label"] for row in cross),
        },
        "generator_counts": {
            "train": Counter(row["generator"] for row in train),
            "val": Counter(row["generator"] for row in val),
            "cross_generator_test": Counter(row["generator"] for row in cross),
        },
        "locked_test_used": False,
        "train_holdout_overlap": 0,
        "source_sha256": {
            "train": sha256(args.train),
            "val": sha256(args.val),
            "cross": sha256(args.cross),
            "holdout": sha256(args.holdout),
        },
    }
    (args.output / "manifest_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
