#!/usr/bin/env python3
"""Map a frozen raw-source split onto one converted Cosmos dataset tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..dataset import save_manifest
from .paired_index import index_converted_records, load_source_ids, parse_collection_name_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val EpisodeRecord manifests from a shared raw split"
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-train", required=True, type=Path)
    parser.add_argument("--source-val", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--collection-name-map",
        action="append",
        default=[],
        help="Optional DIR=COLLECTION aliases; default infers 0803_pm_success from folder names",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=["success"],
        help="Episode outcomes to index. Default is success-only.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    by_id = index_converted_records(
        dataset_root,
        collection_name_map=parse_collection_name_map(args.collection_name_map),
        outcomes=set(args.outcomes),
    )
    train_ids = load_source_ids(args.source_train.expanduser().resolve())
    val_ids = load_source_ids(args.source_val.expanduser().resolve())
    requested = train_ids + val_ids
    missing = sorted(set(requested) - set(by_id))
    unexpected = sorted(set(by_id) - set(requested))
    if missing or unexpected:
        raise ValueError(
            f"converted/source split mismatch: missing={missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )

    output = args.output_dir.expanduser().resolve()
    save_manifest(
        [by_id[source_id] for source_id in train_ids],
        output / "train.json",
        overwrite=args.overwrite,
    )
    save_manifest(
        [by_id[source_id] for source_id in val_ids],
        output / "val.json",
        overwrite=args.overwrite,
    )
    print(
        f"[PAIRED-MANIFEST] dataset={dataset_root} "
        f"train={len(train_ids)} val={len(val_ids)} output={output}"
    )


if __name__ == "__main__":
    main()
