#!/usr/bin/env python3
"""CLI compatibility wrapper for the independent synthetic HDF5 generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .hdf5_generator import create_synthetic_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    hdf5_path, stats_path = create_synthetic_fixture(args.output.expanduser().resolve())
    print(f"HDF5: {hdf5_path}")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
