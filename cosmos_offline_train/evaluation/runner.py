"""Independent offline evaluation command."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from ..config import load_config
from ..train import _run_validation, _run_validation_suite
from .report import compare_episode_summaries, compare_summaries


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("summary must use NAME=PATH")
    name, path = value.split("=", 1)
    resolved = Path(path).expanduser().resolve()
    if not name or not resolved.is_file():
        raise argparse.ArgumentTypeError(f"invalid summary: {value}")
    return name, resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--base", action="store_true")
    evaluate.add_argument(
        "--objectives", nargs="+",
        choices=("policy", "world", "value", "inverse_dynamics"),
    )
    evaluate.add_argument(
        "--inverse-ablations", nargs="+",
        choices=("normal", "current_only", "future_only", "future_shuffle"),
    )
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--output-dir", type=Path)

    suite = subparsers.add_parser("evaluate-suite")
    suite.add_argument("--config", required=True)
    suite.add_argument("--base", action="store_true")
    suite.add_argument(
        "--checkpoint", action="append", default=[], type=_parse_named_path,
        help="Named checkpoint in NAME=PATH form; NAME becomes its output directory",
    )
    suite.add_argument(
        "--objectives", nargs="+",
        choices=("policy", "world", "value", "inverse_dynamics"),
    )
    suite.add_argument(
        "--inverse-ablations", nargs="+",
        choices=("normal", "current_only", "future_only", "future_shuffle"),
    )
    suite.add_argument("--max-batches", type=int)
    suite.add_argument("--output-dir", required=True, type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--summary", action="append", required=True, type=_parse_named_path)
    compare.add_argument("--output-dir", required=True, type=Path)

    compare_episodes = subparsers.add_parser("compare-episodes")
    compare_episodes.add_argument(
        "--summary", action="append", required=True, type=_parse_named_path
    )
    compare_episodes.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args()
    if args.command in {"compare", "compare-episodes"}:
        paths = dict(args.summary)
        if len(paths) < 2:
            raise SystemExit(f"{args.command} requires at least two distinct summaries")
        comparison = (
            compare_summaries
            if args.command == "compare"
            else compare_episode_summaries
        )
        for path in comparison(paths, args.output_dir.expanduser().resolve()):
            print(path)
        return

    config = load_config(args.config)
    if args.objectives:
        config["evaluation"]["objectives"] = args.objectives
    if args.inverse_ablations:
        config["evaluation"]["inverse_condition_ablations"] = args.inverse_ablations
    if args.max_batches is not None:
        if args.max_batches < 1:
            raise SystemExit("--max-batches must be >= 1")
        config["evaluation"]["max_validation_batches"] = args.max_batches
    if args.output_dir is not None:
        config["runtime"]["output_dir"] = str(args.output_dir.expanduser().resolve())
    if args.command == "evaluate-suite":
        if not args.base and not args.checkpoint:
            raise SystemExit("evaluate-suite requires --base or --checkpoint NAME=PATH")
        names = [name for name, _ in args.checkpoint]
        if len(names) != len(set(names)) or "base" in names:
            raise SystemExit("checkpoint names must be unique and cannot use 'base'")
        _run_validation_suite(config, args.checkpoint, include_base=args.base)
        return
    mode = "validate-base" if args.base else "validate"
    _run_validation(config, SimpleNamespace(mode=mode, resume=args.checkpoint))


if __name__ == "__main__":
    main()
