#!/usr/bin/env python3
"""Inventory BOS batch roots and compare them with local downloads.

This command only lists and reports. It never calls ``bos sync``.
Comparison uses batch roots of the form ``BOS::raw_data/{robot}/{batch}``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from openpyxl import Workbook

from download_raw_data import (
    DEFAULT_BCECMD,
    DEFAULT_BOS_BASE,
    SUCCESSFUL_DOWNLOAD_STATUSES,
    load_mapping,
    parse_nine_device_sheet,
    task_from_bos_path,
    utc_now,
    write_csv,
    write_jsonl_atomic,
)


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL = TOOLS_DIR.parent.parent / "9台天轶设备.xlsx"
DEFAULT_MAPPING = TOOLS_DIR / "raw_data_mapping.json"
DEFAULT_LOCAL_ROOTS = (
    Path("/media/jushen/project-rl-dataset/raw_data_0804download"),
    Path("/media/jushen/project-rl-dataset/raw_data_0831download_by_schema"),
    Path("/media/jushen/project-rl-dataset/cosmos_raw_data"),
)
DEFAULT_RESULTS = (
    Path(
        "/media/jushen/project-rl-dataset/raw_data_0831download_by_schema"
        "/_metadata/download_state/download_results.jsonl"
    ),
    Path(
        "/media/jushen/project-rl-dataset/raw_data_0831download_by_schema"
        "/_metadata/download_state/incremental_20260903/download_results.jsonl"
    ),
)
DEFAULT_STATE_PARENT = Path(
    "/media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state"
)

FAMILY_DIRS = {
    "A_head_pose_force",
    "B_head_pose_no_force",
    "C_top_joint_only",
    "D_head_joint_only",
    "E_missing_images",
}
QUALITY_DIRS = {"_quarantine", "_review"}
PRE_RE = re.compile(r"\bPRE\s+(\S+)")
ROBOT_NAME_RE = re.compile(r"^(tienyi_prod2_dualArm-gripper-3cameras_\d+)")
BOS_PREFIX = "BOS::"


@dataclass
class InventoryConfig:
    robot_dir_prefix: str
    exclude_local_dir_suffixes: tuple[str, ...]
    organized_skip_dirs: frozenset[str]


@dataclass
class LocalHit:
    bos_path: str | None
    batch: str
    local_path: str
    source_root: str
    hdf5_count: int
    has_complete_marker: bool
    resolved: bool
    origin: str


@dataclass
class BosBatch:
    bos_path: str
    robot_type: str
    batch: str
    task_id: str
    matched_task: bool


def inventory_config(mapping: dict[str, Any]) -> InventoryConfig:
    payload = mapping.get("inventory") if isinstance(mapping.get("inventory"), dict) else {}
    suffixes = payload.get("exclude_local_dir_suffixes") or ["_left_gripper_cut"]
    skip = payload.get("organized_skip_dirs") or ["_metadata"]
    return InventoryConfig(
        robot_dir_prefix=str(payload.get("robot_dir_prefix") or "tienyi_prod2_dualArm-gripper-3cameras_"),
        exclude_local_dir_suffixes=tuple(str(item) for item in suffixes),
        organized_skip_dirs=frozenset(str(item) for item in skip),
    )


def strip_bos_prefix(path: str) -> str:
    text = path.strip().replace("\\", "/")
    if text.startswith("http://BOS::"):
        text = text[len("http://") :]
    if text.startswith(BOS_PREFIX):
        text = text[len(BOS_PREFIX) :]
    if text.startswith("bos:/"):
        remainder = text[len("bos:/") :]
        _bucket, _, rest = remainder.partition("/")
        text = rest
    return text.strip("/")


def normalize_batch_root(path: str) -> str | None:
    """Keep only ``BOS::raw_data/{robot}/{batch}``."""
    parts = [part for part in strip_bos_prefix(path).split("/") if part]
    if len(parts) < 3 or parts[0] != "raw_data":
        return None
    return f"{BOS_PREFIX}raw_data/{parts[1]}/{parts[2]}"


def bos_root_from_batch_name(batch: str, prefix: str) -> str | None:
    match = ROBOT_NAME_RE.match(batch)
    if match is None or not batch.startswith(prefix):
        return None
    return f"{BOS_PREFIX}raw_data/{match.group(1)}/{batch}"


def is_excluded_dir(name: str, suffixes: Iterable[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def count_hdf5(directory: Path) -> int:
    return sum(1 for path in directory.rglob("trajectory.hdf5") if path.is_file())


def has_complete_marker(directory: Path) -> bool:
    return (directory / ".download_complete").is_file() or (directory / ".download_complete.json").is_file()


def detect_layout(root: Path) -> str:
    names = {path.name for path in root.iterdir() if path.is_dir()}
    if names & (FAMILY_DIRS | QUALITY_DIRS | {"_metadata"}):
        return "organized"
    return "flat"


def parse_pre_listing(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        match = PRE_RE.search(line)
        if match:
            names.append(match.group(1).rstrip("/"))
    return names


def list_bos_prefixes(bcecmd: Path, uri: str) -> list[str]:
    command = [str(bcecmd), "bos", "ls", "--all", uri]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip() or f"exit {completed.returncode}"
        raise RuntimeError(f"bcecmd ls failed for {uri}: {detail}")
    return parse_pre_listing(completed.stdout)


def iter_organized_batch_dirs(root: Path, skip_dirs: frozenset[str]) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in skip_dirs:
            continue
        if child.name in QUALITY_DIRS:
            for family in sorted(child.iterdir()):
                if family.is_dir() and family.name in FAMILY_DIRS:
                    yield from _batches_under_family(family)
        elif child.name in FAMILY_DIRS:
            yield from _batches_under_family(child)


def _batches_under_family(family: Path) -> Iterable[Path]:
    for schema in sorted(family.iterdir()):
        if not schema.is_dir():
            continue
        for station in sorted(schema.iterdir()):
            if not station.is_dir():
                continue
            for task in sorted(station.iterdir()):
                if not task.is_dir():
                    continue
                for batch in sorted(task.iterdir()):
                    if batch.is_dir():
                        yield batch


def scan_flat_root(root: Path, config: InventoryConfig) -> list[LocalHit]:
    hits: list[LocalHit] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or is_excluded_dir(child.name, config.exclude_local_dir_suffixes):
            continue
        bos_path = bos_root_from_batch_name(child.name, config.robot_dir_prefix)
        hits.append(
            LocalHit(
                bos_path=bos_path,
                batch=child.name,
                local_path=str(child),
                source_root=str(root),
                hdf5_count=count_hdf5(child),
                has_complete_marker=has_complete_marker(child),
                resolved=bos_path is not None,
                origin="flat",
            )
        )
    return hits


def scan_organized_root(root: Path, config: InventoryConfig, aliases: dict[str, str]) -> list[LocalHit]:
    hits: list[LocalHit] = []
    for batch_dir in iter_organized_batch_dirs(root, config.organized_skip_dirs):
        bos_path = bos_root_from_batch_name(batch_dir.name, config.robot_dir_prefix) or aliases.get(batch_dir.name)
        hits.append(
            LocalHit(
                bos_path=bos_path,
                batch=batch_dir.name,
                local_path=str(batch_dir),
                source_root=str(root),
                hdf5_count=count_hdf5(batch_dir),
                has_complete_marker=has_complete_marker(batch_dir),
                resolved=bos_path is not None,
                origin="organized",
            )
        )
    return hits


def scan_local_root(root: Path, config: InventoryConfig, aliases: dict[str, str]) -> list[LocalHit]:
    if not root.is_dir():
        raise FileNotFoundError(f"local root not found: {root}")
    if detect_layout(root) == "organized":
        return scan_organized_root(root, config, aliases)
    return scan_flat_root(root, config)


def load_result_aliases(results_paths: Iterable[Path]) -> tuple[list[LocalHit], dict[str, str]]:
    hits: list[LocalHit] = []
    aliases: dict[str, str] = {}
    latest: dict[str, dict[str, Any]] = {}
    for results_path in results_paths:
        if not results_path.is_file():
            continue
        with results_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {results_path}:{line_number}: {error}") from error
                record_id = str(record.get("record_id") or f"{results_path}:{line_number}")
                latest[record_id] = record
                destination = Path(str(record.get("destination") or ""))
                normalized = normalize_batch_root(str(record.get("bos_path") or ""))
                if destination.name and normalized:
                    aliases.setdefault(destination.name, normalized)
    for record in latest.values():
        if record.get("status") not in SUCCESSFUL_DOWNLOAD_STATUSES:
            continue
        if int(record.get("hdf5_count") or 0) < 1:
            continue
        bos_path = normalize_batch_root(str(record.get("bos_path") or ""))
        destination = str(record.get("destination") or "")
        hits.append(
            LocalHit(
                bos_path=bos_path,
                batch=Path(destination).name if destination else (bos_path or "").rsplit("/", 1)[-1],
                local_path=destination,
                source_root=str(Path(destination).parent) if destination else "",
                hdf5_count=int(record.get("hdf5_count") or 0),
                has_complete_marker=True,
                resolved=bos_path is not None,
                origin="results",
            )
        )
    return hits, aliases


def successful_local_hits(hits: Iterable[LocalHit]) -> list[LocalHit]:
    return [
        hit
        for hit in hits
        if hit.resolved and hit.bos_path and (hit.hdf5_count >= 1 or hit.has_complete_marker)
    ]


def merge_successful(hits: Iterable[LocalHit]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for hit in successful_local_hits(hits):
        assert hit.bos_path is not None
        entry = merged.setdefault(
            hit.bos_path,
            {
                "bos_path": hit.bos_path,
                "batch": hit.batch,
                "hdf5_count": 0,
                "sources": [],
                "local_paths": [],
            },
        )
        entry["hdf5_count"] += hit.hdf5_count
        entry["sources"].append(hit.origin)
        entry["local_paths"].append(hit.local_path)
    for entry in merged.values():
        entry["sources"] = sorted(set(entry["sources"]))
        entry["local_paths"] = sorted(set(entry["local_paths"]))
    return merged


def load_excel_bos_roots(excel: Path, mapping_path: Path) -> set[str]:
    items, _ = parse_nine_device_sheet(excel, mapping_path, Path("/inventory-unused"), set(), set())
    roots: set[str] = set()
    for item in items:
        normalized = normalize_batch_root(item.bos_path)
        if normalized:
            roots.add(normalized)
    return roots


def classify_bos_batch(path: str, mapping: dict[str, Any]) -> BosBatch:
    normalized = normalize_batch_root(path)
    if normalized is None:
        raise ValueError(f"not a batch root: {path}")
    robot_type, batch = normalized[len(f"{BOS_PREFIX}raw_data/") :].split("/", 1)
    task_id, _, _, _ = task_from_bos_path(normalized, mapping)
    return BosBatch(
        bos_path=normalized,
        robot_type=robot_type,
        batch=batch,
        task_id=task_id,
        matched_task=task_id != "unknown_task",
    )


def list_bos_batches(
    *,
    bcecmd: Path,
    bos_base: str,
    config: InventoryConfig,
    mapping: dict[str, Any],
    list_children: Callable[[Path, str], list[str]],
) -> tuple[list[BosBatch], list[dict[str, Any]]]:
    raw_uri = f"{bos_base.rstrip('/')}/raw_data/"
    robots = [
        name
        for name in list_children(bcecmd, raw_uri)
        if name.startswith(config.robot_dir_prefix)
    ]
    batches: list[BosBatch] = []
    unmatched: list[dict[str, Any]] = []
    for robot in robots:
        robot_uri = f"{bos_base.rstrip('/')}/raw_data/{robot}/"
        for batch in list_children(bcecmd, robot_uri):
            bos_path = f"{BOS_PREFIX}raw_data/{robot}/{batch}"
            record = classify_bos_batch(bos_path, mapping)
            if record.matched_task:
                batches.append(record)
            else:
                unmatched.append(asdict(record))
    return batches, unmatched


def write_missing_report(
    path: Path,
    *,
    generated_at: str,
    excel: Path,
    local_roots: list[Path],
    buckets: dict[str, list[str]],
    counts: dict[str, int],
) -> None:
    lines = [
        "# BOS inventory vs local downloads",
        "",
        f"- Generated: `{generated_at}`",
        f"- Excel: `{excel}`",
        "- Local roots:",
    ]
    lines.extend(f"  - `{root}`" for root in local_roots)
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- BOS matched batches: {counts['bos_matched']}",
            f"- BOS unmatched task pattern: {counts['bos_unmatched']}",
            f"- Local successful batch roots: {counts['local_successful']}",
            f"- Excel batch roots: {counts['excel']}",
            f"- In Excel, not local: {len(buckets['in_excel_not_local'])}",
            f"- On BOS only, not local: {len(buckets['bos_only_not_local'])}",
            f"- In Excel, not on BOS: {len(buckets['excel_not_on_bos'])}",
            f"- Local, not on BOS: {len(buckets['local_not_on_bos'])}",
            f"- Unresolved local directories: {counts['local_unresolved']}",
            "",
        ]
    )
    titles = (
        ("in_excel_not_local", "In Excel and on BOS, not downloaded"),
        ("bos_only_not_local", "On BOS, not in Excel, not downloaded"),
        ("excel_not_on_bos", "In Excel, missing on BOS"),
        ("local_not_on_bos", "Downloaded locally, missing on BOS"),
    )
    for key, title in titles:
        lines.extend([f"## {title}", ""])
        records = buckets[key]
        if not records:
            lines.append("None.")
        else:
            lines.extend(f"- `{item}`" for item in records)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def default_state_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return DEFAULT_STATE_PARENT / f"bos_inventory_{stamp}"


def run_inventory(args: argparse.Namespace) -> int:
    mapping = load_mapping(args.mapping)
    config = inventory_config(mapping)
    if getattr(args, "state_dir", None) is None:
        args.state_dir = default_state_dir()
    local_roots = list(args.local_root or DEFAULT_LOCAL_ROOTS)
    results_paths = list(args.results or DEFAULT_RESULTS)
    list_children = args.list_children or list_bos_prefixes

    result_hits, aliases = load_result_aliases(results_paths)
    local_hits = list(result_hits)
    for root in local_roots:
        local_hits.extend(scan_local_root(root, config, aliases))

    successful = merge_successful(local_hits)
    unresolved = [hit for hit in local_hits if hit.origin != "results" and not hit.resolved]
    excel_roots = load_excel_bos_roots(args.excel, args.mapping) if args.excel else set()

    bos_batches: list[BosBatch] = []
    unmatched: list[dict[str, Any]] = []
    if not args.skip_bos_list:
        bos_batches, unmatched = list_bos_batches(
            bcecmd=args.bcecmd,
            bos_base=args.bos_base,
            config=config,
            mapping=mapping,
            list_children=list_children,
        )
    bos_roots = {item.bos_path for item in bos_batches}
    local_roots_set = set(successful)
    if args.skip_bos_list:
        buckets = {
            "in_excel_not_local": sorted(excel_roots - local_roots_set),
            "bos_only_not_local": [],
            "excel_not_on_bos": [],
            "local_not_on_bos": [],
        }
    else:
        buckets = {
            "in_excel_not_local": sorted((excel_roots & bos_roots) - local_roots_set),
            "bos_only_not_local": sorted(bos_roots - excel_roots - local_roots_set),
            "excel_not_on_bos": sorted(excel_roots - bos_roots),
            "local_not_on_bos": sorted(local_roots_set - bos_roots),
        }

    generated_at = utc_now()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.state_dir / "bos_inventory.jsonl", (asdict(item) for item in bos_batches))
    write_jsonl_atomic(args.state_dir / "local_successful.jsonl", successful.values())
    write_jsonl_atomic(args.state_dir / "unmatched_bos.jsonl", unmatched)
    write_jsonl_atomic(args.state_dir / "unresolved_local.jsonl", (asdict(hit) for hit in unresolved))
    write_csv(
        args.state_dir / "missing_buckets.csv",
        [{"bucket": key, "bos_path": path} for key, paths in buckets.items() for path in paths],
    )
    counts = {
        "bos_matched": len(bos_batches),
        "bos_unmatched": len(unmatched),
        "local_successful": len(successful),
        "excel": len(excel_roots),
        "local_unresolved": len(unresolved),
    }
    write_missing_report(
        args.state_dir / "missing_report.md",
        generated_at=generated_at,
        excel=args.excel,
        local_roots=local_roots,
        buckets=buckets,
        counts=counts,
    )
    print(f"BOS matched batches: {counts['bos_matched']}")
    print(f"BOS unmatched task pattern: {counts['bos_unmatched']}")
    print(f"local successful: {counts['local_successful']}")
    print(f"excel roots: {counts['excel']}")
    print(f"in excel, not local: {len(buckets['in_excel_not_local'])}")
    print(f"on BOS only, not local: {len(buckets['bos_only_not_local'])}")
    print(f"in excel, not on BOS: {len(buckets['excel_not_on_bos'])}")
    print(f"unresolved local: {counts['local_unresolved']}")
    print(f"report: {args.state_dir / 'missing_report.md'}")
    return 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        mapping_payload = {
            "stations": {"天轶9": "tienyi_9"},
            "tasks": {
                "15 背部提手安装": {
                    "task_id": "back_handle",
                    "task_description_en": "Install handle.",
                    "reviewed": True,
                }
            },
            "source_overrides": {},
            "path_task_patterns": {"back-handle-installation": "back_handle"},
            "inventory": {
                "robot_dir_prefix": "tienyi_prod2_dualArm-gripper-3cameras_",
                "exclude_local_dir_suffixes": ["_left_gripper_cut"],
                "organized_skip_dirs": ["_metadata"],
            },
        }
        mapping_path = root / "mapping.json"
        mapping_path.write_text(json.dumps(mapping_payload), encoding="utf-8")

        robot = "tienyi_prod2_dualArm-gripper-3cameras_394"
        batch = f"{robot}_back-handle-installation_20260803_pm"
        fail = f"{robot}_back-handle-installation_20260803_pm-fail"
        cut = f"{robot}_back-handle-installation_20260803_pm_left_gripper_cut"
        alias_batch = "back_handle_20260804_am_success"
        alias_bos = f"BOS::raw_data/{robot}/{robot}_back-handle-installation_20260804_am"

        flat = root / "flat"
        for name in (batch, fail, cut):
            hdf5 = flat / name / "0803_120000" / "data"
            hdf5.mkdir(parents=True)
            (hdf5 / "trajectory.hdf5").write_bytes(b"x")
        (flat / cut / "truncation_report.json").write_text("{}", encoding="utf-8")

        organized = root / "organized"
        organized_batch = (
            organized
            / "A_head_pose_force"
            / "schema_test"
            / "unknown_station"
            / "back_handle"
            / alias_batch
            / "success_episodes"
            / "0804_100000"
            / "data"
        )
        organized_batch.mkdir(parents=True)
        (organized_batch / "trajectory.hdf5").write_bytes(b"y")
        (organized / "_metadata").mkdir()

        results = root / "results.jsonl"
        results.write_text(
            json.dumps(
                {
                    "record_id": "row_1",
                    "status": "downloaded",
                    "hdf5_count": 1,
                    "bos_path": f"{alias_bos}/success_episodes/0804_100000/data",
                    "destination": str(organized_batch.parents[2]),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        excel_book = Workbook()
        sheet = excel_book.active
        sheet.title = "Sheet1"
        sheet.append(["任务英文名称", "原始路径"])
        sheet.append([batch, f"raw_data/{robot}/{batch}"])
        sheet.append(["missing", f"raw_data/{robot}/{robot}_back-handle-installation_20260899_am"])
        excel = root / "nine.xlsx"
        excel_book.save(excel)

        listing = {
            f"{DEFAULT_BOS_BASE}/raw_data/": [robot, "agilex_other"],
            f"{DEFAULT_BOS_BASE}/raw_data/{robot}/": [
                batch,
                f"{robot}_back-handle-installation_20260804_am",
                f"{robot}_Plug_in_the_charging_cable_20260501",
            ],
        }

        def fake_list(_bcecmd: Path, uri: str) -> list[str]:
            return listing[uri]

        state_dir = root / "state"
        args = argparse.Namespace(
            mapping=mapping_path,
            excel=excel,
            bcecmd=Path("/bin/true"),
            bos_base=DEFAULT_BOS_BASE,
            local_root=[flat, organized],
            results=[results],
            state_dir=state_dir,
            skip_bos_list=False,
            list_children=fake_list,
        )
        assert run_inventory(args) == 0
        report = (state_dir / "missing_report.md").read_text(encoding="utf-8")
        successful = [
            json.loads(line) for line in (state_dir / "local_successful.jsonl").read_text(encoding="utf-8").splitlines() if line
        ]
        successful_paths = {row["bos_path"] for row in successful}
        assert f"BOS::raw_data/{robot}/{batch}" in successful_paths
        assert f"BOS::raw_data/{robot}/{fail}" in successful_paths
        assert alias_bos in successful_paths
        assert not any(cut in row["bos_path"] for row in successful)
        assert f"BOS::raw_data/{robot}/{robot}_back-handle-installation_20260899_am" in report
        unmatched = [
            json.loads(line) for line in (state_dir / "unmatched_bos.jsonl").read_text(encoding="utf-8").splitlines() if line
        ]
        assert any("Plug_in_the_charging_cable" in row["batch"] for row in unmatched)
        assert normalize_batch_root(f"{alias_bos}/success_episodes/0804_100000/data") == alias_bos
        assert parse_pre_listing("                                               PRE  foo/\n") == ["foo"]
    print("inventory_bos self-test: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--bcecmd", type=Path, default=DEFAULT_BCECMD)
    parser.add_argument("--bos-base", default=DEFAULT_BOS_BASE)
    parser.add_argument("--local-root", type=Path, action="append")
    parser.add_argument("--results", type=Path, action="append")
    parser.add_argument("--state-dir", type=Path, default=default_state_dir())
    parser.add_argument("--skip-bos-list", action="store_true")
    parser.set_defaults(list_children=None, handler=run_inventory)
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "self-test":
        raise SystemExit(run_self_test())
    args = build_parser().parse_args()
    raise SystemExit(run_inventory(args))


if __name__ == "__main__":
    main()
