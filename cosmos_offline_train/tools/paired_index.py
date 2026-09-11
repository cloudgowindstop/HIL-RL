"""Map converted Cosmos episode trees onto frozen raw-source identities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..action_spec import get_action_spec
from ..dataset import EpisodeRecord, build_episode_index, discover_dataset_roots

COLLECTION_FROM_DIR = re.compile(
    r"(?P<date>\d{8})_(?P<period>am|pm)(?:_(?P<outcome>success|failure))?$",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def load_source_ids(path: Path) -> list[str]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"source split must be a list: {path}")
    return [str(record["source_id"]) for record in payload]


def parse_collection_name_map(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"collection map must use DIR=COLLECTION: {value}")
        directory, collection = value.split("=", 1)
        directory, collection = directory.strip(), collection.strip()
        if not directory or not collection:
            raise ValueError(f"invalid collection map: {value}")
        mapping[directory] = collection
    return mapping


def infer_collection_name(dataset_root: Path) -> str:
    match = COLLECTION_FROM_DIR.search(dataset_root.name)
    if match is None:
        return dataset_root.name
    date = match.group("date")
    period = match.group("period").lower()
    outcome = (match.group("outcome") or "success").lower()
    return f"{date[4:]}_{period}_{outcome}"


def source_id_from_path(path: str) -> str:
    source = Path(path)
    return source.parent.parent.name if source.parent.name == "data" else source.parent.name


def source_paths_for_dataset(dataset_root: Path) -> list[str]:
    conversion_manifest = dataset_root / "meta" / "conversion_manifest.json"
    if conversion_manifest.is_file():
        payload = load_json(conversion_manifest)
        sources = payload.get("sources", [])
        paths = [str(item.get("path", "")) for item in sources]
        if any(paths):
            return paths
    metadata_path = dataset_root / "cosmos_dataset_metadata.json"
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        episodes = (
            metadata.get("statistics", {})
            .get("range_report", {})
            .get("episodes", [])
        )
        paths = [str(item.get("episode", "")) for item in episodes]
        if any(paths):
            return paths
    return []


def index_converted_records(
    root: str | Path,
    *,
    collection_name_map: dict[str, str] | None = None,
    outcomes: set[str] | None = None,
) -> dict[str, EpisodeRecord]:
    """Index converted parquet episodes by frozen `{collection}/{episode_id}`."""
    root = Path(root).expanduser().resolve()
    mapping = dict(collection_name_map or {})
    allowed = {value.lower() for value in outcomes} if outcomes else None
    by_id: dict[str, EpisodeRecord] = {}
    for dataset_root in discover_dataset_roots(root):
        metadata = load_json(dataset_root / "cosmos_dataset_metadata.json")
        outcome = str(metadata.get("episode_labeling", {}).get("outcome", "unknown"))
        if allowed is not None and outcome.lower() not in allowed:
            continue
        collection = mapping.get(dataset_root.name) or infer_collection_name(dataset_root)
        source_paths = source_paths_for_dataset(dataset_root)
        records = [
            record
            for record in build_episode_index(dataset_root)
            if Path(record.dataset_root) == dataset_root.resolve()
        ]
        for record in records:
            match = re.search(r"episode_(\d+)\.parquet$", Path(record.path).name)
            episode_index = int(match.group(1)) if match else -1
            source_path = record.source_path
            if (not source_path) and 0 <= episode_index < len(source_paths):
                source_path = source_paths[episode_index]
            episode_id = source_id_from_path(source_path) if source_path else record.source_id
            if not episode_id:
                raise ValueError(f"converted record lacks source identity: {record.path}")
            paired_id = f"{collection}/{episode_id}"
            if paired_id in by_id:
                raise ValueError(f"duplicate converted source identity: {paired_id}")
            by_id[paired_id] = EpisodeRecord(
                path=record.path,
                dataset_root=record.dataset_root,
                group=record.group,
                task=record.task,
                outcome=record.outcome,
                rows=record.rows,
                source_path=source_path or record.source_path,
                source_id=episode_id,
            )
    return by_id


def physical_action_channels(encoding: str) -> dict[str, tuple[int, ...]]:
    spec = get_action_spec(encoding)
    left_translation = (0, 1, 2)
    right_translation = (
        spec.per_arm_dimension,
        spec.per_arm_dimension + 1,
        spec.per_arm_dimension + 2,
    )
    return {
        "left_translation": left_translation,
        "right_translation": right_translation,
        "left_gripper": (spec.gripper_index,),
        "right_gripper": (spec.per_arm_dimension + spec.gripper_index,),
    }
