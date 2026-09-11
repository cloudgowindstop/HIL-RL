"""Compact GPFS/storage diagnostics with a durable /tmp mirror.

The monitor is opt-in.  Periodic samples only inspect statvfs and process/cgroup
memory.  Explicit probes allocate and fsync a small file, so they can detect a
filesystem that reports aggregate free space but currently refuses allocation.
"""

from __future__ import annotations

import errno
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory_monitor import _process_memory, _read_int, _read_key_values


_MONITOR: "StorageMonitor | None" = None
_STORAGE_ERRNOS = {errno.ENOSPC, errno.EDQUOT, errno.EIO}


def _mount_info(path: Path) -> dict[str, str | None]:
    """Return longest matching mount from /proc/self/mountinfo."""
    resolved = str(path.resolve())
    best: tuple[int, dict[str, str | None]] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return {"mount_point": None, "filesystem_type": None, "mount_source": None}
    for line in lines:
        fields = line.split()
        if "-" not in fields or len(fields) < 7:
            continue
        separator = fields.index("-")
        mount_point = fields[4].replace("\\040", " ")
        if resolved != mount_point and not resolved.startswith(mount_point.rstrip("/") + "/"):
            continue
        record = {
            "mount_point": mount_point,
            "filesystem_type": fields[separator + 1] if separator + 1 < len(fields) else None,
            "mount_source": fields[separator + 2] if separator + 2 < len(fields) else None,
        }
        candidate = (len(mount_point), record)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best else {
        "mount_point": None,
        "filesystem_type": None,
        "mount_source": None,
    }


def _filesystem_snapshot(path: Path) -> dict[str, Any]:
    stat = os.statvfs(path)
    fragment_size = stat.f_frsize or stat.f_bsize
    total = stat.f_blocks * fragment_size
    free = stat.f_bfree * fragment_size
    available = stat.f_bavail * fragment_size
    used = total - free
    return {
        "storage_path": str(path),
        "storage_block_size": fragment_size,
        "storage_total_bytes": total,
        "storage_used_bytes": used,
        "storage_free_bytes": free,
        "storage_available_bytes": available,
        "storage_used_percent": (100.0 * used / total) if total else None,
        "storage_total_inodes": stat.f_files,
        "storage_free_inodes": stat.f_ffree,
        "storage_available_inodes": stat.f_favail,
        **_mount_info(path),
    }


class StorageMonitor:
    """Small JSONL monitor whose primary log remains writable when GPFS fails."""

    def __init__(self, output_dir: Path, interval_s: float, probe_mib: int):
        self.output_dir = Path(output_dir)
        self.interval_s = interval_s
        self.probe_bytes = probe_mib * 1024 * 1024
        self.local_path = Path("/tmp") / f"cosmos_storage_{os.getpid()}.jsonl"
        self.output_path = self.output_dir / "logs" / "storage_events.jsonl"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "monitor_start"
        self._context: dict[str, Any] = {}
        self._output_warning_printed = False

    def _snapshot(self, kind: str, stage: str, **extra: Any) -> dict[str, Any]:
        rss, vms = _process_memory()
        memory_events = _read_key_values("/sys/fs/cgroup/memory.events")
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_s": round(time.monotonic(), 6),
            "kind": kind,
            "stage": stage,
            "pid": os.getpid(),
            "rss_bytes": rss,
            "vms_bytes": vms,
            "cgroup_current_bytes": _read_int("/sys/fs/cgroup/memory.current"),
            "cgroup_limit_bytes": _read_int("/sys/fs/cgroup/memory.max"),
            "cgroup_oom": memory_events.get("oom"),
            "cgroup_oom_kill": memory_events.get("oom_kill"),
            **self._context,
            **extra,
        }
        try:
            record.update(_filesystem_snapshot(self.output_dir))
        except OSError as exc:
            record.update({
                "storage_snapshot_errno": exc.errno,
                "storage_snapshot_error": str(exc),
            })
        return record

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
        except OSError as exc:
            if not self._output_warning_printed:
                self._output_warning_printed = True
                print(
                    f"[STORAGE] output log unavailable: {exc}; local log={self.local_path}",
                    flush=True,
                )

    def start(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.failure("monitor_start_failed", exc)
            raise
        self.event("monitor_start", interval_s=self.interval_s, probe_bytes=self.probe_bytes)
        self._thread = threading.Thread(
            target=self._sample_loop, name="storage-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.event("monitor_stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s + 1.0))
        print(f"[STORAGE] durable local log: {self.local_path}", flush=True)

    def set_context(self, **context: Any) -> None:
        with self._lock:
            self._context = {**self._context, **context}

    def event(self, stage: str, **extra: Any) -> None:
        with self._lock:
            self._stage = stage
            record = self._snapshot("event", stage, **extra)
            self._append(record)
        available = record.get("storage_available_bytes")
        used_percent = record.get("storage_used_percent")
        available_text = "n/a" if available is None else f"{available / 1024**4:.2f}TiB"
        used_text = "n/a" if used_percent is None else f"{used_percent:.1f}%"
        print(
            f"[STORAGE] {stage}: available={available_text} used={used_text}",
            flush=True,
        )

    def probe(self, stage: str) -> None:
        probe_path = self.output_dir / f".cosmos_storage_probe_{os.getpid()}"
        started = time.monotonic()
        try:
            with probe_path.open("wb") as handle:
                remaining = self.probe_bytes
                block = bytes(min(1024 * 1024, self.probe_bytes))
                while remaining > 0:
                    chunk = block if remaining >= len(block) else block[:remaining]
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            self.event(
                stage,
                probe_status="success",
                probe_bytes=self.probe_bytes,
                probe_elapsed_s=round(time.monotonic() - started, 6),
            )
        except OSError as exc:
            self.failure(
                stage,
                exc,
                probe_status="failed",
                probe_bytes=self.probe_bytes,
                probe_elapsed_s=round(time.monotonic() - started, 6),
            )
            raise
        finally:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass

    def failure(self, stage: str, error: BaseException, **extra: Any) -> None:
        details: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "errno": getattr(error, "errno", None),
            **extra,
        }
        try:
            parquets = list(self.output_dir.rglob("*.parquet"))
            details.update({
                "parquet_count": len(parquets),
                "zero_byte_parquets": [str(path) for path in parquets if path.stat().st_size == 0],
                "partial_files": [str(path) for path in self.output_dir.rglob("*.partial")],
            })
            info_path = self.output_dir / "meta" / "info.json"
            if info_path.is_file():
                info = json.loads(info_path.read_text())
                details["metadata_total_episodes"] = info.get("total_episodes")
                details["metadata_total_frames"] = info.get("total_frames")
        except (OSError, ValueError, json.JSONDecodeError) as inspect_error:
            details["output_inspection_error"] = str(inspect_error)
        with self._lock:
            self._stage = stage
            record = self._snapshot("failure", stage, **details)
            self._append(record)
        print(
            f"[STORAGE][ERROR] {stage}: {error}; durable log={self.local_path}",
            flush=True,
        )

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                with self._lock:
                    self._append(self._snapshot("sample", self._stage))
            except Exception:
                # Monitoring must not terminate conversion. Explicit probes still raise.
                pass


def start_storage_monitor(
    enabled: bool, output_dir: Path, interval_s: float, probe_mib: int
) -> None:
    global _MONITOR
    if not enabled:
        return
    _MONITOR = StorageMonitor(output_dir, interval_s, probe_mib)
    _MONITOR.start()


def stop_storage_monitor() -> None:
    global _MONITOR
    if _MONITOR is not None:
        _MONITOR.stop()
        _MONITOR = None


def set_storage_context(**context: Any) -> None:
    if _MONITOR is not None:
        _MONITOR.set_context(**context)


def storage_event(stage: str, **extra: Any) -> None:
    if _MONITOR is not None:
        _MONITOR.event(stage, **extra)


def storage_probe(stage: str) -> None:
    if _MONITOR is not None:
        _MONITOR.probe(stage)


def storage_failure(stage: str, error: BaseException, **extra: Any) -> None:
    if _MONITOR is not None:
        _MONITOR.failure(stage, error, **extra)


def is_storage_error(error: BaseException) -> bool:
    return isinstance(error, OSError) and error.errno in _STORAGE_ERRNOS
