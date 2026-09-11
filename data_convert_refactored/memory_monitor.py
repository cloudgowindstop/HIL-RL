"""低开销、可选的进程RSS和容器cgroup内存监控。"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MONITOR: "MemoryMonitor | None" = None


def _read_int(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _read_key_values(path: str) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, value = line.split(maxsplit=1)
            result[key] = int(value)
    except (OSError, ValueError):
        pass
    return result


def _process_memory() -> tuple[int | None, int | None]:
    """直接读取/proc返回RSS/VMS字节，避免新增psutil依赖。"""
    try:
        fields = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSize:")):
                key, value, _unit = line.split()
                fields[key.rstrip(":")] = int(value) * 1024
        return fields.get("VmRSS"), fields.get("VmSize")
    except (OSError, ValueError):
        return None, None


class MemoryMonitor:
    """同时记录关键事件JSONL和固定间隔CSV采样；监控失败不得中断转换。"""

    def __init__(self, log_dir: Path, interval_s: float):
        self.log_dir = Path(log_dir)
        self.interval_s = interval_s
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "memory_events.jsonl"
        self.samples_path = self.log_dir / "memory_samples.csv"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "monitor_start"
        self._context: dict[str, Any] = {}
        self._warning_printed = False

    def _snapshot(self, kind: str, stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        rss, vms = _process_memory()
        memory_stat = _read_key_values("/sys/fs/cgroup/memory.stat")
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
            "cgroup_peak_bytes": _read_int("/sys/fs/cgroup/memory.peak"),
            "cgroup_anon_bytes": memory_stat.get("anon"),
            "cgroup_file_bytes": memory_stat.get("file"),
            "cgroup_shmem_bytes": memory_stat.get("shmem"),
            "cgroup_oom": memory_events.get("oom"),
            "cgroup_oom_kill": memory_events.get("oom_kill"),
            **self._context,
        }
        if extra:
            record.update(extra)
        return record

    def start(self) -> None:
        """初始化日志并启动daemon采样线程。"""
        header_needed = not self.samples_path.exists() or self.samples_path.stat().st_size == 0
        if header_needed:
            with self.samples_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([
                    "timestamp", "stage", "episode_index", "episode_frames", "rss_bytes",
                    "cgroup_current_bytes", "cgroup_limit_bytes", "cgroup_anon_bytes",
                    "cgroup_file_bytes", "cgroup_shmem_bytes", "cgroup_oom_kill",
                ])
        self.event("monitor_start", interval_s=self.interval_s)
        self._thread = threading.Thread(target=self._sample_loop, name="memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """记录最终事件并等待采样线程退出。"""
        self.event("monitor_stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s + 1.0))

    def set_context(self, **context: Any) -> None:
        with self._lock:
            self._context = {**self._context, **context}

    def event(self, stage: str, **extra: Any) -> None:
        """在业务关键节点同步记录一次内存快照。"""
        try:
            with self._lock:
                self._stage = stage
                record = self._snapshot("event", stage, extra)
                with self.events_path.open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    handle.flush()
            rss = record.get("rss_bytes")
            current = record.get("cgroup_current_bytes")
            limit = record.get("cgroup_limit_bytes")
            rss_text = "n/a" if rss is None else f"{rss / 1024**3:.2f}GiB"
            cg_text = "n/a" if current is None else f"{current / 1024**3:.2f}GiB"
            pct_text = "" if current is None or not limit else f" ({100 * current / limit:.1f}%)"
            print(f"[MEM] {stage}: RSS={rss_text} cgroup={cg_text}{pct_text}", flush=True)
        except Exception as exc:  # 监控是诊断功能，任何异常都不能中断正式转换。
            if not self._warning_printed:
                self._warning_printed = True
                print(f"[WARNING] memory monitor disabled event after error: {exc}", flush=True)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                with self._lock:
                    record = self._snapshot("sample", self._stage)
                    row = [
                        record.get("timestamp"), record.get("stage"), record.get("episode_index"),
                        record.get("episode_frames"), record.get("rss_bytes"),
                        record.get("cgroup_current_bytes"), record.get("cgroup_limit_bytes"),
                        record.get("cgroup_anon_bytes"), record.get("cgroup_file_bytes"),
                        record.get("cgroup_shmem_bytes"), record.get("cgroup_oom_kill"),
                    ]
                    with self.samples_path.open("a", newline="") as handle:
                        csv.writer(handle).writerow(row)
                        handle.flush()
            except Exception:
                pass


def start_memory_monitor(enabled: bool, log_dir: Path, interval_s: float) -> None:
    """启用全局监控器；关闭状态不创建目录或线程。"""
    global _MONITOR
    if not enabled:
        return
    _MONITOR = MemoryMonitor(log_dir, interval_s)
    _MONITOR.start()


def stop_memory_monitor() -> None:
    """停止并清除全局监控器。"""
    global _MONITOR
    if _MONITOR is not None:
        _MONITOR.stop()
        _MONITOR = None


def set_memory_context(**context: Any) -> None:
    """更新后续事件携带的episode编号、路径和帧数。"""
    if _MONITOR is not None:
        _MONITOR.set_context(**context)


def memory_event(stage: str, **extra: Any) -> None:
    """记录命名业务节点；监控未启用时为空操作。"""
    if _MONITOR is not None:
        _MONITOR.event(stage, **extra)


def tensor_nbytes(value: Any) -> int | None:
    """计算Tensor逻辑字节数；不复制Tensor，也不触发GPU到CPU传输。"""
    try:
        return int(value.numel()) * int(value.element_size())
    except (AttributeError, TypeError, ValueError):
        return None
