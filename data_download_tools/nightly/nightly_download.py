#!/usr/bin/env python3
"""七台工厂天轶：每天扫 BOS，把还没有的批次下到本地。

目录约定（天轶 + 任务 + 数据）::

    {output}/{robot_type}/{task_id}/{batch}/

流程：加锁 → 列 BOS → 扫本地已有 → 下载缺口 → 写账本。
只下载，不整理 schema、不改其它目录。

每天 22:00 自动跑（本机没有 cron，用 ``daily`` 常驻进程）::

    PYTHONUNBUFFERED=1 nohup python nightly_download.py daily \\
      >> /media/jushen/project-rl-dataset/raw_data_0909_autoupdate/_metadata/cron.log 2>&1 &
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import h5py


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ROBOT_IDS = (128, 134, 193, 368, 394, 468, 515)
ROBOT_TYPES = tuple(f"tienyi_prod2_dualArm-gripper-3cameras_{n}" for n in ROBOT_IDS)
ROBOT_PREFIX = "tienyi_prod2_dualArm-gripper-3cameras_"
UNKNOWN_TASK = "unknown_task"

OUTPUT_ROOT = Path("/media/jushen/project-rl-dataset/raw_data_0909_autoupdate")
MAPPING_PATH = Path(__file__).resolve().parent / "raw_data_mapping.json"
BCECMD = Path("/media/linux-bcecmd-0.5.1/bcecmd")
BOS_BASE = "bos:/bd-dp-ten-6spt6-scjd"

# 已有数据根：用来跳过白天已经下过的批次，不会往这些目录里写。
ALREADY_ROOTS = (
    Path("/media/jushen/project-rl-dataset/raw_data_0804download"),
    Path("/media/jushen/project-rl-dataset/raw_data_0831download_by_schema"),
    Path("/media/jushen/project-rl-dataset/cosmos_raw_data"),
)
RESULT_LOGS = (
    Path(
        "/media/jushen/project-rl-dataset/raw_data_0831download_by_schema"
        "/_metadata/download_state/download_results.jsonl"
    ),
    Path(
        "/media/jushen/project-rl-dataset/raw_data_0831download_by_schema"
        "/_metadata/download_state/incremental_20260903/download_results.jsonl"
    ),
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
MIN_FREE_BYTES = 200 * 1024**3
SKIP_DIR_SUFFIXES = ("_left_gripper_cut",)
OK_STATUSES = {"downloaded", "skipped_complete"}
BATCH_NAME_RE = re.compile(r"^(tienyi_prod2_dualArm-gripper-3cameras_\d+)")
DATE_SUFFIX_RE = re.compile(r"_\d{8}(_.*)?$")
DATE_IN_NAME_RE = re.compile(r"(20\d{6})")
PRE_RE = re.compile(r"\bPRE\s+(\S+)")
# 只下 6 月及之后：批次名里的日期 >= 20260601。没有日期的 test 等跳过。
MIN_BATCH_DATE = "20260601"


@dataclass(frozen=True)
class Batch:
    """一个待处理或已列出的 BOS 批次。"""

    bos_path: str
    robot_type: str
    task_id: str
    batch: str
    destination: Path


# ---------------------------------------------------------------------------
# 时间、路径、任务
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """当前 UTC 时间。

    输入: 无。
    输出: ISO 时间字符串，写入账本用。
    """
    return datetime.now(timezone.utc).isoformat()


def shanghai_today() -> str:
    """上海时区的当天日期。

    输入: 无。
    输出: ``YYYY-MM-DD``，用作一次夜间任务的 run id。
    """
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d")


def wait_until(target: datetime) -> None:
    """等到上海时区的目标时间再返回。本机没有 cron，定时等在这里做。

    输入:
        target: 带时区的目标时刻。
    输出: 无；到点或已过点则立刻返回。
    """
    if target.tzinfo is None:
        target = target.replace(tzinfo=SHANGHAI)
    while True:
        remain = (target - datetime.now(SHANGHAI)).total_seconds()
        if remain <= 0:
            return
        print(f"waiting until {target.isoformat()} ({remain:.0f}s left)", flush=True)
        time.sleep(min(remain, 30))


def resolve_wait(after_minutes: int, wait_until_hhmm: str | None) -> datetime | None:
    """把命令行上的定时参数换成一个目标时刻。

    输入:
        after_minutes: 从现在起再等多少分钟，0 表示不等。
        wait_until_hhmm: ``HH:MM``，今天这个点；已过则等到明天。
    输出:
        要等到的时刻；两个参数都空则 ``None``（立刻跑）。
    """
    now = datetime.now(SHANGHAI)
    if after_minutes > 0:
        return now + timedelta(minutes=after_minutes)
    if not wait_until_hhmm:
        return None
    hour, minute = (int(part) for part in wait_until_hhmm.split(":", 1))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def load_task_patterns(path: Path) -> dict[str, str]:
    """读取 mapping 里的路径片段 → 任务 id。

    输入:
        path: ``raw_data_mapping.json``。
    输出:
        例如 ``{"back-handle-installation": "back_handle"}``。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    patterns = payload.get("path_task_patterns") or {}
    if not isinstance(patterns, dict):
        raise ValueError(f"{path} 的 path_task_patterns 必须是对象")
    return {str(key): str(value) for key, value in patterns.items()}


def task_slug_from_batch(batch: str, robot_type: str) -> str:
    """从批次名里抠任务段：去掉机器前缀和日期后缀。

    输入:
        batch: 例如 ``..._128_Plug_in_the_charging_cable_20260506``。
        robot_type: 当前机器目录，例如 ``tienyi_prod2_dualArm-gripper-3cameras_128``。
    输出:
        任务 slug；抠不出来则空字符串。
    """
    rest = batch
    prefix = f"{robot_type}_"
    if rest.startswith(prefix):
        rest = rest[len(prefix) :]
    else:
        match = BATCH_NAME_RE.match(rest)
        if match and rest.startswith(f"{match.group(1)}_"):
            rest = rest[len(match.group(1)) + 1 :]
    rest = DATE_SUFFIX_RE.sub("", rest).strip("_")
    if rest in {"", robot_type} or rest.startswith(ROBOT_PREFIX):
        return ""
    return rest


def task_id_of(batch: str, robot_type: str, patterns: dict[str, str]) -> str:
    """用批次名判断任务。

    输入:
        batch: BOS 批次目录名。
        robot_type: 该批次所在机器目录。
        patterns: ``load_task_patterns`` 的结果。
    输出:
        对上 mapping 用短名（如 ``plug_cables``）；否则用批次里的任务 slug；
        两边都读不出来才是 ``unknown_task``。
    """
    hits = [task for needle, task in patterns.items() if needle.lower() in batch.lower()]
    if len(hits) == 1:
        return hits[0]
    return task_slug_from_batch(batch, robot_type) or UNKNOWN_TASK


def batch_ymd(batch: str) -> str | None:
    """从批次名取出采集日期。

    输入: 批次目录名。
    输出: ``YYYYMMDD``；没有日期则为 ``None``。
    """
    found = DATE_IN_NAME_RE.findall(batch)
    return found[-1] if found else None


def keep_batch(batch: str, min_date: str) -> bool:
    """是否属于要下载的时间范围。

    输入:
        batch: 批次名。
        min_date: 下限 ``YYYYMMDD``；空字符串表示不按日期过滤。
    输出:
        有日期且 ``>= min_date`` 为 True；没日期或更早为 False。
    """
    if not min_date:
        return True
    ymd = batch_ymd(batch)
    return ymd is not None and ymd >= min_date


def destination_for(output: Path, robot_type: str, task_id: str, batch: str) -> Path:
    """拼本地落盘路径：天轶 / 任务 / 数据。

    输入: 输出根、机器目录名、任务 id、批次名。
    输出: ``output/robot_type/task_id/batch``。
    """
    return output / robot_type / task_id / batch


def bos_uri(bos_path: str) -> str:
    """把 ``BOS::raw_data/...`` 转成 bcecmd 用的 ``bos:/bucket/...``。

    输入: 内部使用的 BOS 批次根。
    输出: ``bos:/bd-dp-ten-6spt6-scjd/raw_data/{robot}/{batch}``。
    """
    text = bos_path.strip()
    if text.startswith("BOS::"):
        text = text[len("BOS::") :]
    return f"{BOS_BASE.rstrip('/')}/{text.lstrip('/')}"


def bos_path_from_batch(batch: str) -> str | None:
    """从批次目录名还原 BOS 根。

    输入: 例如 ``tienyi_prod2_dualArm-gripper-3cameras_368_back-handle-installation_20260907``。
    输出: ``BOS::raw_data/{robot_type}/{batch}``；名字对不上则 ``None``。
    """
    match = BATCH_NAME_RE.match(batch)
    if match is None:
        return None
    return f"BOS::raw_data/{match.group(1)}/{batch}"


def robot_of(bos_path: str) -> str | None:
    """取出 BOS 根里的机器目录名。

    输入: ``BOS::raw_data/{robot}/{batch}``。
    输出: ``robot``；格式不对则 ``None``。
    """
    parts = bos_path.replace("\\", "/").split("/")
    if len(parts) < 3:
        return None
    return parts[-2]


# ---------------------------------------------------------------------------
# 本地已有批次
# ---------------------------------------------------------------------------

def is_successful_dir(directory: Path) -> bool:
    """判断一个批次目录算不算已经下成功。

    输入: 本地批次目录。
    输出: 有 ``trajectory.hdf5`` 或完成标记则为 True。
    """
    if (directory / ".download_complete.json").is_file() or (directory / ".download_complete").is_file():
        return True
    return any(directory.rglob("trajectory.hdf5"))


def walk_successful_batches(root: Path) -> set[str]:
    """在一个本地根下找出所有已成功的工厂机批次。

    扁平目录、schema 目录、夜间三层目录都能扫：走到批次名就停，不再往里走。

    输入:
        root: 例如 ``cosmos_raw_data`` 或 ``raw_data_0909_autoupdate``。
    输出:
        已成功批次的 BOS 根集合。目录不存在则空集。
    """
    found: set[str] = set()
    if not root.is_dir():
        return found
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name != "_metadata" and not name.endswith(SKIP_DIR_SUFFIXES) and not name.endswith(".partial")
        ]
        batch = Path(dirpath).name
        if not batch.startswith(ROBOT_PREFIX):
            continue
        # 机器目录本身也叫 tienyi_..._{id}，不是批次，要继续往任务/批次里走。
        if BATCH_NAME_RE.fullmatch(batch):
            continue
        dirnames.clear()
        bos_path = bos_path_from_batch(batch)
        if bos_path and robot_of(bos_path) in ROBOT_TYPES and is_successful_dir(Path(dirpath)):
            found.add(bos_path)
    return found


def load_successful_from_logs(paths: list[Path]) -> set[str]:
    """从历史 download_results.jsonl 读出已成功的 BOS 根。

    输入:
        paths: jsonl 路径列表；文件不存在则跳过。
    输出:
        status 为 downloaded/skipped_complete 且 hdf5≥1 的批次根。
    """
    found: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") not in OK_STATUSES:
                    continue
                if int(record.get("hdf5_count") or 0) < 1:
                    continue
                bos_path = str(record.get("bos_path") or "").strip().rstrip("/")
                if robot_of(bos_path) in ROBOT_TYPES:
                    found.add(bos_path)
    return found


def already_downloaded(output: Path, extra_roots: list[Path], result_logs: list[Path]) -> set[str]:
    """汇总「不用再下」的批次。

    输入:
        output: 夜间下载根。
        extra_roots: 0804 / 0831 / cosmos 等历史根。
        result_logs: 历史 jsonl 账本。
    输出:
        已成功 BOS 根的并集。
    """
    found = walk_successful_batches(output)
    for root in extra_roots:
        found |= walk_successful_batches(root)
    found |= load_successful_from_logs(result_logs)
    return found


# ---------------------------------------------------------------------------
# BOS 列表
# ---------------------------------------------------------------------------

def parse_ls(text: str) -> list[str]:
    """解析 ``bos ls --all`` 的目录行。

    输入: bcecmd 标准输出。
    输出: 子目录名列表（去掉末尾 ``/``）。
    """
    names: list[str] = []
    for line in text.splitlines():
        match = PRE_RE.search(line)
        if match:
            names.append(match.group(1).rstrip("/"))
    return names


def list_bos_dir(bcecmd: Path, uri: str) -> list[str]:
    """列 BOS 上某一层的子目录，不递归。

    输入:
        bcecmd: bcecmd 可执行文件。
        uri: 例如 ``bos:/bucket/raw_data/{robot}/``。
    输出:
        该层目录名。失败则抛 ``RuntimeError``。
    """
    completed = subprocess.run(
        [str(bcecmd), "bos", "ls", "--all", uri],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip() or f"exit {completed.returncode}"
        raise RuntimeError(f"bos ls 失败 {uri}: {detail}")
    return parse_ls(completed.stdout)


def list_factory_batches(
    bcecmd: Path,
    patterns: dict[str, str],
    list_dir: Callable[[Path, str], list[str]],
) -> list[Batch]:
    """列出七台机器在 BOS 上的全部批次。

    输入:
        bcecmd: bcecmd 路径。
        patterns: 任务匹配表。
        list_dir: 列目录函数，正式跑用 ``list_bos_dir``，自测可注入假数据。
    输出:
        七台机器下的批次列表（此时 ``destination`` 仍为空路径）。
    """
    batches: list[Batch] = []
    for robot_type in ROBOT_TYPES:
        uri = f"{BOS_BASE.rstrip('/')}/raw_data/{robot_type}/"
        try:
            names = list_dir(bcecmd, uri)
        except RuntimeError as error:
            print(f"skip list {robot_type}: {error}", file=sys.stderr)
            continue
        for batch in names:
            batches.append(
                Batch(
                    bos_path=f"BOS::raw_data/{robot_type}/{batch}",
                    robot_type=robot_type,
                    task_id=task_id_of(batch, robot_type, patterns),
                    batch=batch,
                    destination=Path(),
                )
            )
    return batches


# ---------------------------------------------------------------------------
# 下载与校验
# ---------------------------------------------------------------------------

def verify_hdf5(directory: Path) -> tuple[int, int, list[str]]:
    """打开目录里每个 ``trajectory.hdf5``；打不开的跳过，不让整批失败。

    输入: 刚 sync 下来的批次目录（或已完成目录）。
    输出:
        ``(能读的文件数, 能读的总字节, 被跳过的坏文件说明)``。
        一个都读不出来时，第三项会带 ``no readable trajectory.hdf5``。
    """
    files = sorted(directory.rglob("trajectory.hdf5"))
    skipped: list[str] = []
    total = 0
    readable = 0
    for path in files:
        try:
            size = path.stat().st_size
            with h5py.File(path, "r") as handle:
                list(handle.keys())
            total += size
            readable += 1
        except Exception as error:
            skipped.append(f"{path}: {error}")
    if readable == 0:
        skipped.append("no readable trajectory.hdf5 found")
    return readable, total, skipped


def append_jsonl(path: Path, record: dict) -> None:
    """往账本追加一行 JSON。

    输入:
        path: ``download_results.jsonl``。
        record: 一条下载结果。
    输出: 无；写盘并 fsync。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def download_one(
    item: Batch,
    *,
    bcecmd: Path,
    logs_dir: Path,
    resume: bool,
    sync: Callable[[list[str], Path], int] | None,
) -> dict:
    """下载一个批次：先写 ``.partial``，校验通过再改成正式目录。

    输入:
        item: 带好 ``destination`` 的批次。
        bcecmd: bcecmd 路径。
        logs_dir: 每个批次一份 ``{batch}.log``。
        resume: 已有 ``.partial`` 时是否接着 sync。
        sync: 正式跑为 ``None``（调 bcecmd）；自测注入假下载。
    输出:
        一条结果字典，含 ``status`` / ``hdf5_count`` / ``error``。
    """
    destination = item.destination
    partial = destination.with_name(destination.name + ".partial")
    record = {
        "bos_path": item.bos_path,
        "robot_type": item.robot_type,
        "task_id": item.task_id,
        "batch": item.batch,
        "destination": str(destination),
        "hdf5_count": 0,
        "total_bytes": 0,
        "error": None,
        "completed_at": utc_now(),
    }
    if destination.is_dir() and (destination / ".download_complete.json").is_file():
        count, total, skipped = verify_hdf5(destination)
        record.update(hdf5_count=count, total_bytes=total, status="skipped_complete")
        if skipped:
            record["skipped_bad_hdf5"] = skipped
        if count < 1:
            record.update(status="failed_verification", error="; ".join(skipped))
        return record
    if partial.exists() and not resume:
        record.update(status="failed", error="partial directory exists; use --resume")
        return record

    partial.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{item.batch}.log"
    command = [str(bcecmd), "bos", "sync", bos_uri(item.bos_path), str(partial)]
    try:
        if sync is None:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{utc_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        else:
            code = sync(command, partial)
    except OSError as error:
        record.update(status="failed", error=str(error))
        return record
    if code != 0:
        record.update(status="failed", error=f"bcecmd exited with {code}; log={log_path}")
        return record

    count, total, skipped = verify_hdf5(partial)
    record.update(hdf5_count=count, total_bytes=total)
    if skipped:
        record["skipped_bad_hdf5"] = skipped
    if count < 1:
        record.update(status="failed_verification", error="; ".join(skipped))
        return record
    marker = {
        "bos_path": item.bos_path,
        "robot_type": item.robot_type,
        "task_id": item.task_id,
        "hdf5_count": count,
        "total_bytes": total,
        "completed_at": utc_now(),
    }
    if skipped:
        marker["skipped_bad_hdf5"] = skipped
    (partial / ".download_complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(partial, destination)
    record.update(status="downloaded", destination=str(destination))
    return record


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def write_run_summary(state_dir: Path, run_id: str, payload: dict) -> Path:
    """写当天的运行摘要。

    输入:
        state_dir: ``output/_metadata``。
        run_id: ``YYYY-MM-DD``。
        payload: 摘要内容。
    输出: 写成的 ``runs/{run_id}.json`` 路径。
    """
    path = state_dir / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_run_status(state_dir: Path, **fields: object) -> Path:
    """更新 ``current_run.json``，给终端 ``status`` 和启动脚本读。

    输入: ``_metadata`` 目录，以及要合并进去的字段。
    输出: 写成的路径。
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "current_run.json"
    current: dict = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    current["updated_at"] = utc_now()
    current.setdefault("pid", os.getpid())
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def pid_alive(pid: object) -> bool:
    """进程号是否还在。

    输入: ``current_run.json`` 里的 pid。
    输出: 还在为 True。
    """
    try:
        number = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    try:
        os.kill(number, 0)
    except OSError:
        return False
    return True


def print_run_status(output: Path) -> int:
    """把 ``current_run.json`` 打成可读的一段。

    输入: 输出根（默认夜间下载目录）。
    输出: 0 有状态文件，1 还没跑过。
    """
    path = output / "_metadata" / "current_run.json"
    if not path.is_file():
        print("还没有跑过，没有 current_run.json")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    pid = data.get("pid")
    alive = "运行中" if pid_alive(pid) else "已结束"
    phase = str(data.get("phase") or "?")
    queue = list(data.get("queue") or [])
    done = list(data.get("done") or [])
    current = data.get("current")
    total = int(data.get("new_batches") or len(queue) or 0)
    finished = len(done)
    print(f"pid={pid}  {alive}  mode={data.get('mode', '?')}  phase={phase}")
    if data.get("next_at"):
        print(f"下次触发: {data['next_at']}")
    if data.get("bos_batches") is not None:
        print(
            f"BOS {data.get('bos_batches')} 个批次，"
            f"本地已有 {data.get('already')}，"
            f"日期太旧跳过 {data.get('skipped_old')}，"
            f"本轮要下 {total} 个"
            + (f"（已完成 {finished}/{total}）" if total and phase == "downloading" else "")
        )
    if current:
        print(f"正在下: {current}")
    if data.get("error"):
        print(f"错误: {data['error']}")
    if queue:
        print("待下清单:")
        done_set = set(done)
        for item in queue:
            if item in done_set:
                mark = "x"
            elif item == current:
                mark = ">"
            else:
                mark = " "
            print(f"  [{mark}] {item}")
    elif phase in {"planned", "done"} and total == 0:
        print("没有新批次")
    if data.get("summary"):
        print(f"summary: {data['summary']}")
    print(f"状态文件: {path}")
    return 0


def run_nightly(
    *,
    mapping: Path,
    output: Path,
    bcecmd: Path,
    extra_roots: list[Path],
    result_logs: list[Path],
    dry_run: bool,
    resume: bool,
    min_free_bytes: int,
    min_date: str = MIN_BATCH_DATE,
    list_dir: Callable[[Path, str], list[str]] | None = None,
    sync: Callable[[list[str], Path], int] | None = None,
    mode: str = "run",
) -> int:
    """跑一轮夜间下载。

    输入: 见参数名；``list_dir`` / ``sync`` 仅自测注入。
    输出: 0 成功，1 失败（锁冲突、磁盘不够、或有批次校验失败）。
    """
    patterns = load_task_patterns(mapping)
    state_dir = output / "_metadata"
    state_dir.mkdir(parents=True, exist_ok=True)
    write_run_status(state_dir, pid=os.getpid(), mode=mode, phase="scanning", current=None, queue=[], done=[])
    print(f"pid={os.getpid()} mode={mode} scanning BOS and local", flush=True)
    lock_path = state_dir / "nightly.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another nightly_download run is in progress", file=sys.stderr)
            write_run_status(state_dir, phase="locked", error="another nightly_download run is in progress")
            return 1

        already = already_downloaded(output, extra_roots, result_logs)
        listed = list_factory_batches(bcecmd, patterns, list_dir or list_bos_dir)
        skipped_old = sum(1 for item in listed if not keep_batch(item.batch, min_date))
        missing = [
            Batch(
                bos_path=item.bos_path,
                robot_type=item.robot_type,
                task_id=item.task_id,
                batch=item.batch,
                destination=destination_for(output, item.robot_type, item.task_id, item.batch),
            )
            for item in listed
            if item.bos_path not in already and keep_batch(item.batch, min_date)
        ]

        run_id = shanghai_today()
        queue = [f"{item.robot_type}/{item.task_id}/{item.batch}" for item in missing]
        print(f"bos batches: {len(listed)}")
        print(f"skipped before {min_date or 'none'}: {skipped_old}")
        print(f"already downloaded: {len(already)}")
        print(f"new batches: {len(missing)}")
        for item in missing:
            print(f"  {item.robot_type}/{item.task_id}/{item.batch}")
        write_run_status(
            state_dir,
            phase="planned",
            bos_batches=len(listed),
            already=len(already),
            skipped_old=skipped_old,
            new_batches=len(missing),
            queue=queue,
            done=[],
            current=None,
            error=None,
        )

        results: list[dict] = []
        if dry_run:
            status_counts = {"dry_run": len(missing)}
        else:
            usage = os.statvfs(output)
            free = usage.f_bavail * usage.f_frsize
            if free < min_free_bytes:
                print(f"not enough disk space: {free} bytes free", file=sys.stderr)
                write_run_summary(
                    state_dir,
                    run_id,
                    {
                        "run_id": run_id,
                        "status": "skipped_disk",
                        "bos_batches": len(listed),
                        "already": len(already),
                        "new_batches": len(missing),
                        "free_bytes": free,
                    },
                )
                write_run_status(state_dir, phase="skipped_disk", error=f"free_bytes={free}")
                return 1
            for item in missing:
                key = f"{item.robot_type}/{item.task_id}/{item.batch}"
                write_run_status(state_dir, phase="downloading", current=key, done=[r.get("key") for r in results])
                print(f"download {item.task_id}/{item.batch}")
                try:
                    record = download_one(
                        item,
                        bcecmd=bcecmd,
                        logs_dir=state_dir / "logs",
                        resume=resume,
                        sync=sync,
                    )
                except Exception as error:
                    record = {
                        "bos_path": item.bos_path,
                        "robot_type": item.robot_type,
                        "task_id": item.task_id,
                        "batch": item.batch,
                        "destination": str(item.destination),
                        "hdf5_count": 0,
                        "total_bytes": 0,
                        "status": "failed",
                        "error": str(error),
                    }
                record["run_id"] = run_id
                append_jsonl(state_dir / "download_results.jsonl", record)
                record["key"] = key
                results.append(record)
                print(f"  {record['status']} hdf5={record['hdf5_count']}")
                write_run_status(
                    state_dir,
                    phase="downloading",
                    current=None,
                    done=[str(row.get("key") or "") for row in results],
                    last_status=record["status"],
                )
            status_counts: dict[str, int] = {}
            for record in results:
                status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

        summary_path = write_run_summary(
            state_dir,
            run_id,
            {
                "run_id": run_id,
                "generated_at": utc_now(),
                "output": str(output),
                "bos_batches": len(listed),
                "already": len(already),
                "new_batches": len(missing),
                "status_counts": status_counts,
                "new_paths": [item.bos_path for item in missing],
            },
        )
        print(f"summary: {summary_path}")
        failed = sum(1 for record in results if str(record.get("status", "")).startswith("failed"))
        write_run_status(
            state_dir,
            phase="done",
            current=None,
            done=[str(row.get("key") or "") for row in results] or queue,
            summary=str(summary_path),
            exit_code=1 if failed else 0,
            status_counts=status_counts,
        )
        return 1 if failed else 0


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _write_tiny_hdf5(path: Path) -> None:
    """自测用：写一个能被 h5py 打开的小文件。

    输入: 目标 ``trajectory.hdf5`` 路径。
    输出: 无。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=[0])


def run_self_test() -> int:
    """不连 BOS，用临时目录验证：已有批次跳过、新批次落到 机器/任务/数据。

    输入: 无。
    输出: 0 通过；断言失败则进程退出。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        mapping = root / "mapping.json"
        mapping.write_text(
            json.dumps({"path_task_patterns": {"back-handle-installation": "back_handle"}}),
            encoding="utf-8",
        )
        robot = "tienyi_prod2_dualArm-gripper-3cameras_368"
        patterns = {"back-handle-installation": "back_handle"}
        assert task_id_of(f"{robot}_back-handle-installation_20260907", robot, patterns) == "back_handle"
        assert task_id_of(f"{robot}_Plug_in_the_charging_cable_20260506", robot, patterns) == "Plug_in_the_charging_cable"
        assert task_id_of(f"{robot}_material_packaging_new_20260612", robot, patterns) == "material_packaging_new"
        assert task_id_of(f"{robot}_test", robot, patterns) == "test"
        assert task_id_of("urdf_test", robot, patterns) == "urdf_test"
        assert batch_ymd(f"{robot}_Plug_in_the_charging_cable_20260623") == "20260623"
        assert keep_batch(f"{robot}_Plug_in_the_charging_cable_20260601", MIN_BATCH_DATE)
        assert not keep_batch(f"{robot}_Plug_in_the_charging_cable_20260531", MIN_BATCH_DATE)
        assert not keep_batch(f"{robot}_test", MIN_BATCH_DATE)
        old_batch = f"{robot}_back-handle-installation_20260828"
        new_batch = f"{robot}_back-handle-installation_20260907"
        may_batch = f"{robot}_Plug_in_the_charging_cable_20260531"
        cosmos = root / "cosmos"
        _write_tiny_hdf5(cosmos / old_batch / "0828_100000" / "data" / "trajectory.hdf5")
        output = root / "nightly"

        listing = {f"{BOS_BASE}/raw_data/{robot}/": [old_batch, new_batch, may_batch]}
        for other in ROBOT_TYPES:
            listing.setdefault(f"{BOS_BASE}/raw_data/{other}/", [])

        def fake_list(_bcecmd: Path, uri: str) -> list[str]:
            return listing.get(uri, [])

        def fake_sync(_command: list[str], partial: Path) -> int:
            _write_tiny_hdf5(partial / "0907_220000" / "data" / "trajectory.hdf5")
            broken = partial / "0907_220001" / "data" / "trajectory.hdf5"
            broken.parent.mkdir(parents=True, exist_ok=True)
            broken.write_bytes(b"not a real hdf5")
            return 0

        kwargs = dict(
            mapping=mapping,
            output=output,
            bcecmd=Path("/bin/true"),
            extra_roots=[cosmos],
            result_logs=[],
            dry_run=True,
            resume=True,
            min_free_bytes=0,
            min_date=MIN_BATCH_DATE,
            list_dir=fake_list,
            sync=None,
        )
        assert run_nightly(**kwargs) == 0
        dry = json.loads((output / "_metadata" / "runs" / f"{shanghai_today()}.json").read_text())
        assert dry["new_batches"] == 1
        assert new_batch in dry["new_paths"][0]
        assert may_batch not in "".join(dry["new_paths"])

        kwargs["dry_run"] = False
        kwargs["sync"] = fake_sync
        assert run_nightly(**kwargs) == 0
        dest = destination_for(output, robot, "back_handle", new_batch)
        assert (dest / "0907_220000" / "data" / "trajectory.hdf5").is_file()
        assert (dest / "0907_220001" / "data" / "trajectory.hdf5").is_file()
        marker = json.loads((dest / ".download_complete.json").read_text())
        assert marker["hdf5_count"] == 1
        assert marker.get("skipped_bad_hdf5")
        assert dest.as_posix().endswith(f"{robot}/back_handle/{new_batch}")
        assert not destination_for(output, robot, "back_handle", old_batch).exists()
    print("nightly_download self-test: PASS")
    return 0


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    """给 ``run`` / ``daily`` 加上同一组参数。

    输入: 子命令解析器。
    输出: 无。
    """
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bcecmd", type=Path, default=BCECMD)
    parser.add_argument("--local-root", type=Path, action="append")
    parser.add_argument("--results", type=Path, action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--min-free-bytes", dest="min_free_bytes", type=int, default=MIN_FREE_BYTES)
    parser.add_argument(
        "--min-date",
        default=MIN_BATCH_DATE,
        help="只下批次名日期 >= 该日的数据，默认 20260601。空字符串表示不过滤",
    )


def nightly_kwargs(args: argparse.Namespace) -> dict:
    """从命令行参数拼出 ``run_nightly`` 的关键字参数。

    输入: ``run`` 或 ``daily`` 解析后的 args。
    输出: 传给 ``run_nightly`` 的字典。
    """
    return dict(
        mapping=args.mapping,
        output=args.output,
        bcecmd=args.bcecmd,
        extra_roots=list(ALREADY_ROOTS if args.local_root is None else args.local_root),
        result_logs=list(RESULT_LOGS if args.results is None else args.results),
        dry_run=args.dry_run,
        resume=args.resume,
        min_free_bytes=args.min_free_bytes,
        min_date=args.min_date,
        mode=getattr(args, "handler", "run"),
    )


def run_daily(args: argparse.Namespace) -> int:
    """每天到点扫一次 BOS 并下载缺口，然后继续等下一天。

    输入: ``daily`` 子命令参数，``--at`` 默认 22:00（上海时间）。
    输出: 正常不返回；被杀掉才退出。
    """
    state_dir = args.output / "_metadata"
    while True:
        target = resolve_wait(0, args.at)
        write_run_status(
            state_dir,
            pid=os.getpid(),
            mode="daily",
            phase="waiting",
            next_at=target.isoformat(),
            current=None,
        )
        print(f"next daily run at {target.isoformat()}", flush=True)
        wait_until(target)
        print(f"daily run start {utc_now()}", flush=True)
        code = run_nightly(**nightly_kwargs(args))
        print(f"daily run done exit={code} at {utc_now()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """组装命令行。

    输入: 无。
    输出: 含 ``run`` / ``daily`` / ``self-test`` 的解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="立刻扫七台机器并下载缺失批次")
    add_run_flags(run)
    run.add_argument("--after-minutes", type=int, default=0, help="等 N 分钟后再跑")
    run.add_argument("--wait-until", dest="wait_until", default=None, help="等到今天的 HH:MM 再跑")
    run.set_defaults(handler="run")
    daily = sub.add_parser("daily", help="常驻：每天 HH:MM 自动扫并下载")
    add_run_flags(daily)
    daily.add_argument("--at", default="22:00", help="每天触发时间，默认 22:00（上海时区）")
    daily.set_defaults(handler="daily")
    test = sub.add_parser("self-test")
    test.set_defaults(handler="self-test")
    status = sub.add_parser("status", help="看当前进程、待下清单和进度")
    status.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    status.set_defaults(handler="status")
    return parser


def main() -> None:
    """命令行入口。

    输入: ``sys.argv``。
    输出: 以子命令返回码退出。
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = build_parser().parse_args()
    if args.handler == "self-test":
        raise SystemExit(run_self_test())
    if args.handler == "status":
        raise SystemExit(print_run_status(args.output))
    if args.handler == "daily":
        raise SystemExit(run_daily(args))
    target = resolve_wait(args.after_minutes, args.wait_until)
    if target is not None:
        wait_until(target)
    raise SystemExit(run_nightly(**nightly_kwargs(args)))


if __name__ == "__main__":
    main()
