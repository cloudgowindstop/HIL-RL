#!/usr/bin/env bash
# 用法: ./start_nightly_download.sh [daily|run|status|stop]
# daily（默认）每天 22:00；run 立刻下一轮。改点: NIGHTLY_AT=21:30 $0 daily
set -euo pipefail

PY=/media/jushen/mingbo-ge/.venv/bin/python
SCRIPT="$(cd "$(dirname "$0")" && pwd)/nightly_download.py"
META=/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/_metadata
LOG=$META/cron.log
STATUS=$META/current_run.json
AT="${NIGHTLY_AT:-22:00}"
cmd="${1:-daily}"
shift || true

pids() { pgrep -f "$SCRIPT" || true; }

show_status() { "$PY" "$SCRIPT" status; }

wait_for_plan() {
  local pid="$1"
  echo "正在列 BOS / 扫本地（通常一两分钟）..."
  local i=0
  while (( i < 90 )); do
    if [[ -f "$STATUS" ]]; then
      phase=$("$PY" -c "import json; print(json.load(open('$STATUS')).get('phase',''))")
      case "$phase" in
        planned|downloading|done|skipped_disk|locked) return 0 ;;
      esac
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 2
    i=$((i + 1))
  done
  echo "等扫描超时，用 $0 status 查看" >&2
  return 1
}

case "$cmd" in
  status) show_status ;;
  stop)   pids | grep -q . && pkill -f "$SCRIPT" && echo 已停止 || echo 未在运行 ;;
  daily|run)
    if pids | grep -q .; then
      echo "已经在跑: $(pids)" >&2
      show_status
      exit 1
    fi
    mkdir -p "$META"
    extra=(); [[ "$cmd" == daily ]] && extra=(--at "$AT")
    nohup env PYTHONUNBUFFERED=1 "$PY" "$SCRIPT" "$cmd" "${extra[@]}" "$@" >>"$LOG" 2>&1 &
    pid=$!
    echo "已启动  pid=$pid  mode=$cmd"
    echo "日志    $LOG"
    if [[ "$cmd" == run ]]; then
      wait_for_plan "$pid" || true
      echo
      show_status || true
      if kill -0 "$pid" 2>/dev/null; then
        echo
        echo "下载在后台继续。再看进度: $0 status"
        echo "跟日志: tail -f $LOG"
      fi
    else
      echo "每天 ${AT} 自动扫。看状态: $0 status"
    fi
    ;;
  *) echo "用法: $0 [daily|run|status|stop]" >&2; exit 2 ;;
esac
