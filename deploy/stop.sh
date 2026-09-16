#!/bin/bash
# stop.sh — 停止本仓库的 vLLM 服务（start.sh / start_1m.sh 启动）
# 端口: 8000（可用 PORT 环境变量覆盖）
# 用法: bash stop.sh [--force]
#       --force  跳过优雅等待，立刻 SIGKILL
#       按启动脚本固定传入的 --served-model-name Deepseek 匹配进程，
#       与权重目录名无关；找不到时回退按端口定位。

set -u

PORT="${PORT:-8000}"
MODEL_PATTERN="served-model-name Deepseek"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

echo "查找本服务进程（端口 ${PORT}，匹配 \"${MODEL_PATTERN}\"）..."

PIDS=""
if command -v pgrep >/dev/null 2>&1; then
    PIDS=$(pgrep -f "vllm.*${MODEL_PATTERN}" 2>/dev/null | tr '\n' ' ' || true)
else
    PIDS=$(ps -eo pid,args --no-headers 2>/dev/null \
        | grep -E "vllm.*${MODEL_PATTERN}" | grep -v grep \
        | awk '{print $1}' | tr '\n' ' ' || true)
fi
PIDS=$(echo "$PIDS" | xargs 2>/dev/null)

if [ -z "$PIDS" ] && command -v ss >/dev/null 2>&1; then
    PIDS=$(ss -tlnp "sport = :${PORT}" 2>/dev/null \
        | grep -oE 'pid=[0-9]+' | cut -d= -f2 | tr '\n' ' ' || true)
elif [ -z "$PIDS" ] && command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti:${PORT} 2>/dev/null | tr '\n' ' ' || true)
fi
PIDS=$(echo "$PIDS" | xargs 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "✓ 没有运行中的服务进程"
    exit 0
fi

echo "  匹配到 PID: $PIDS"
echo "  发送 SIGTERM..."
for pid in $PIDS; do
    kill -TERM "$pid" 2>/dev/null || true
done

WAIT=20
[ "$FORCE" = "1" ] && WAIT=3
for i in $(seq 1 $WAIT); do
    sleep 1
    SURVIVORS=""
    for pid in $PIDS; do
        kill -0 "$pid" 2>/dev/null && SURVIVORS="$SURVIVORS $pid"
    done
    [ -z "$SURVIVORS" ] && break
done

SURVIVORS=""
for pid in $PIDS; do
    kill -0 "$pid" 2>/dev/null && SURVIVORS="$SURVIVORS $pid"
done
if [ -n "$SURVIVORS" ]; then
    echo "  仍在存活: $SURVIVORS，发送 SIGKILL"
    for pid in $SURVIVORS; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1
fi

REMAINING=""
for pid in $PIDS; do
    kill -0 "$pid" 2>/dev/null && REMAINING="$REMAINING $pid"
done
if [ -n "$REMAINING" ]; then
    echo "✗ 仍有进程无法停止: $REMAINING（可能属于其他用户或 PID 已重用）"
    exit 1
fi

if command -v ss >/dev/null 2>&1; then
    if ss -tln "sport = :${PORT}" 2>/dev/null | grep -q ":${PORT}\b"; then
        echo "⚠ 进程已退出但端口 ${PORT} 仍被占用（可能被其他服务占用）"
        exit 0
    fi
fi

echo "✓ 服务已停止（端口 ${PORT} 已释放）"
