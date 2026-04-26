#!/usr/bin/env bash
# Dump the running container's logs + kernel ring buffer to logs/ before any
# docker compose down/restart. Lets you preserve application-side state when
# the engine gets stuck (see "Known issues" in README  -  DFlash spec-decode
# can leave the EngineCore worker spinning after a mid-decode client kill).
#
# Usage: ./scripts/dump_logs.sh [tag]
# Output: logs/engine_<tag>_<timestamp>.log + logs/dmesg_<tag>_<timestamp>.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-snapshot}"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$REPO_ROOT/logs"

ENGINE_LOG="$REPO_ROOT/logs/engine_${TAG}_${TS}.log"
DMESG_LOG="$REPO_ROOT/logs/dmesg_${TAG}_${TS}.log"

echo "Dumping engine logs to $ENGINE_LOG"
docker logs vllm-awq4-qwen > "$ENGINE_LOG" 2>&1 || echo "(no container)"

echo "Dumping kernel amdgpu/kfd messages to $DMESG_LOG"
journalctl -k --since "30 min ago" --grep "amdgpu|kfd|hsa" > "$DMESG_LOG" 2>&1 || true

ls -lh "$ENGINE_LOG" "$DMESG_LOG"
echo "Done. Now safe to docker compose down/restart without losing diagnostics."
