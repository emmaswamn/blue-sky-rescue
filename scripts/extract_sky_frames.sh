#!/usr/bin/env bash
# Step 4 · 从本视频抽帧到 data/sky-seg/images/train
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO="${ROOT}/data/sky-filter/pexels-15982565.mp4"
OUT="${ROOT}/data/sky-seg/images/train"
FPS="${1:-1/2}"  # 每 2 秒 1 帧；可传 1/3 等

if [[ ! -f "$VIDEO" ]]; then
  echo "缺视频: $VIDEO" >&2
  exit 1
fi
mkdir -p "$OUT"
ffmpeg -y -i "$VIDEO" -vf "fps=${FPS}" -q:v 2 "${OUT}/frame_%04d.jpg"
echo "saved $(ls -1 "$OUT"/*.jpg 2>/dev/null | wc -l) frames → $OUT"
echo "→ 删糊/重复帧，留 35～40 张，再标注 sky 多边形"
