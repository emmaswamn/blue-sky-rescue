#!/usr/bin/env bash
# blue-sky-rescue · 命令速查（分段复制，勿一键全跑）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

# ========== 0 环境 ==========
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# sudo apt install ffmpeg

# ========== 1 单帧 ==========
# ffmpeg -y -ss 00:00:05 -i data/sky-filter/pexels-15982565.mp4 \
#   -frames:v 1 -q:v 2 output/sky-filter/frame_raw.jpg

# ========== 2 HSV mask ==========
# python scripts/run_mask_hsv_a.py
# python scripts/run_mask_hsv_b.py

# ========== 3 unified ==========
# python scripts/run_sky_filter_unified.py

# ========== 4 伪标注 + 训练 ==========
# bash scripts/extract_sky_frames.sh
# python scripts/generate_pseudo_sky_labels.py
# yolo segment train data=data/sky-seg/sky.yaml model=yolov8n-seg.pt \
#   epochs=80 imgsz=640 batch=8 patience=15 amp=False \
#   project="$ROOT/runs/sky-seg" name=train1
# cp runs/sky-seg/train1/weights/best.pt weights/sky-seg.pt

# ========== 5 YOLO + 对比片 ==========
# python scripts/run_yolo_sky_mask.py
# python scripts/run_sky_filter_yolo.py
# python scripts/run_sky_filter_video.py --seconds 10 --mask-every 5 --side-by-side

# ========== 6 伪流 ==========
# docker compose up -d go2rtc
# docker compose restart go2rtc
# python scripts/run_sky_filter_stream.py --mask-every 5
# python scripts/run_sky_filter_stream_hsv.py --mask-every 5

# ========== ROI 对比（可选） ==========
# python scripts/compare_roi_horizon.py

echo "见 playbook/README.md — 请分段取消注释后运行"
