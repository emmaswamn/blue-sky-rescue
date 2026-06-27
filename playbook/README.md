# blue-sky-rescue · 操作手册

> **素材**：`pexels-15982565.mp4`（1280×720）
> **状态**：Step 1～6 + HSV 平替 **已验收** ✅


---

## 路线一览

```text
mp4 → 单帧 → HSV A/B′ mask → unified 滤镜
     → 抽帧 → HSV 伪标注 → YOLOv8n-seg
     → YOLO mask → unified → compare_10s.mp4
     → go2rtc：test / test_blue（YOLO）/ test_blue_hsv（HSV 可选）
```

---

## 分步（按序）

### 0 · 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg
```

### 1 · 单帧

```bash
ffmpeg -y -ss 00:00:05 -i data/sky-filter/pexels-15982565.mp4 \
  -frames:v 1 -q:v 2 output/sky-filter/frame_raw.jpg
```

### 2 · HSV mask

```bash
python scripts/run_mask_hsv_a.py          # 默认 ROI bottom_up
python scripts/run_mask_hsv_b.py
# 验收 output/sky-filter/mask_hsv.png · mask_hsv_overlay.jpg
```

六点 JSON：`output/sky-filter/hsv_threshold_a.json`（取色后生成）

### 3 · unified 滤镜

```bash
python scripts/run_sky_filter_unified.py
# → output/sky-filter/frame_after_unified.jpg
```

### 4 · 伪标注 + 训练 YOLO

```bash
bash scripts/extract_sky_frames.sh
python scripts/generate_pseudo_sky_labels.py --limit 5 --save-steps  # 先试 5 帧
python scripts/generate_pseudo_sky_labels.py

yolo segment train \
  data=data/sky-seg/sky.yaml model=yolov8n-seg.pt \
  epochs=80 imgsz=640 batch=8 patience=15 amp=False \
  project="$(pwd)/runs/sky-seg" name=train1

cp runs/sky-seg/train1/weights/best.pt weights/sky-seg.pt
```

### 5 · YOLO 推理 + 对比片

```bash
python scripts/run_yolo_sky_mask.py
python scripts/run_sky_filter_yolo.py
python scripts/run_sky_filter_video.py --seconds 10 --mask-every 5 --side-by-side
```

### 6 · 浏览器伪流

```bash
# B1 窗口（可选）
pip install opencv-python   # 需要 GUI
python scripts/run_sky_filter_preview.py --side-by-side

# B2 原流
docker compose up -d go2rtc
# http://localhost:1984/ → test

# B3 YOLO 滤镜
python scripts/run_sky_filter_stream.py --mask-every 5

# B3b HSV 平替
docker compose restart go2rtc   # yaml 含 test_blue_hsv 后首次必做
python scripts/run_sky_filter_stream_hsv.py --mask-every 5
```

---

## HSV vs YOLO 平替

| | HSV `test_blue_hsv` | YOLO `test_blue` |
|--|---------------------|------------------|
| 脚本 | `run_sky_filter_stream_hsv.py` | `run_sky_filter_stream.py` |
| 硬件 | CPU | GPU |
| 本片 fps | ~14–15 | ~22 |
| 效果 | 肉眼≈YOLO | 默认交付 |

公平对比：两边都用 `--mask-every 5`。

---

## 关键产物

```text
output/sky-filter/
  frame_raw.jpg · mask_hsv.png · frame_after_unified.jpg
  mask_yolo.png · frame_after_yolo.jpg
  before_10s.mp4 · after_10s.mp4 · compare_10s.mp4
weights/sky-seg.pt
config/go2rtc.yaml
```

---

## 命令速查

见 [commands.sh](./commands.sh)（分段注释，复制即用）
`
