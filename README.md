# blue-sky-rescue

固定机位海岸视频 · **天空分割 + 蓝天滤镜** · 离线对比片 + go2rtc 浏览器伪流。

支持两种 mask 引擎：**YOLOv8n-seg**（默认交付）与 **HSV 五道门**（无 GPU 平替）。调色逻辑统一在 `sky_filter_core.py`。

---

## 演示效果

| 产物 | 说明 |
|------|------|
| `compare_10s.mp4` | 10 秒原片 vs 滤镜并排（Step 5） |
| go2rtc `test` / `test_blue` | 浏览器原流 vs YOLO 滤镜流 |
| go2rtc `test_blue_hsv` | HSV 滤镜流（可选对比） |

[对比图](./docs/images/compare.jpg) 

---

## 架构

```text
视频源（mp4 循环 / RTSP）
    ↓
天空 mask
    ├─ YOLOv8n-seg（weights/sky-seg.pt）  ← 默认 · 需 GPU
    └─ HSV A+B′（六点 JSON + sky_horizon） ← 可选 · 纯 CPU
    ↓
sky_filter_core · unified 调色（H−8 · S×1.25 · 软边）
    ↓
单帧 / mp4 / RTSP 伪流（go2rtc WebRTC）
```

**换真 RTSP 只改 `--input`**；换 mask 引擎只换脚本，滤镜参数不动。

---

## 快速开始

### 1. 环境

```bash
git clone <your-repo> blue-sky-rescue && cd blue-sky-rescue
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg   # 若无
```

准备素材（不进 git）：

```text
data/sky-filter/pexels-15982565.mp4   # 1280×720 海岸片
weights/sky-seg.pt                    # 自训权重，见 playbook Step 4
```

### 2. 单帧链路（Step 1～3）

```bash
# 抽帧
ffmpeg -y -ss 00:00:05 -i data/sky-filter/pexels-15982565.mp4 \
  -frames:v 1 -q:v 2 output/sky-filter/frame_raw.jpg

# HSV mask（需先配置 output/sky-filter/hsv_threshold_a.json）
python scripts/run_mask_hsv_a.py
python scripts/run_mask_hsv_b.py
python scripts/run_sky_filter_unified.py
```

### 3. YOLO 链路（Step 4～5）

```bash
bash scripts/extract_sky_frames.sh
python scripts/generate_pseudo_sky_labels.py
# 训练 → cp best.pt weights/sky-seg.pt（见 playbook/04）

python scripts/run_yolo_sky_mask.py
python scripts/run_sky_filter_yolo.py
python scripts/run_sky_filter_video.py --seconds 10 --mask-every 5 --side-by-side
```

### 4. 浏览器伪流（Step 6）

```bash
docker compose up -d go2rtc
# 浏览器 http://localhost:1984/ → test（原片）

# 终端 2 · YOLO 滤镜流
python scripts/run_sky_filter_stream.py --mask-every 5

# 终端 3 · HSV 平替（可选）
docker compose restart go2rtc   # 首次加 test_blue_hsv 后必做
python scripts/run_sky_filter_stream_hsv.py --mask-every 5
```

---

## 脚本索引

| 脚本 | 用途 |
|------|------|
| `run_mask_hsv_a.py` / `run_mask_hsv_b.py` | Step 2 HSV mask |
| `sky_horizon.py` | 海天裁切 · 动态 ROI |
| `hsv_sky_mask.py` | HSV mask 函数（伪标注 + 伪流） |
| `sky_filter_core.py` | **统一滤镜**（全链路共用） |
| `generate_pseudo_sky_labels.py` | HSV → YOLO 伪标注 |
| `run_sky_filter_video.py` | 10s 对比 mp4 |
| `run_sky_filter_preview.py` | 窗口伪流 |
| `run_sky_filter_stream.py` | RTSP → `test_blue`（YOLO） |
| `run_sky_filter_stream_hsv.py` | RTSP → `test_blue_hsv`（HSV） |
| `compare_roi_horizon.py` | ROI bottom_up vs 0.65 对比图 |

完整分步手册 → **[playbook/README.md](./playbook/README.md)**  
命令速查 → **[playbook/commands.sh](./playbook/commands.sh)**

---

## HSV vs YOLO（本片 1280×720）

| | HSV | YOLO |
|--|-----|------|
| 训练 | 无 | 32 帧伪标注训 1 次 |
| 硬件 | CPU | GPU 建议 |
| 伪流 fps（mask-every=5） | ~14–15 | ~22 |
| 本片效果 | 肉眼≈YOLO | 交付默认 |
| 适用 | 固定机位 · 已调六点 · 边缘盒子 | 泛化 · 有 GPU |

---

## 目录结构

```text
blue-sky-rescue/
  scripts/           # 全部可执行脚本
  config/go2rtc.yaml # test / test_blue / test_blue_hsv
  docker-compose.yml # go2rtc（+ processor profile full 迭代中）
  playbook/          # 操作手册（GitHub 公开）
  data/              # 素材（gitignore）
  output/            # 产物（gitignore）
  weights/           # sky-seg.pt（gitignore）
```

---

## Docker

```bash
docker compose up -d go2rtc          # Web UI :1984 · RTSP :8554
docker compose restart go2rtc      # 改 go2rtc.yaml 后
docker compose stop go2rtc
```

`config/go2rtc.yaml` 流表：

```yaml
streams:
  test:        # mp4 循环原片
  test_blue:   # YOLO 滤镜 ingest
  test_blue_hsv: []  # HSV 滤镜 ingest
```

---

## 依赖

- Python 3.11+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（seg）
- OpenCV · NumPy
- ffmpeg
- Docker（伪流可选）
- NVIDIA GPU（YOLO 训练 / 实时推荐）

---

## 已知现象（demo 可忽略）

- 读 `test`（mp4 循环）时终端偶发 `h264 error while decoding MB` / `RTP bad cseq` — 循环接缝，YOLO 与 HSV 均有
- 改 `go2rtc.yaml` 新增流名后须 **restart**，否则 RTSP publish 失败

---

## License / 素材

- 演示 mp4 来源：Pexels https://www.pexels.com/video/scenic-ocean-view-with-rugged-coastline-37693951/ 
- `weights/sky-seg.pt` 在Release


