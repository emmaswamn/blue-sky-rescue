#!/usr/bin/env python3
"""Step 6 B3 · RTSP 滤镜伪流：go2rtc test → YOLO+滤镜 → 推 test_blue

前置：docker compose up -d go2rtc（test 原流可播）

用法:
  python scripts/run_sky_filter_stream.py
  python scripts/run_sky_filter_stream.py --input rtsp://127.0.0.1:8554/test --mask-every 5

浏览器：http://localhost:1984/ → test_blue（WebRTC/MSE）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sky_filter_core import (
    MASK_BLUR_SIGMA,
    apply_unified_sky_filter,
    extend_sky_mask_to_top,
    soft_mask,
)

WEIGHTS = Path("weights/sky-seg.pt")
DEFAULT_IN = "rtsp://127.0.0.1:8554/test"
DEFAULT_OUT = "rtsp://127.0.0.1:8554/test_blue"
DEFAULT_OUT_FPS = 25.0
SKY_CLASS = 0


def quiet_ffmpeg_logs() -> None:
    """OpenCV 读 RTSP 时 libav 会把循环/keyframe 警告打到 stderr，demo 可静音。"""
    os.environ.setdefault("AV_LOG_LEVEL", "error")
    os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def yolo_sky_mask(model: YOLO, frame: np.ndarray, sky_class: int = SKY_CLASS) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for r in model(frame, verbose=False):
        if r.masks is None:
            continue
        for seg, cls in zip(r.masks.data, r.boxes.cls):
            if int(cls) != sky_class:
                continue
            m = seg.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8) * 255)
    return extend_sky_mask_to_top(mask)


def open_rtsp(url: str) -> cv2.VideoCapture:
    # err_detect=ignore_err：go2rtc mp4 循环边界偶发坏帧，解码器可跳过
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|err_detect;ignore_err"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def start_ffmpeg_publisher(width: int, height: int, fps: float, rtsp_out: str) -> subprocess.Popen:
    gop = max(1, int(round(fps)))
    cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-rtsp_transport",
        "tcp",
        "-f",
        "rtsp",
        rtsp_out,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step6 B3 RTSP 读 test → YOLO+滤镜 → 推 test_blue")
    parser.add_argument("--input", default=DEFAULT_IN, help="RTSP 输入（go2rtc test）")
    parser.add_argument("--output", default=DEFAULT_OUT, help="RTSP 推流地址（go2rtc ingest）")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--mask-every", type=int, default=5)
    parser.add_argument(
        "--out-fps",
        type=float,
        default=DEFAULT_OUT_FPS,
        help="推流/ pacing 帧率（默认 25；勿跟源 60fps 走）",
    )
    args = parser.parse_args()

    quiet_ffmpeg_logs()

    if not shutil.which("ffmpeg"):
        raise SystemExit("需要 ffmpeg（sudo apt install ffmpeg）")
    if not args.weights.exists():
        raise SystemExit(f"缺 {args.weights}")

    print(f"input  {args.input}")
    print(f"output {args.output}")
    print(f"mask-every={args.mask_every}  Ctrl+C 退出")

    model = YOLO(str(args.weights))
    last_mask: np.ndarray | None = None
    frame_idx = 0
    ff: subprocess.Popen | None = None
    w = h = 0
    fps = args.out_fps
    frame_interval = 1.0 / fps
    next_tick = time.perf_counter()
    shown = 0
    t0 = time.perf_counter()

    while True:
        cap = open_rtsp(args.input)
        if not cap.isOpened():
            print("RTSP 打不开，3s 后重试…", args.input)
            time.sleep(3)
            continue

        if w == 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            ff = start_ffmpeg_publisher(w, h, fps, args.output)
            print(f"ffmpeg → {args.output}  {w}x{h} @{fps:.1f}fps (src≈{src_fps:.1f})")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("RTSP 断流，重连…")
                cap.release()
                if ff and ff.poll() is None:
                    try:
                        ff.stdin.close()
                    except BrokenPipeError:
                        pass
                ff = None
                last_mask = None
                break

            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))

            if frame_idx % args.mask_every == 0 or last_mask is None:
                last_mask = yolo_sky_mask(model, frame)
            m = soft_mask(last_mask, MASK_BLUR_SIGMA)
            out = apply_unified_sky_filter(frame, m)

            assert ff is not None and ff.stdin is not None
            try:
                ff.stdin.write(out.tobytes())
            except BrokenPipeError:
                print("ffmpeg 管道断开，重启推流…")
                cap.release()
                ff = None
                last_mask = None
                break

            frame_idx += 1
            shown += 1
            if shown % 60 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {shown} frames  {shown / elapsed:.1f} fps effective")

            next_tick += frame_interval
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
