#!/usr/bin/env python3
"""Step 6 · RTSP 滤镜伪流（HSV 版）：go2rtc test → HSV mask + 滤镜 → 推 test_blue_hsv

与 run_sky_filter_stream.py（YOLO → test_blue）并排对比，验硬件有限时 HSV 平替效果。

前置：docker compose up -d go2rtc（test 原流可播；yaml 含 test_blue_hsv ingest）

用法:
  python scripts/run_sky_filter_stream_hsv.py
  python scripts/run_sky_filter_stream_hsv.py --input rtsp://127.0.0.1:8554/test

浏览器：http://localhost:1984/ → test（原片）· test_blue（YOLO）· test_blue_hsv（本脚本）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2

from hsv_sky_mask import DEFAULT_EXCLUDE, DEFAULT_THRESH, compute_hsv_sky_mask
from sky_filter_core import MASK_BLUR_SIGMA, apply_unified_sky_filter, soft_mask

DEFAULT_IN = "rtsp://127.0.0.1:8554/test"
DEFAULT_OUT = "rtsp://127.0.0.1:8554/test_blue_hsv"
DEFAULT_OUT_FPS = 25.0


def quiet_ffmpeg_logs() -> None:
    os.environ.setdefault("AV_LOG_LEVEL", "error")
    os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def open_rtsp(url: str) -> cv2.VideoCapture:
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


def stop_ffmpeg(ff: subprocess.Popen | None) -> None:
    if ff is None or ff.poll() is not None:
        return
    try:
        if ff.stdin:
            ff.stdin.close()
    except BrokenPipeError:
        pass
    try:
        ff.wait(timeout=3)
    except subprocess.TimeoutExpired:
        ff.kill()


def ensure_ffmpeg(
    ff: subprocess.Popen | None,
    w: int,
    h: int,
    fps: float,
    rtsp_out: str,
) -> subprocess.Popen:
    if ff is not None and ff.poll() is None:
        return ff
    stop_ffmpeg(ff)
    ff = start_ffmpeg_publisher(w, h, fps, rtsp_out)
    print(f"ffmpeg → {rtsp_out}  {w}x{h} @{fps:.1f}fps")
    return ff


def main() -> None:
    parser = argparse.ArgumentParser(description="Step6 RTSP 读 test → HSV mask + unified → 推 test_blue_hsv")
    parser.add_argument("--input", default=DEFAULT_IN, help="RTSP 输入（go2rtc test）")
    parser.add_argument("--output", default=DEFAULT_OUT, help="RTSP 推流地址（go2rtc ingest）")
    parser.add_argument("--thresh", type=Path, default=DEFAULT_THRESH)
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument(
        "--roi-mode",
        choices=("bottom_up", "fixed", "full"),
        default="bottom_up",
        help="与 run_mask_hsv_a.py 一致",
    )
    parser.add_argument(
        "--mask-every",
        type=int,
        default=1,
        help="HSV 便宜，默认每帧算；省 CPU 可加大",
    )
    parser.add_argument("--out-fps", type=float, default=DEFAULT_OUT_FPS)
    args = parser.parse_args()

    quiet_ffmpeg_logs()

    if not shutil.which("ffmpeg"):
        raise SystemExit("需要 ffmpeg（sudo apt install ffmpeg）")
    if not args.thresh.exists():
        raise SystemExit(f"缺 {args.thresh} — 先跑 Step 2 取色 / run_mask_hsv_a.py")

    print(f"input   {args.input}")
    print(f"output  {args.output}")
    print(f"thresh  {args.thresh}  roi-mode={args.roi_mode}  mask-every={args.mask_every}")
    print("HSV mask（无 GPU）  Ctrl+C 退出")

    last_mask = None
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
            print(f"source {w}x{h} (src≈{src_fps:.1f}fps)")

        try:
            ff = ensure_ffmpeg(ff, w, h, fps, args.output)
        except Exception as e:
            print(f"ffmpeg 起不来: {e} — 确认 go2rtc 已 restart 且 yaml 含 test_blue_hsv")
            cap.release()
            time.sleep(3)
            ff = None
            continue

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("RTSP 断流，重连…")
                cap.release()
                stop_ffmpeg(ff)
                ff = None
                last_mask = None
                break

            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))

            if frame_idx % args.mask_every == 0 or last_mask is None:
                last_mask = compute_hsv_sky_mask(
                    frame,
                    thresh_path=args.thresh,
                    exclude_path=args.exclude,
                    roi_mode=args.roi_mode,
                )
            m = soft_mask(last_mask, MASK_BLUR_SIGMA)
            out = apply_unified_sky_filter(frame, m)

            try:
                assert ff.stdin is not None
                ff.stdin.write(out.tobytes())
            except (BrokenPipeError, AssertionError):
                print("ffmpeg 管道断开，重启推流…")
                cap.release()
                stop_ffmpeg(ff)
                ff = None
                last_mask = None
                time.sleep(1)
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
