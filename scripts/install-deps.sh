#!/usr/bin/env bash
# 明早一键装依赖：Python venv + pip + docker pull go2rtc
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> [1/3] Python venv (.venv)"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [2/3] pip install -r requirements.txt"
pip install -U pip
pip install -r requirements.txt

echo "==> [3/3] docker pull go2rtc (alexxit/go2rtc:1.9.14)"
if command -v docker >/dev/null 2>&1; then
  docker compose pull go2rtc
else
  echo "WARN: docker 未找到，跳过 pull。装好 Docker 后执行："
  echo "  docker compose pull go2rtc"
fi

echo ""
echo "完成。激活环境："
echo "  source .venv/bin/activate"
echo "起 go2rtc："
echo "  docker compose up -d go2rtc"
echo "浏览器预览："
echo "  http://localhost:1984/"
