#!/usr/bin/env bash
# Debian/Ubuntu setup. Run from the repo root:  bash scripts/setup_linux.sh
set -euo pipefail

echo "== System packages (needs sudo) =="
sudo apt-get update
# python3-venv: PEP 668 forbids pip installs into system Python on Debian 12+, so we use a venv.
# libxcb-cursor0 + libxkbcommon-x11-0 + libegl1 + libgl1: required by Qt 6 (PySide6) on X11.
# v4l-utils: `v4l2-ctl --list-devices` for camera debugging.
sudo apt-get install -y python3-venv python3-pip git v4l-utils \
    libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1 libglib2.0-0

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .

echo
echo "== Environment report =="
python -m orbit doctor
echo
echo "== Fetching hand model =="
python -m orbit download-models || echo "Model download failed - see the manual URL printed above."
echo
echo "Done. Next: source .venv/bin/activate && python -m orbit"
