#!/usr/bin/env bash
# duplex-voice 启动（macOS / Linux）——跨平台逻辑见 start.py
# 用法：./start.sh [--vad omni|rule]
cd "$(dirname "$0")"
exec python3 start.py "$@"
