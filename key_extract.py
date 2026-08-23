#!/usr/bin/env python3
"""从 ~/.zshrc 提取 DASHSCOPE_API_KEY（start.sh 调用——避免 shell 引号地狱）。"""
import os
import re

p = os.path.expanduser("~/.zshrc")
try:
    for line in open(p, encoding="utf-8", errors="replace"):
        if line.startswith("export DASHSCOPE_API_KEY"):
            m = re.search(r'=\s*"?\'?([A-Za-z0-9._-]+)"?\'?', line.strip())
            if m:
                print(m.group(1))
            break
except FileNotFoundError:
    pass
