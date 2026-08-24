#!/usr/bin/env python3
"""duplex-voice Web 版跨平台启动器（Windows / Linux / macOS 通用）。

用法：
    python start.py                 # 默认语义 VAD = omni
    python start.py --vad rule      # 规则模式（前端 🧠 按钮也可运行中切换）

流程：检查 Python → 检查/自动安装依赖 → 检查 API key → 启动 server。
"""
import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DEPS = ["fastapi", "uvicorn", "websockets", "httpx", "httpx_sse", "numpy"]


def _importable(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def check_deps() -> None:
    print("==> [2/4] 依赖检查")
    missing = [m for m in DEPS if not _importable(m)]
    if missing:
        print(f"缺少依赖: {', '.join(missing)} → 安装中（pip install -r requirements.txt）…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    print("依赖 OK")


def find_key() -> str | None:
    """API key：读 config.yaml（model.dashscope_api_key）——不读环境变量（2026-08-24 用户要求）。
    不存在时从 ~/.zshrc 等提取并写入 config.yaml（迁移友好）。"""
    cfg_path = ROOT / "config.yaml"
    try:
        import json as _json
        k = _json.loads(cfg_path.read_text(encoding="utf-8")).get("model", {}).get("dashscope_api_key", "")
        if k.startswith("sk-"):
            return k
    except Exception:
        pass
    for rc in ("~/.zshrc", "~/.bashrc", "~/.bash_profile"):
        p = Path(os.path.expanduser(rc))
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("export DASHSCOPE_API_KEY"):
                m = re.search(r'=\s*"?\'?([A-Za-z0-9._-]+)"?\'?', line.strip())
                if m and m.group(1).startswith("sk-"):
                    import json as _json
                    try:
                        cfg_path.write_text(_json.dumps({"model": {"dashscope_api_key": m.group(1)}},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"  已从 {rc} 提取 key 写入 config.yaml（{m.group(1)[:4]}…）")
                    except Exception:
                        pass
                    return m.group(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="duplex-voice Web 版启动器（跨平台）")
    parser.add_argument("--vad", choices=["omni", "rule"], default=os.environ.get("SEMANTIC_VAD", "omni"),
                        help="语义 VAD 初始模式：omni 模型 / rule 规则（默认 omni）")
    args = parser.parse_args()

    print("==> [1/4] Python 检查")
    if sys.version_info < (3, 10):
        print(f"❌ 需要 Python 3.10+（当前 {sys.version_info.major}.{sys.version_info.minor}）")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    check_deps()

    print("==> [3/4] API key 检查")
    key = find_key()
    if not key:
        print("❌ 未找到 DASHSCOPE_API_KEY（sk- 开头）")
        print("   请先设置环境变量（或写入 ~/.zshrc / ~/.bashrc）：")
        print("     Windows:  set DASHSCOPE_API_KEY=你的key")
        print("     macOS/Linux:  export DASHSCOPE_API_KEY=你的key")
        sys.exit(1)
    print(f"API key 已就绪（{key[:4]}…）")

    print("==> [4/4] 启动 server")
    env = dict(os.environ)
    # 不设 DASHSCOPE_API_KEY——server 读 config.yaml（2026-08-24 用户要求）
    env["SEMANTIC_VAD"] = args.vad
    print(f"启动：SEMANTIC_VAD={args.vad} → http://127.0.0.1:8787 （Ctrl+C 停止）")
    try:
        subprocess.run([sys.executable, str(WEB / "server.py")], cwd=str(WEB), env=env)
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
