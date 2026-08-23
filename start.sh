#!/usr/bin/env bash
# ============================================================
# duplex-voice Web 版启动脚本
# 流程：检查 Python → 检查/安装依赖 → 检查 API key → 启动
# 用法：./start.sh          （语义 VAD 默认 omni）
#       SEMANTIC_VAD=rule ./start.sh   （切规则模式）
# ============================================================
set -e
cd "$(dirname "$0")"          # 脚本目录（任何位置执行都安全）

echo "==> [1/4] Python 环境"
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "❌ 未找到 python3（需 Python 3.10+）"
  exit 1
fi
$PY --version

echo "==> [2/4] 依赖检查"
MISSING=""
for mod in fastapi uvicorn websockets httpx httpx_sse numpy; do
  if ! $PY -c "import $mod" >/dev/null 2>&1; then
    MISSING="$MISSING $mod"
  fi
done
if [ -n "$MISSING" ]; then
  echo "缺少依赖:$MISSING → 安装中（pip install -r requirements.txt）…"
  $PY -m pip install -r requirements.txt
fi
echo "依赖 OK"

echo "==> [3/4] API key 检查"
KEY="${DASHSCOPE_API_KEY:-}"
case "$KEY" in sk-*) ;; *) KEY="" ;; esac   # 格式校验：非 sk- 开头（坏值/残留）忽略，走 zshrc 提取
if [ -z "$KEY" ]; then
  # 尝试从 ~/.zshrc 提取（key_extract.py——避免 shell 引号地狱）
  if [ -f "$HOME/.zshrc" ]; then
    KEY=$($PY key_extract.py)
  fi
fi
if [ -z "$KEY" ]; then
  echo "❌ 未找到 DASHSCOPE_API_KEY"
  echo "   请先：export DASHSCOPE_API_KEY=你的key   （或写入 ~/.zshrc）"
  exit 1
fi
echo "API key 已就绪（${KEY:0:4}…）"

echo "==> [4/4] 启动 server"
cd web
export DASHSCOPE_API_KEY="$KEY"
export SEMANTIC_VAD="${SEMANTIC_VAD:-omni}"    # 语义 VAD 初始模式：omni 模型 / rule 规则
echo "启动：SEMANTIC_VAD=$SEMANTIC_VAD  →  http://127.0.0.1:8787"
exec $PY server.py
