"""设备鉴权（App 客户端 + 服务端模式）。

设计：
- 首次连接 POST /api/device/register（局域网内）→ 颁发 device_token（保存到 devices.json）
- 之后所有请求（含 WS /api/stream）带 Authorization: Bearer <device_token>
- 局域网部署可设 auth.required=false（config.server.auth.required）跳过校验
"""
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

DEVICES_FILE = Path(__file__).resolve().parent / "devices.json"
_lock = threading.Lock()


def _load() -> dict:
    if DEVICES_FILE.exists():
        try:
            return json.loads(DEVICES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(d: dict) -> None:
    DEVICES_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def register(name: str) -> dict:
    """注册设备 → {token, device_id}。token 只在注册时返回一次。"""
    token = "dvt_" + secrets.token_urlsafe(24)
    with _lock:
        db = _load()
        device_id = "dev_" + hashlib.sha256(token.encode()).hexdigest()[:12]
        db[device_id] = {"name": name, "registered_at": time.time(), "token_hash": _hash(token)}
        _save(db)
    return {"token": token, "device_id": device_id}


def verify(token: str) -> str | None:
    """校验 token → device_id（有效）或 None（无效）。"""
    if not token:
        return None
    h = _hash(token)
    with _lock:
        db = _load()
        for did, info in db.items():
            if info.get("token_hash") == h:
                return did
    return None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def list_devices() -> list[dict]:
    with _lock:
        db = _load()
    return [{"device_id": k, "name": v.get("name"), "registered_at": v.get("registered_at")}
            for k, v in db.items()]
