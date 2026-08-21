"""共享测试工具：状态日志（可观测性 §10 简化）。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateLog:
    """状态转移日志（供测试断言）。"""
    entries: list = field(default_factory=list)

    def __call__(self, state, sub, old):
        self.entries.append((old, (state, sub)))
