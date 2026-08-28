"""音频模块：采集（T1）、播放（T3）、pre-roll 环形缓冲（采集侧 AEC 后）。

懒加载：sounddevice/PortAudio 仅在本机设备模式实际使用时才 import，
服务器/Web 模式（无音频设备）不会触发，避免无 PortAudio 环境崩溃。
"""
from .pre_roll import PreRollBuffer

__all__ = ["AudioCapture", "AudioPlayer", "PreRollBuffer"]


def __getattr__(name):
    if name == "AudioCapture":
        from .capture import AudioCapture
        return AudioCapture
    if name == "AudioPlayer":
        from .player import AudioPlayer
        return AudioPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
