"""VAD 判定层（可插拔能力接缝，软件设计 §2.2）。

VadJudge Protocol —— 唯一替换点：
  RuleVadJudge      vad.judge="rule"      缺省：半双工（Silero 双实例 + 双钳制）
  SemanticVadJudge  vad.judge="semantic"  插上：全双工（8 状态 token，P1）
"""
from .judge import VadDecision, VadJudge
from .rule_judge import RuleVadJudge
from .semantic_judge import SemanticVadJudge

__all__ = ["VadDecision", "VadJudge", "RuleVadJudge", "SemanticVadJudge"]
