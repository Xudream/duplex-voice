# duplex-voice · 级联全双工语音交互系统（第一版实现 v0.1）

> 设计依据：`../级联全双工语音交互系统-软件设计.md`（v2.1）
> 技术栈：Silero VAD v6 + Qwen3-ASR/LLM/TTS（流式级联），两档能力（rule 半双工缺省 / semantic 全双工 P1）

## 结构

```
duplex_voice/
├── main.py            # 装配 + 入口（--text 文本模式 / 默认麦克风）
├── events.py          # 事件信封（seq/ts/trace_id）+ 高优先级通道
├── config.py          # 配置（config.yaml 加载）
├── audio/             # 采集(T1) / 播放(T3) / pre-roll 环形缓冲(采集侧)
├── vad/               # VadJudge 接缝：RuleVadJudge(Silero双实例+双钳制) / Semantic(占位)
├── adapter/           # 模型适配器：ASR/LLM/TTS Provider + 注册表（mock/qwen3）
├── fsm/               # DuplexFSM：四态+子状态 / 转移表 / 定时器 / 打断裁决
└── session.py         # 会话管理 + JSONL 落盘
tests/                 # 17 个测试（pre-roll/双钳制/FSM/冒烟）
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 文本驱动模式（无麦克风，CI/验证）
python3 -m duplex_voice.main --text

# 麦克风模式（默认 mock 模型，跑通闭环）
python3 -m duplex_voice.main

# 接入真实模型（改 config.yaml）
#   model.llm_provider: openai  + llm_api_key（LLM 即可真）
#   model.asr_provider / tts_provider: qwen3（需部署对应服务，协议见设计 §6）
```

## 测试

```bash
python3 -m pytest tests/ -v
```

## 配置速查（config.yaml）

| 键 | 默认 | 说明 |
|----|------|------|
| vad.judge | rule | rule=半双工（缺省）/ semantic=全双工（P1 插上） |
| vad.min_silence_ms | 500 | 判停（250-1250 调档） |
| vad.pre_roll_ms | 1200 | 打断回滚（采集侧 AEC 后） |
| model.asr_provider | mock | mock / qwen3 |
| model.llm_provider | mock | mock / openai / qwen3 |
| model.tts_provider | mock | mock / qwen3 |

## 状态机（软件设计 §2.4）

四态 LISTEN/THINK/SPEAK/YIELD + 子状态（voicing/finalizing/post_wait…）；
声学打断（takeover_noise）= rule 档基础能力：双钳制 + 窗口 80% + pre-roll 1.2s 回滚。

## 已知边界（P0 第一版）

- AEC 为占位（aec_enabled=false，真 AEC P1）；打断检测依赖双钳制阈值
- Silero 在 Python 3.14 下 torch.jit 不可用 → 自动回退能量+过零率判定（生产建议 Python 3.11/3.12 或 ONNX）
- SemanticVadJudge 为接口占位（8 状态 token 模型训练后接入）
- Qwen3-ASR/TTS 协议以官方发布为准（§12 风险 1）
