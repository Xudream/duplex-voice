# sidetalk（旁对话）与拒识洞察 2026

> 版本 v1.0　·　2026-10-27　·　范围：全双工语音交互中"旁对话"识别与拒识决策的业界进展
> 口径约定：【论文】取自公开论文摘要/正文；【官方】取自官方页；【估算】为工程推算。
> 注意：各基准的指标口径不同（F1 / 静默率 / 拒识率 / 准确率），**不可横向比较**，只能各自看趋势。

---

## 1. 结论与成熟度

### 1.1 一句话结论

2026 年这个问题的性质变了：不再是"能不能检测旁对话"，而是**"同一句话换个说话人，模型会不会闭嘴"**。
现有证据给出的答案是——**端到端音频大模型基本不会闭嘴**（VGBench 上 6 个原生模型 + 3 种免训练适配，最高静默率仅 14%），
必须靠独立门控或专门后训练才能到 91%。

产品侧也不是没做，而是**已经形成五层格局**（见 §4，收录 33 个模型与产品）：模型层 12 个（GPT-Live、Gemini Live、Qwen-Audio-3.1-Realtime、Gander、Seeduplex、Step-Audio 2、GLM-Realtime 等）、
基础设施层 7 个（LiveKit 轮次检测、Pipecat Smart Turn v3、ElevenLabs、Deepgram、AssemblyAI、OpenAI `semantic_vad`、Silero）、
产品层 9 个（ChatGPT Voice、Gemini for Home、Alexa、Siri/Apple UAD、AirPods、Ray-Ban Meta、AI Pin、车机音区、Teams）、
国内落地 5 个（车机多音区、免唤醒+可见即可说、车载语音 A 级认证、阿里云智能外呼「优雅打断」、火山豆包实时语音）。
**只有阿里公开了 sidetalk 量化（背景语音响应 73%→13%），没有一家公开误拒率**——这是我们对标时最缺的口径。

### 1.2 成熟度（四个维度分开看）

| 维度 | 阶段 | 代表证据 |
|---|---|---|
| 问题定义 | 已收敛 | 从"单句分类"转为"基于交互历史的序列路由 / action-level addressedness"【论文】 |
| 评测基准 | 体系化 | 2026 年四套并行：HumDial、CoDeTT、WearVox、VGBench【论文】 |
| 数据 | 半成熟 | 真实录制已出现（HumDial 演员录制、SLT 106h 四通道），中文真实 sidetalk 仍缺 |
| 方法 | 有解但不稳 | 纯音频 F1 0.86 / 音视频 0.95；端到端模型几乎不主动静默 |
| 产品 | 早期但已成格局 | 五层收录 33 个（§4）：模型层 12、基础设施层 7、产品层 9、国内落地 5；仅阿里公开 sidetalk 量化，无一家公开误拒率 |
| 长时行为 | 空白 | sidetalk 的持续性（连续多轮维持静默）几乎无基准覆盖 |

### 1.3 与既有文档的关系

| 文档 | 原有内容 | 本文补充 |
|---|---|---|
| `语义VAD方案.md` | `user_reject` = 听不清/不理解 → 澄清 | **sidetalk 与之动作相反**：应静默、不澄清（见 §2.3） |
| `全双工状态机设计.md` | reject 的行为表 | 需拆出 sidetalk / self-talk 两个分支（见 §7） |
| `四层架构方案.md` §2 | L1 八个状态 | 状态表需扩到 10 态（见 §7.1） |

---

## 2. 术语与边界

### 2.1 拒识的四个子场景（HumDial 口径）【论文】

ICASSP 2026 HumDial Challenge 把 Track II 的拒识拆成四类：

| 缩写 | 全称 | 中文 | 系统应有动作 |
|---|---|---|---|
| URB | user real-time backchannels | 用户实时附和 | 静默继续（不打断） |
| PH | pause handling | 停顿处理 | 等待，不抢话 |
| TPS | third-party speech | 第三方在说话 | 静默 |
| **SDO** | **speech directed at others** | **对他人说的话（sidetalk）** | **静默 + 记上下文** |

**sidetalk ≈ SDO**（主用户转向旁人），TPS 是它的相邻场景（背景里别人在聊，主用户没参与）。

### 2.2 三类"不该回应"的语音（VGBench 口径）【论文】

| 类型 | 定义 | 例子 | 文本层是否可分 |
|---|---|---|---|
| side-talk | 主用户对旁人说话 | "帮我把这个放桌上"（对同事） | 不可分（与指令同形） |
| self-talk | 自言自语 | "嗯……这个放哪来着" | 部分可分（无请求对象） |
| speaker-switch | 同一句话换说话人/距离/时间边界 | 同一句指令由旁人说 vs 穿戴者说 | **完全不可分** |

speaker-switch 是三者中最难的：**词不变、只有来源变**，任何纯文本模型都无能为力。

### 2.3 与现有 8 态 `reject` 的冲突（这是产品级问题）

现状：`user_reject` 定义为"听不清/不理解"，动作是**澄清**（"没听清，再说一遍"）。

冲突：sidetalk 场景下如果复用 `user_reject`，系统会对着一句对旁人说的话追问"没听清，再说一遍"——
这比不回应更糟。两者**动作互斥**：

| 状态 | 触发 | 动作 | 动作方向 |
|---|---|---|---|
| `user_reject` | 说了但没听清/听不懂 | 澄清 | **主动发话** |
| `sidetalk` | 听清了，但不是说给系统 | 静默 + 记上下文 | **完全不发话** |

**结论：不能复用，必须拆态。**

---

## 3. 2026 年三个关键进展

### 3.1 问题被重定义：从分类到序列路由【论文】

SAS / SDAR（arXiv:2604.08412，2026-04）：

| 项 | 内容 |
|---|---|
| 设定 | 预 ASR、端上部署：决定"这段音频要不要送去转写" |
| 核心主张 | 多说话人 + 时间歧义下，应建模为**基于交互历史的序列路由**，而非逐句分类 |
| 结果（60h 多说话人英文测试集） | 纯音频 F1 **0.86**（P 0.89 / R 0.83）；加摄像头 F1 **0.95**（P 0.97 / R 0.93） |
| 关键消融 | **去掉因果交互历史：F1 从 0.95 掉到 0.57（±0.03）** |

含义：**历史上下文是最大的单一信号**，比声学特征更值钱。这对只有单麦的手机/耳机形态尤其重要。

### 3.2 端到端的失败被量化【论文】

VGBench（arXiv:2609.32536，2026-09）：

| 项 | 内容 |
|---|---|
| 规模 | 1,018 条诊断样本 |
| 覆盖 | side-talk / self-talk / speaker-switch |
| 动作空间 | 静默 / 调工具 / 正常回答（action-level addressedness） |
| 关键设计 | speaker-switch 配对：**词完全相同**，只改来源、距离渲染、时间边界 |
| 主要发现 | 6 个原生 Audio LLM + 3 种免训练适配：**常能选对工具，但很少选择闭嘴**；最高 switch 静默率仅 **14%** |
| 后训练案例（VoxGate） | 监督训练后静默 switch 命令 **91.3%**；GRPO 阶段 side-talk 准确率 68.4% → **70.9%**，self-talk 静默 52.0% → **60.0%** |

含义：sidetalk 失败不是"听不清"，而是"**不愿意不回答**"——这是训练目标问题，不是 ASR 精度问题。

### 3.3 数据与赛事在补位【论文/官方】

| 来源 | 规模 | 与 sidetalk 的关系 |
|---|---|---|
| SLT 2026 SmartGlasses Challenge（arXiv:2608.12034） | **中文**（Mandarin，已核实）、106 小时四通道 egocentric、714 段会话；两条赛道（双人对话理解 / 多人会议理解），评 TSA-ASR + SLU | 多人重叠仍是主要瓶颈 |
| HumDial（ICASSP 2026） | 真实演员录制，非合成混合 | 拒识最好 **72.2%**，基线 **35.2%**；**除 Cascade（30.9% vs 28.1%，两项均极低）外均低于打断** |
| WearVox（Meta, arXiv:2601.02391） | 3,842 条多通道眼镜录音，5 任务 | Side-Talk Rejection 任务；多通道把拒识从 **85.6% → 93.9%**（同基座单/多通道模型对照，属案例研究非通用 SOTA） |
| CoDeTT（arXiv:2603.25434） | 18,000 条决策 / 14 类场景 / 300+ 小时中英双语 | 两种系统状态各 1,000 条 sidetalk；提出 SMR 指标 |
| ChorusDuplex-SideTalk | 100 对话 / 1,667 轮 / 134 条 side_talk / 3.4h（每变体） | 中文专做旁对话；TTS 合成；CC-BY-4.0 |

---

## 4. 业界产品与模型全景

这一章回答"业界到底谁在做、怎么做"，按**四层**铺开：模型层（端到端实时语音模型）、基础设施层（判停/拒识组件）、产品层（消费与企业）、以及横向对比。只收有官方文档、论文或可验证事实的条目。

### 4.1 四层地图

| 层 | 回答的问题 | 本文收录 | 代表 |
|---|---|---|---|
| A 模型层 | 模型自己能不能决定"不说" | 12 个 | GPT-Live、Gemini Live、Qwen-Audio-3.1-Realtime、Gander、Seeduplex、Step-Audio 2、GLM-Realtime |
| B 基础设施层 | 判停/拒识怎么被工程化 | 7 个 | LiveKit 轮次检测、Pipecat Smart Turn v3、ElevenLabs、Deepgram、AssemblyAI、OpenAI semantic_vad、Silero |
| C 产品层 | 真实产品怎么兜底 | 9 个 | ChatGPT Voice、Gemini for Home、Alexa、Siri/Apple UAD、AirPods、Ray-Ban Meta、AI Pin、车机音区、Teams |
| D 国内落地 | 车机与客服怎么工程化 | 5 个 | 车机多音区、免唤醒可见即可说、车载语音 A 级认证、阿里云智能外呼、火山豆包实时语音 |
| E 结论 | 共性 / 分歧 / 空白 | — | 见 §4.8 |

### 4.2 A 层：端到端实时语音模型（12 个）

| # | 模型 / 产品 | 厂商 | 拒识与时机机制 | 公开数据 | 证据 |
|---|---|---|---|---|---|
| 1 | **GPT-Live-1 / 1 mini / 1.1** | OpenAI（2026-07-08 起） | **全双工**；会用 "mhmm / yeah" 附和，也会在用户思考时**保持安静**；复杂任务后台委派给前沿模型（发布时 GPT-5.5），等待期间继续对话 | 官方描述，未公开静默率 | 官方发布（2026-07-08；1.1 增 13 音色、减少 "checking…"；07-31 起加 SynthID 水印） |
| 2 | GPT-Realtime-2 | OpenAI | Realtime API：`server_vad` / `semantic_vad`、`interrupt_response` | 作为基线出现在 Qwen 报告（OAB 87.31 / VoiceBench 83.37 / AMC 50.33） | 官方文档 + 论文 |
| 3 | Gemini Live（Gemini 3.x Flash / for Home） | Google | 支持打断；Continued Conversation 免唤醒续聊 | 未公开拒识率 | 官方支持页 |
| 4 | Qwen-Audio-3.1-Realtime | 阿里 | `smart_turn` 语义轮次（"嗯/啊"不打断）+ **说话人增强**屏蔽旁人 | FD-Bench v1.5 对背景语音响应率 73%→13% | 官方文档 + arXiv:2609.25176 |
| 5 | Qwen3.5-Omni / Qwen3.8 Omni Flash Realtime | 阿里 | 端到端，支持视频输入（眼镜/座舱形态） | 未公开 | 官方文档 + qwen-audio-agent 前端支持 |
| 6 | Seeduplex 1.2.6.1 / 3.0 | 字节 | 端到端全双工 | Qwen 报告基线：OAB 85.38 / VoiceBench 87.43 / AMC 41.91；人格 20+ 轮 3.75（高于 Qwen 3.1 的 3.46） | 论文（Qwen 报告 Table 2/8） |
| 7 | **Gander** | 腾讯混元（Apache-2.0 开源） | "先决策、后内容"解耦，时机训进模型 | FD-Bench v3：接管 100% / 抢话 8.0%（对照 GPT-Realtime 96%/13.5%、Gemini Live 3.1 接管 78%、级联抢话 33%） | 项目报告 |
| 8 | Step-Audio 2 / 2 mini | 阶跃星辰（开源） | 端到端语音理解 + 生成，支持中英日与粤语、情感 | URO Bench 开源端到端最高（自报） | 官方仓库 |
| 9 | GLM-Realtime | 智谱 | 音视频通话，2 分钟记忆，函数调用 | 音频 0.18 元/分钟、视频 1.2 元/分钟 | 官方文档 |
| 10 | MiniCPM-o 4.5 | OpenBMB | 全模态；Gander 的基座模型 | — | 项目报告 |
| 11 | Moshi | Kyutai（开源） | 全双工、可打断，本地部署 | 未公开拒识指标 | 官方仓库 |
| 12 | Baichuan-Audio | 百川（开源） | 端到端中英双语实时对话 | 未单独披露拒识/打断指标 | 官方技术文 |

### 4.3 B 层：判停与拒识的基础设施（7 个）

| # | 组件 | 提供方 | 机制 | 可调项 / 关键数字 | 证据 |
|---|---|---|---|---|---|
| 1 | **Turn detector model** | LiveKit Agents | 在 VAD 之上，用**语义 + 声学**预测"这一轮说完没有"，输出 `end_of_turn_probability`；另有 background voice cancellation、preemptive generation、agent 侧播报调度 | 四种模式：轮次检测模型（默认）/ 实时模型内置 / 纯 VAD / STT endpointing | 官方文档 |
| 2 | **Smart Turn v3** | Pipecat（开源） | 直接看**原始波形**（不是转写）判断是否说完，即"语义 VAD" | Whisper Tiny 编码器 + 线性头，8M 参数，8MB int8 ONNX；v3.2 基准 31,527 条、跨多语种 | 模型卡 |
| 3 | ElevenLabs Agents | ElevenLabs | `turn eagerness`（抢话积极度）+ 打断开关；新轮次系统用 Scribe v2 Realtime 信号 | 三档 eagerness；可关闭打断 | 官方文档 |
| 4 | Endpointing / UtteranceEnd | Deepgram | 按静音时长切句，并用词级时间戳做"utterance end" | `endpointing`(ms)、`utterance_end_ms` | 官方文档 |
| 5 | Turn Detection | AssemblyAI | 流式转写按"轮"输出，自带轮次边界 | 按 turn 交付，配 VAD | 官方文档 |
| 6 | `semantic_vad` 参数族 | OpenAI | 模型判断"话说完了吗"再切 | `eagerness`、`prefix_padding_ms`、`silence_duration_ms`、`create_response`、`interrupt_response`；降噪 `near_field` / `far_field` | 官方文档 |
| 7 | Silero VAD | 开源 | 纯声学、毫秒级 | 无语义能力，是上述方案的共同底座 | 开源项目 |

### 4.4 C 层：真实产品怎么兜底（9 个）

| # | 产品 | 形态 | 拒识 / 让路做法 | 结果或事故 | 证据 |
|---|---|---|---|---|---|
| 1 | ChatGPT Voice（GPT-Live） | 手机/桌面 | 全双工会附和也会沉默；复杂任务后台跑、前台继续聊 | 官方未公布静默率 | 官方发布 |
| 2 | Gemini for Home / Google Assistant | 音箱、屏 | Continued Conversation 免唤醒续聊；Gemini Live 支持打断 | Continued Conversation 在 Google Assistant 上仅美式英语 | 官方支持页 |
| 3 | Alexa（Follow-Up Mode） | 音箱 | DDSD：声学 + ASR + 语义/句法 + **上一轮话语** | EER 相对最佳基线改善 14% | ICASSP 论文 / 官方博客 |
| 4 | Siri + Apple UAD | 手机/手表/HomePod | 误唤醒抑制（FTM）：共享声学编码器 + 每种唤醒方式独立分类头 | 触控唤醒场景 FPR@99%TPR 相对改善 50% / 35% | Apple ML 论文 |
| 5 | AirPods Conversation Awareness | 耳机 | **让路**：检测佩戴者开口 → 降媒体音量、增强人声 | 无公开指标 | Apple 2023 发布 + 媒体实测 |
| 6 | Ray-Ban Meta（Hey Meta） | 眼镜 | 唤醒词门控 + Conversation focus（增强面对面人声） | **2026-10 事故**：无唤醒词也录音，用户发现后官方修复 | 官方发布说明 + Ars Technica（2026-10-16） |
| 7 | Humane AI Pin | 穿戴（无屏） | 以常听 + 触控为主，不做唤醒词 | **失败案例**：2025-02 停止销售、服务关停，HP 以 1.16 亿美元收购 | 多家媒体（2025-02） |
| 8 | 车机多音区（双音区→四音区） | 车载 | 波束成形 + 声源定位，按音区归属响应（副驾说"开窗"只开副驾窗） | 无统一指标 | 专利 CN109754803B + 行业分析 |
| 9 | Microsoft Teams / 会议 | 会议 | voice isolation 降噪 + **声纹注册做说话人归属**（speaker recognition） | 无公开拒识率 | 官方支持文档 |

### 4.5 国内车机：把"拒识"变成"音区归属"

| 做法 | 机制 | 证据 |
|---|---|---|
| 多音区波束成形 + 声源定位 | 麦克风阵列按方向锁声，双音区扩到四音区；按音区决定执行主体（副驾说"开窗"只开副驾窗） | 专利 CN109754803B + 行业分析【专利/行业】 |
| 免唤醒 + 可见即可说 | 取消每句唤醒词，改用音区 + 场景约束 + 界面可点区域约束一起缩小误收范围 | 媒体横评（2025–2026 问界 / 理想 / 蔚来 多篇）【媒体】 |
| 车载语音助手 A 级认证 | 中汽研华诚认证（CATARC）自 2022 年起做车载语音助手第三方认证 | 认证机构与媒体报道【媒体】 |

**判断**：车机的解法不是"判这句话对谁说"，而是**用空间把 addressee 物理限定**——四音区一开，旁人说话天然落在别的音区。这是单麦手机做不到的，也是我们不该抄的部分。
另外，国内车厂**均未公开判停/拒识的技术参数**，只有体验层评价；对标时只能记"能力有无"，不能记数字。

### 4.6 国内客服与外呼：把"拒识"变成"优雅打断"

| 厂商 / 产品 | 机制（官方功能项） | 可验证参数 | 证据 |
|---|---|---|---|
| 阿里云智能外呼机器人（语音&VUI） | **静音检测**（VAD 阈值）做断句；**优雅打断**："音义一体"，基于对比学习融合语音与语义，实现语义打断、噪音过滤、**背景人声过滤**；**智能断句**：结合人类说话行为特征与每句语音/语义特征，区分"句中停顿"与"说完了"；**语气承接**（Beta）：回答前播口语化承接语；情绪识别（语音+语义） | 静音检测一般 **600–800ms**，上限 2000ms；过长会导致响应变慢 | 官方文档（2026-10 核对） |
| 火山引擎 豆包实时语音（Seeduplex） | 端到端全双工实时语音对话；RealtimeAPI 支持中文与英语、WebSocket 流式；Dialog 语音对话 SDK 内置 AEC（同时开录音与播放时消回声） | 语种 2（中/英）；协议 WebSocket | 官方文档片段 |

**三条判断**

1. **客服把"拒识"拆成三个可配置件**：静音阈值（判停）、优雅打断（噪音 + 背景人声过滤）、智能断句（语义完整度）。这与我们 §7.2 的"判定信号优先级"同构，可直接拿来做参数设计参照
2. **客服默认静音阈值 600–800ms，比一般助手更激进**（更快接话），代价是误判上升，所以必须配"优雅打断"兜底——**快和准要靠两个件配合，不是调一个参数**
3. **"语气承接"已被产品化成降延迟体感的标准件**（Beta 阶段），说明我们做的垫词不是自选动作，而是行业标配；差别只在"模板池 vs 现场生成"

### 4.7 三张横向对比

**（一）模型层：谁真的会"闭嘴"**

| 能力 | 有明确机制 | 有公开数字 | 备注 |
|---|---|---|---|
| 语义判停（说完没有） | GPT-Live、Gemini Live、Qwen、Gander、Step-Audio | 部分 | 已是标配 |
| 拒绝旁人语音（sidetalk） | Qwen（说话人增强）、Gander（时机内化） | Qwen：背景语音响应 73%→13% | **只有阿里给了量化** |
| 主动附和（backchannel） | GPT-Live（"mhmm/yeah"）、Gander（Filler 51.6%） | Gander 有 | 附和多≠体验好，Gander Filler 偏高被列为弱项 |
| 沉默等待（用户思考时不抢话） | GPT-Live（"stay quiet"） | 无 | 全靠产品体感 |
| 前后台并行（边等边聊） | GPT-Live（委派前沿模型）、Qwen Voice Harness | Qwen Harness：91.04% 路由成功率 | 两家都这么做，说明是方向 |

**（二）基础设施层：判停的四代做法**

| 代 | 做法 | 代表 | 代价 |
|---|---|---|---|
| 一 | 纯 VAD 静音阈值 | Silero、厂商 VAD | 说慢了就误判说完 |
| 二 | STT endpointing | Deepgram、AssemblyAI | 受转写质量牵制 |
| 三 | 语义/声学融合模型 | LiveKit turn detector、Pipecat Smart Turn v3、OpenAI semantic_vad、阿里 smart_turn | 需要模型与语种覆盖 |
| 四 | 训进端到端模型 | Gander、GPT-Live、Qwen 3.1 | 最难调、最不可解释 |

**（三）产品层：判为"旁人说"之后怎么办**

| 动作 | 代表 | 适用形态 |
|---|---|---|
| 静默（不说话） | Qwen / Gander 的模型内决策 | 手机、耳机、眼镜 |
| 让路（降音量/暂停） | AirPods Conversation Awareness、Ray-Ban Conversation focus | 耳机、眼镜（用户正在听/正面对面聊） |
| 归属到人（按音区/声纹） | 车机四音区、Teams 说话人归属 | 车载、会议 |
| 轻确认（追问一次） | 通用 B 档做法 | 客服、任务型场景 |

### 4.8 结论：共性、分歧、空白与动作

**四条共性**

1. **都吃上一轮上下文**：Alexa 的 DDSD 明确编码前一句；与 SAS 消融"去掉历史 F1 从 0.95 掉到 0.57"互证
2. **都有兜底动作**：静默、让路、归属、轻确认——没有产品把"二分类硬判"当终点
3. **"是否对设备说"与"是否说完了"是两个判断**：OpenAI 的 `turn_detection` 只管说完，拒识靠模型或上层；阿里把两者合进 `smart_turn`
4. **快和准靠两个件配合，不是一个参数**：客服把静音阈值压到 600–800ms 换速度，同时用"优雅打断"（噪音 + 背景人声过滤 + 语义判断）兜误判——我们调判停阈值时必须成对调门控

**两处必须选边的分歧**

| 分歧 | 选项 A | 选项 B | 我们的选择 |
|---|---|---|---|
| 判为旁人说之后 | 静默 | 让路（降音量） | **静默 + 可回看**（我们是手机/耳机，让路会打断用户自己的播放） |
| 门控放哪 | 训进模型（Gander / 阿里） | 参数 + 外挂（OpenAI、LiveKit、Pipecat） | **外挂门控 + 参数化契约**（用云 API 没有权重） |

**四个空白（产品侧真没人做好）**

| 空白 | 说明 | 对我们的含义 |
|---|---|---|
| sidetalk 长时态 | 旁人连聊 30 秒以上如何重新锚定，无任何产品公开方案 | 只能自建评测 |
| 多人交替对系统说 | 会议里 A、B 轮流提问 | 行业仍靠唤醒词 |
| 电视 / 儿童噪声 | Alexa 明说 TV 与小孩喊叫是最难负例 | 负例集必须单列这两类 |
| 误拒指标 | 无一家公开误拒率，只报准确率或 EER | 自己定 FAR/FRR 工作点，否则无法对标 |

**四条动作**

1. **判定输入补"上一轮 + 场景"**：我们当前只看当前段音频，与所有玩家都不一致
2. **输出改三档**（对设备说 / 旁人说 / 不确定），不确定走轻确认，不硬判
3. **把语义轮次与说话人锁定做成可替换接缝**：与现有 `VadJudge` 同构，云 API 的 `smart_turn`、LiveKit 的 turn detector、Pipecat Smart Turn 可互换
4. **对标表补"误唤醒 / 误拒"口径**：并加一条产品事故维度（Ray-Ban 2026-10 无唤醒词录音、AI Pin 关服）

---

## 5. 基准与数据对照

### 5.1 六个资源横向对照

| 资源 | 语言 | 真实/合成 | 规模 | 提供什么指标 |
|---|---|---|---|---|
| HumDial Track II | 中/英（演员） | 真实录制 | 拒识 4 子类；整条赛道测试 5,000 条（其中拒识 2,000 条） | 拒识率、早期打断率、首响应延迟 |
| CoDeTT | 中/英 | 真实+合成 | 18,000 决策 | 4 动作准确率、**SMR（语义错配率）** |
| WearVox | 英 | 真实（AI 眼镜） | 3,842 条多通道 | 各任务准确率（含 Side-Talk Rejection） |
| VGBench | 英 | 合成+受控 | 1,018 条 | 静默率、工具选择、回答质量 |
| ChorusDuplex-SideTalk | 中文 | 合成（TTS） | 1,667 轮 | 逐轮 speaker attribution + 是否该静默 |
| SLT 2026 SmartGlasses | 中文 | 真实（四通道） | 106h / 714 段 | TSA-ASR + SLU |

### 5.2 怎么用（角色分工）

| 角色 | 用哪个 | 用法 |
|---|---|---|
| 淘汰/对标 | HumDial、WearVox、SLT 2026 | 判断自家系统在真实录制上是否明显落后 |
| 诊断 | CoDeTT（SMR）、VGBench（静默率） | 判断错在"没识别出"还是"识别了但不肯闭嘴" |
| 训练数据 | ChorusDuplex-SideTalk + 自建 | 中文合成集只够跑通链路，上产品需补真实录音 |

---

## 6. 已解决 vs 未解决

### 6.1 一张表看清现状

| 已解决 | 未解决 |
|---|---|
| 多通道/可穿戴的 sidetalk 拒识可用（93.9%，WearVox 案例研究口径） | 单通道（手机/耳机）只能到 F1 0.86 量级 |
| 交互历史是强信号（消融 +38 点） | 端到端模型默认不肯静默（14%） |
| 评测能区分"为什么没说"（SMR） | 误拒的代价未被建模（把指令当 sidetalk 的惩罚） |
| 后训练可显著改善（91.3%） | 中文真实数据缺，现有为合成 |
| 音视频融合可到 0.95 | sidetalk 持续性 / 结束后的重锚，无基准 |

### 6.2 对选型的含义

| 结论 | 含义 |
|---|---|
| 单通道上限约 F1 0.86 | 手机/耳机形态不要按"零误判"设计，需要"轻确认"兜底 |
| 端到端默认不肯静默 | 静默能力必须由独立门控保证，不能指望模型自觉 |
| 历史上下文是最大杠杆 | 判定时必带最近 3–5 轮，单句判定注定不准 |
| 中文真实数据缺 | 自建数据是必选项，不是可选项 |

---

## 7. 对 8 态 VAD 的改造建议

### 7.1 状态表：8 态 → 10 态

| 状态 | 含义 | 动作 | 相对现状 |
|---|---|---|---|
| complete | 说完了 | 回复 | 不变 |
| incomplete | 没说完 | 等待续说 | 不变 |
| backchannel | 附和 | 不打断、不应声 | 不变 |
| noise | 环境音 | 丢弃 | 不变 |
| tts_echo | 回声 | 忽略 | 不变 |
| barge_in_intentional | 有意打断 | 停播 + 接管 | 不变 |
| barge_in_unintentional | 无意打断 | 忽略 | 不变 |
| **user_reject** | 听不清/听不懂 | **澄清** | 语义收窄（只保留"没听清"） |
| **sidetalk**（新） | 对旁人说 | **静默 + 记上下文** | 新增，动作与 reject 相反 |
| **self_talk**（新） | 自言自语 | 静默（可选：低置信时轻确认一次） | 新增 |

### 7.2 判定信号优先级

| 优先级 | 信号 | 来源 | 说明 |
|---|---|---|---|
| 1 | 交互历史 | 会话状态 | SAS 消融证明最大杠杆（+38 点） |
| 2 | 称呼/第二人称 | ASR 文本 | "帮我/你/小X" vs 无称呼 |
| 3 | 说话人变化 | 声纹/多通道方向 | 单通道得不到，多通道可得 |
| 4 | 声学距离感 | 直达/混响比 | 远场更可能是旁对话 |
| 5 | 语义请求性 | 文本 | 有明确请求对象才像指令 |

**注意顺序**：文本语义排最后，因为 sidetalk 与指令同形（VGBench 的 speaker-switch 就是词完全相同）。

### 7.3 数据需求

| 类别 | 最小条数 | 说明 |
|---|---|---|
| sidetalk | 500 | 双人/三人 + 快递员/电话等真实打断源；现有中文合成集仅 134 条 |
| self_talk | 200 | 自言自语、嘟囔 |
| user_reject | 300 | 听不清/噪声掩盖（现有） |
| 近邻负例 | 每类 1:1 | 同一句话"对系统说 vs 对旁人说"配对，这是最值钱的样本 |

数据量用统一公式：`N = K × n_min × f_scene × f_neg`，语音信道系数取 2.0–3.0，负例系数 2.0。

### 7.4 服务化要点

| 项 | 建议 |
|---|---|
| 判定位置 | L1（声学+称呼粗判）→ L2（语义+历史细判），不交给 L3 |
| 静默时的上下文 | 不要直接丢：sidetalk 里常有后续要用的信息（"等会儿提醒我拿快递"） |
| 持续性 | 旁对话常连续 3–5 轮，需"维持静默"状态机，而不是每轮重判 |
| 重锚 | 旁对话结束后如何接回主线，交给 L3 的转述能力 |

---

## 8. 评测指标

### 8.1 基础三项

| 指标 | 定义 | 目标 |
|---|---|---|
| 拒识率（Rejection Rate） | sidetalk 样本中系统保持静默的比例 | ≥90% |
| 误拒率（False Rejection） | 真实指令被判成 sidetalk 的比例 | ≤2%（比拒识率更重要） |
| 静默下的打断率 | 本应静默却开始说话的比例 | ≤3% |

### 8.2 理由正确性：SMR【论文】

CoDeTT 的 Semantic Misalignment Rate：**拒识对了但理由错了也算错**
（把指令当噪声而静默，与识别出"这是旁对话"而静默，是两件事）。

自建实现：标注每条样本的"正确理由"（sidetalk / self_talk / noise / backchannel），
系统输出理由与标注不一致即计 SMR。目标 <10%。

### 8.3 产品口径：FAR / FRR 工作点

F1 高不代表能用。产品要的是**在给定误拒预算下的拒识率**，因此报告：

| 报告项 | 说明 |
|---|---|
| FRR@FAR=1% | 误拒 1% 时能拒掉多少 sidetalk |
| 曲线而非单点 | 与 SAS/VGBench 同类报告方式对齐 |
| 分类别 | sidetalk / self_talk / speaker-switch 分开报 |

### 8.4 持续性指标（自建，业界空白）

| 指标 | 定义 |
|---|---|
| 静默保持率 | 连续 N 轮旁对话中，系统保持静默的轮次比例 |
| 重锚延迟 | 旁对话结束 → 系统重新对主用户发话的间隔 |

---

## 9. 风险与开放问题

### 9.1 风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| 复用 reject 态 | sidetalk 触发澄清话术 | 拆态（§7.1） |
| 误拒指令 | 把真指令当旁对话，用户认为"没反应" | 误拒率设 ≤2%，并给"我在听"的低成本反馈 |
| 只靠文本判 | speaker-switch 场景必然失败 | 引入说话人/历史信号 |
| 单通道硬做 | F1 上限 ~0.86 | 明确产品边界，或用"轻确认"兜底 |
| 丢弃 sidetalk 内容 | 丢掉后续有用信息 | 静默但入上下文 |

### 9.2 开放问题

| # | 问题 | 为什么重要 |
|---|---|---|
| 1 | 单通道手机形态下，历史 + 称呼能把 sidetalk 做到多少 | 无公开数据，直接决定产品形态可行性 |
| 2 | 误拒与误收的代价函数如何定 | 拒错（不回应）与答错（乱答）哪个更伤，决定阈值 |
| 3 | 长时旁对话的边际收益递减点 | 决定状态机维持多久、何时可以忘掉 |
| 4 | 中文真实 sidetalk 数据何时出现 | 决定能否摆脱合成数据的分布偏差 |

---

## 10. 参考来源

### 10.1 论文与数据

1. arXiv:2604.08412《Selective Attention System (SAS)》——SDAR 序列路由；F1 0.86 / 0.95；去历史 0.95→0.57
2. arXiv:2609.32536《Do Audio LLMs Listen Before They Act?》——VGBench 1,018 条；静默率 14%；VoxGate 91.3%
3. arXiv:2608.12034《SLT 2026 SmartGlasses Challenge》——中文（Mandarin）、106h 四通道、714 段会话、TSA-ASR+SLU
4. arXiv:2601.02391《WearVox》——3,842 条多通道；sidetalk 拒识 85.6%→93.9%
5. arXiv:2603.25434《CoDeTT》——18,000 决策 / 14 类 / SMR
6. arXiv:2601.05564《ICASSP 2026 HumDial Challenge》——拒识四子类（URB/PH/TPS/SDO）；拒识 72.2% vs 基线 35.2%
7. arXiv:2507.05609《MMW: Side Talk Rejection Multi-Microphone Whisper on Smart Glasses》——**Whisper 指 ASR 模型架构**（不是"低语"）：多麦 Mix Block + Frame Diarization Mamba + GRPO，做穿戴者导向的 ASR 侧对话抑制；原版 Whisper 无法区分穿戴者与旁观者（WER 约 67%），MMW 降到 12%（无真值切分）/ 8%（有切分）量级【论文，小数位未取】
8. ChorusDuplex-SideTalk（HF, CC-BY-4.0）——中文 100 对话 / 134 条 side_talk
9. 本项目《四层架构方案》§2、《语义VAD方案》《全双工状态机设计》

### 10.2 产品侧公开材料

1. Apple Machine Learning Research《Less Is More: A Unified Architecture for Device-Directed Speech Detection》——UAD/FTM，FPR@99%TPR 相对改善 50% / 35%（2026-10 核对）
2. Amazon Science《How Alexa knows when you're talking to her》（ICASSP 论文配套博客）——DDSD 用语义/句法 + 声学 + 上一轮话语，EER 相对改善 14%
3. OpenAI Realtime API 文档《Voice activity detection》——`server_vad` / `semantic_vad`、`eagerness`、`create_response`、`interrupt_response`、`prefix_padding_ms`、`silence_duration_ms`；降噪 `near_field` / `far_field`
4. Google 支持页《Have a conversation with your Google smart speaker or display》——Gemini Live 支持打断；Continued Conversation（Assistant 仅美式英语）
5. 腾讯 Gander 技术报告与项目报告（本项目《腾讯Gander技术洞察报告》）——FD-Bench v3 接管 100% / 抢话 8.0%，对照 GPT-Realtime 96%/13.5%、Gemini Live 3.1 接管 78%、级联抢话 33%
6. 阿里云百炼《qwen-audio-realtime-user-guides》——`smart_turn` 语义轮次 + 说话人增强
7. 车机多音区：专利 CN109754803B《车载多音区语音交互系统及方法》+ 行业分析（双音区→四音区、波束成形、按音区响应）
8. Apple Newsroom《AirPods redefine the personal audio experience》（2023）+ 媒体实测——Conversation Awareness 降低媒体音量、增强人声
9. OpenAI《Introducing GPT-Live》（2026-07-08）——全双工、会附和也会沉默、复杂任务后台委派 GPT-5.5；1.1 版增 13 音色；07-31 起 SynthID 水印
10. LiveKit 文档《Turns overview》——轮次检测四模式（turn detector 模型 / 实时模型内置 / 纯 VAD / STT endpointing），`end_of_turn_probability`
11. Pipecat《Smart Turn v3》模型卡——波形输入、Whisper Tiny 编码器 + 线性头、8M 参数、8MB int8 ONNX、v3.2 基准 31,527 条跨多语种
12. ElevenLabs 文档——`turn eagerness` 与打断开关，新轮次系统用 Scribe v2 Realtime 信号
13. Deepgram《Endpointing》《Utterance End》、AssemblyAI《Turn Detection》——判停参数化
14. 阶跃星辰 Step-Audio / Step-Audio 2 mini 官方仓库——端到端、开源、支持中英日与粤语
15. 智谱《GLM-Realtime》官方文档——音视频通话、2 分钟记忆、0.18 元/分钟（音频）
16. Meta《AI glasses release notes》Conversation focus + Ars Technica（2026-10-16）《Ray-Ban Meta glasses found recording audio without wearer's consent》
17. Humane AI Pin 关停：2025-02 停止销售、服务关停，HP 以 1.16 亿美元收购（多家媒体）
18. Microsoft Teams《voice isolation》《voice and face recognition》——降噪 + 声纹注册做说话人归属
19. 阿里云《机器人语音交互能力配置（智能外呼机器人 语音&VUI）》——静音检测（VAD 阈值）600–800ms、上限 2000ms；优雅打断（语义打断 + 噪音过滤 + 背景人声过滤）；智能断句；语气承接（Beta）；情绪识别
20. 火山引擎《接入端到端实时语音大模型》《豆包端到端实时语音大模型 API》——Seeduplex 全双工；RealtimeAPI 中英双语、WebSocket 流式；Dialog SDK 内置 AEC
21. 车机侧：专利 CN109754803B + 行业分析 + 2025–2026 媒体座舱横评（问界/理想/蔚来语音体验）；中汽研华诚认证车载语音助手 A 级认证

### 10.3 数字核对记录（2026-10-27 复核，产品章节补充于 10-28）

| 论断/数字 | 出处 | 核对结果 |
|---|---|---|
| 纯音频 F1 0.86（P 0.89 / R 0.83） | arXiv:2604.08412 摘要 | 一致 |
| 音视频 F1 0.95（P 0.97 / R 0.93） | 同上 | 一致 |
| 去掉因果交互历史 F1 0.95 → 0.57±0.03 | 同上 | 一致 |
| VGBench 1,018 条、静默率最高 14%、VoxGate 91.3%、side-talk 68.4→70.9、self-talk 52.0→60.0 | arXiv:2609.32536 摘要 | 一致 |
| HumDial 拒识最好 72.2%、基线 35.2% | arXiv:2601.05564 表 3 | 一致 |
| "拒识全体低于打断" | 同上表 3 | **不成立**：Cascade 30.9 > 28.1（两项均极低）→ 已改为"除 Cascade 外" |
| 拒识四子类 URB/PH/TPS/SDO | 同上 §4 | 一致（原文：user real-time backchannels / pause handling / third-party speech / speech directed at others） |
| 拒识测试 2,000 条（赛道共 5,000） | 同上表 1（600/600/600/200） | 一致 |
| WearVox 3,842 条、85.6%→93.9%、61.9%→66.4% | arXiv:2601.02391 摘要与引言 | 一致；口径为同基座单/多通道案例研究 |
| CoDeTT 18,000 条 / 14 类 / SMR / 每状态 1,000 条 sidetalk | arXiv:2603.25434 表 1、§3 | 一致 |
| SLT 2026：106h、714 段会话、**中文** | arXiv:2608.12034 正文（出现 "Mandarin Chinese challenge"） | 一致（原文为 sessions，不是"段录音"） |
| ChorusDuplex：100 对话 / 1,667 轮 / 134 side_talk / 3.4h / CC-BY-4.0 | HF 数据卡 v2 | 一致 |
| MMW：原版 Whisper WER ≈67%、MMW 12%/8% | arXiv:2507.05609 正文 | 量级一致；**原文小数位未取**，故本文只写量级 |
| SAA：pre-ASR 托管层、fail closed、英文为主 | 厂商页（同团队 arXiv:2604.08412） | 与厂商自述一致，标注【官方】 |
| Apple UAD：FPR@99%TPR 相对改善 50% / 35% | Apple ML 论文摘要 | 一致（50% 对单输出模型、35% 对每类独立模型，场景为触控唤醒） |
| Alexa DDSD：EER 相对改善 14% | Amazon Science 博客（ICASSP 论文） | 一致（相对"最佳基线"，非绝对值） |
| Gander：接管 100% / 抢话 8.0%；GPT-Realtime 96%/13.5%；Gemini Live 3.1 接管 78%；级联抢话 33% | 本项目《腾讯Gander技术洞察报告》（引 FD-Bench v3） | 一致；均为 100 场景口径 |
| 阿里 `smart_turn` + 说话人增强；FD-Bench v1.5 背景语音响应 73%→13% | 百炼用户指南 + arXiv:2609.25176 摘要 | 一致 |
| OpenAI `semantic_vad` 参数族（`eagerness`/`interrupt_response`/`near_field`/`far_field`） | developers.openai.com《Voice activity detection》 | 参数名一致；`eagerness` 取值文档示例仅见 `auto`，未穷举全部档位 |
| GPT-Live：全双工、会附和也会沉默、委派 GPT-5.5、2026-07-08 发布 | OpenAI《Introducing GPT-Live》 | 一致（"stay quiet when you need a moment to think"原文）；静默率未公开 |
| LiveKit：轮次检测四模式，turn detector 为默认 | LiveKit《Turns overview》 | 一致 |
| Smart Turn v3：8M 参数 / 8MB int8 / v3.2 基准 31,527 条 | HF 模型卡 | 一致（模型卡原文） |
| Seeduplex 1.2.6.1：OAB 85.38 / VoiceBench 87.43 / AMC 41.91 / 20+ 轮 3.75 | Qwen 报告 Table 2/8 | 一致（同一协议下基线） |
| Step-Audio 2 mini：2025-09 开源，中英日 + 粤语 | 阶跃官方仓库 | 一致；"URO Bench 开源最高"为自报 |
| GLM-Realtime：音频 0.18 元/分钟 | 智谱官方文档 | 一致（另有视频档 1.2 元/分钟） |
| Ray-Ban Meta：2026-10 无唤醒词录音并修复 | Ars Technica 2026-10-16 等 | 一致（官方承认并修复） |
| Humane AI Pin：2025-02 停售关服、HP 1.16 亿美元收购 | 多家媒体 | 一致 |
| 阿里云外呼：静音检测 600–800ms（上限 2000ms）、优雅打断含背景人声过滤 | 阿里云官方文档（语音&VUI） | 一致（原文"一般设置为 600~800 毫秒之间，上限为 2000 毫秒"） |
| 火山 Seeduplex：全双工端到端、中英双语、WebSocket、SDK 内置 AEC | 火山引擎官方文档 | 一致（页面级核对；详细参数页为 JS 渲染，未取到打断相关项） |
| 车机：四音区/波束成形/按音区响应 | 专利 CN109754803B + 行业分析 | 机制一致；**国内车厂未公开判停/拒识参数**，仅媒体体验层可验证 |

**本轮修正的 5 处**：

1. "拒识全体低于打断" → "除 Cascade 外均低于打断"（Cascade 两项都极低，属特例）
2. "多麦低语场景" → 澄清 Whisper 指 ASR 模型架构，不是"低语"；并补 WER 量级
3. "714 段" → "714 段会话（sessions）"
4. "已有商用 SDK" → "pre-ASR 托管层（SAA）"，并补厂商自述的两条限制
5. HumDial 测试规模 → 明确"赛道 5,000 条，其中拒识 2,000 条"
